"""intelligent mask merging using IoM-based graph algorithm"""

import numpy as np
import base64
import io
import math
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import colorsys

try:
    from scipy import ndimage as sp_ndimage
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when scipy missing
    sp_ndimage = None
    SCIPY_AVAILABLE = False

STRUCT_3x3 = np.ones((3, 3), dtype=bool)
STRUCT_5x5 = np.ones((5, 5), dtype=bool)

MAX_GEODESIC_DILATION_ITERS = 1024


def decode_mask(mask_png_b64: str) -> np.ndarray:
    """decode base64 png mask to binary numpy array"""
    mask_bytes = base64.b64decode(mask_png_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_array = np.array(mask_img) > 127
    return mask_array


def encode_mask(mask_array: np.ndarray) -> str:
    """encode binary numpy mask into base64 png"""
    mask_img = Image.fromarray(mask_array.astype('uint8') * 255, mode='L')
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('ascii')


def mask_to_box(mask_array: np.ndarray) -> List[float]:
    """compute normalized bounding box [x1, y1, x2, y2] from mask"""
    ys, xs = np.where(mask_array)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    h, w = mask_array.shape
    return [float(xs.min()) / w, float(ys.min()) / h,
            float(xs.max()) / w, float(ys.max()) / h]


def dilate_mask(mask_array: np.ndarray, iterations: int = 1) -> np.ndarray:
    """apply binary dilation with fallback when scipy is unavailable"""
    if iterations <= 0:
        return mask_array
    if SCIPY_AVAILABLE:
        return sp_ndimage.binary_dilation(mask_array, structure=STRUCT_3x3, iterations=iterations)
    dilated = mask_array.copy()
    for _ in range(iterations):
        padded = np.pad(dilated, 1, mode='constant', constant_values=False)
        expanded = np.zeros_like(dilated, dtype=bool)
        h, w = dilated.shape
        for dy in range(3):
            for dx in range(3):
                expanded = np.logical_or(expanded, padded[dy:dy + h, dx:dx + w])
        dilated = expanded
    return dilated


def close_mask(mask_array: np.ndarray, iterations: int = 1) -> np.ndarray:
    """binary closing with 5x5 structuring element"""
    if iterations <= 0:
        return mask_array
    if SCIPY_AVAILABLE:
        return sp_ndimage.binary_closing(mask_array, structure=STRUCT_5x5, iterations=iterations)
    # fallback: dilate then erode using simple loops
    closed = mask_array.copy()
    for _ in range(iterations):
        closed = dilate_mask(closed, iterations=1)
        # simple erosion via convolution-style check
        padded = np.pad(closed, 2, mode='constant', constant_values=False)
        eroded = np.zeros_like(closed, dtype=bool)
        h, w = closed.shape
        for y in range(h):
            for x in range(w):
                window = padded[y:y+5, x:x+5]
                if window.all():
                    eroded[y, x] = True
        closed = eroded
    return closed


def geodesic_dilate(marker: np.ndarray, mask: np.ndarray, max_iters: int = MAX_GEODESIC_DILATION_ITERS) -> Tuple[np.ndarray, int, bool]:
    """perform geodesic dilation of marker within mask"""
    current = np.logical_and(marker, mask)
    for iteration in range(1, max_iters + 1):
        dilated = dilate_mask(current, iterations=1)
        dilated = np.logical_and(dilated, mask)
        if np.array_equal(dilated, current):
            return current, iteration - 1, True
        current = dilated
    return current, max_iters, False


def compute_iom(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    compute intersection over minimum (IoM) area
    
    returns: iom in [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()
    
    min_area = min(area1, area2)
    
    if min_area == 0:
        return 0.0
    
    return float(intersection) / float(min_area)


def merge_iom_graph(
    masks1: List[Dict[str, Any]], 
    masks2: List[Dict[str, Any]], 
    iom_threshold: float = 0.9,
    coverage_threshold: float = 0.5,
    debug: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    IoM-based graph merge algorithm
    
    Algorithm:
    1. Combine both mask sets (model1 and model2)
    2. For each pair of masks, compute IoM on actual masks (not bboxes)
    3. If IoM is high, link smaller mask to larger mask (child -> parent)
    4. Build tree structures (each tree has one root)
    5. For each tree: if total coverage of leaves compared to root is good,
       discard the root (it's redundant), keep the leaves
    6. Otherwise keep the root
    
    Args:
        iom_threshold: IoM threshold for linking masks (default 0.5)
        coverage_threshold: if leaves cover >= this fraction of root, discard root
        debug: if True, return debug info with graph structure
    
    Returns:
        (merged_masks, debug_info)
    """
    # combine all masks with source tracking
    all_masks = []
    for i, m in enumerate(masks1):
        m_copy = m.copy()
        m_copy['_source'] = 'model1'
        m_copy['_original_id'] = i
        m_copy['_global_id'] = len(all_masks)
        all_masks.append(m_copy)
    
    for i, m in enumerate(masks2):
        m_copy = m.copy()
        m_copy['_source'] = 'model2'
        m_copy['_original_id'] = i
        m_copy['_global_id'] = len(all_masks)
        all_masks.append(m_copy)
    
    # decode all masks
    decoded_masks = []
    for m in all_masks:
        mask_array = decode_mask(m['png'])
        decoded_masks.append((m, mask_array, mask_array.sum()))
    
    n = len(decoded_masks)
    
    # build adjacency graph: child -> parent links
    # child is the smaller mask, parent is the larger mask
    parent_map = {}  # child_id -> parent_id
    children_map = defaultdict(list)  # parent_id -> [child_ids]
    
    iom_matrix = np.zeros((n, n))
    
    for i in range(n):
        mask_i, array_i, area_i = decoded_masks[i]
        for j in range(i + 1, n):
            mask_j, array_j, area_j = decoded_masks[j]
            
            # compute IoM on actual masks
            iom = compute_iom(array_i, array_j)
            iom_matrix[i, j] = iom
            iom_matrix[j, i] = iom
            
            if iom >= iom_threshold:
                # link smaller to larger
                if area_i < area_j:
                    child_id, parent_id = i, j
                else:
                    child_id, parent_id = j, i
                
                # only link if child doesn't already have a parent
                # (prefer linking to closest/most overlapping parent)
                if child_id not in parent_map:
                    parent_map[child_id] = parent_id
                    children_map[parent_id].append(child_id)
    
    # find roots (nodes with no parent)
    roots = []
    for i in range(n):
        if i not in parent_map:
            roots.append(i)
    
    # build trees: for each root, collect all descendants (leaves)
    def get_all_descendants(node_id):
        """recursively get all leaf descendants"""
        if node_id not in children_map:
            # this is a leaf
            return [node_id]
        
        descendants = []
        for child_id in children_map[node_id]:
            descendants.extend(get_all_descendants(child_id))
        return descendants
    
    # decision: for each root with children, check if leaves cover the root well
    discarded_roots = set()
    kept_leaves = set()
    leaf_keep_reason: Dict[int, str] = {}
    derived_masks: List[Dict[str, Any]] = []
    coverage_stats: Dict[int, Dict[str, Any]] = {}
    
    for root_id in roots:
        if root_id in children_map:
            # root has children
            root_mask, root_array, root_area = decoded_masks[root_id]
            leaf_ids = get_all_descendants(root_id)
            
            # remove root itself from leaves
            leaf_ids = [lid for lid in leaf_ids if lid != root_id]
            
            if len(leaf_ids) == 0:
                continue
            
            # compute union of all (closed) leaves
            leaf_union = np.zeros_like(root_array, dtype=bool)
            closed_leaf_arrays: Dict[int, np.ndarray] = {}
            for lid in leaf_ids:
                closed_leaf = close_mask(decoded_masks[lid][1], iterations=1)
                closed_leaf_arrays[lid] = closed_leaf
                leaf_union = np.logical_or(leaf_union, closed_leaf)
            
            # compute coverage: what fraction of root is covered by leaf union
            intersection = np.logical_and(root_array, leaf_union).sum()
            coverage = intersection / root_area if root_area > 0 else 0.0
            
            coverage_stats[root_id] = {
                'raw': coverage,
                'dilated': coverage,
                'dilation_iters': 0,
                'converged': True,
            }

            if coverage >= coverage_threshold:
                # leaves cover root well -> discard root, keep leaves
                discarded_roots.add(root_id)
                kept_leaves.update(leaf_ids)
                for lid in leaf_ids:
                    leaf_keep_reason[lid] = "leaf_covers_root_well"
                
                if debug:
                    print(f"Tree root {root_id} (source={decoded_masks[root_id][0]['_source']}, area={root_area:.0f}): "
                          f"covered {coverage:.2f} by {len(leaf_ids)} leaves -> DISCARD root, KEEP leaves")
            else:
                if len(leaf_ids) == 1:
                    # single leaf with poor coverage: discard leaf, keep root
                    if debug:
                        print(f"Tree root {root_id} (source={decoded_masks[root_id][0]['_source']}, area={root_area:.0f}): "
                              f"covered {coverage:.2f} by 1 leaf -> KEEP root, DISCARD leaf")
                    continue
                
                # multi-leaf scenario: keep leaves and carve remainder from root
                discarded_roots.add(root_id)
                kept_leaves.update(leaf_ids)
                dilated_union, iters_used, converged = geodesic_dilate(
                    leaf_union, root_array, max_iters=MAX_GEODESIC_DILATION_ITERS
                )
                coverage_after_dilation = (
                    np.logical_and(root_array, dilated_union).sum() / root_area
                    if root_area > 0 else 0.0
                )
                coverage_stats[root_id] = {
                    'raw': coverage,
                    'dilated': coverage_after_dilation,
                    'dilation_iters': iters_used,
                    'converged': converged,
                }

                if coverage_after_dilation >= coverage_threshold:
                    reason = "leaf_covers_root_after_dilation"
                    for lid in leaf_ids:
                        leaf_keep_reason[lid] = reason
                    remainder_area = 0
                    remainder_mask = None
                else:
                    for lid in leaf_ids:
                        leaf_keep_reason[lid] = "leaf_retained_low_coverage"
                    remainder_mask = np.logical_and(root_array, np.logical_not(dilated_union))
                    # Defensive step: ensure remainder does not overlap any kept leaf's original mask
                    kept_original_union = np.zeros_like(root_array, dtype=bool)
                    for lid in leaf_ids:
                        kept_original_union = np.logical_or(kept_original_union, decoded_masks[lid][1])
                    overlap_with_kept = np.logical_and(remainder_mask, kept_original_union).sum()
                    if overlap_with_kept > 0:
                        # remove any overlap to guarantee disjointness
                        remainder_mask = np.logical_and(remainder_mask, np.logical_not(kept_original_union))
                        if debug:
                            print(f"Adjusted remainder for root {root_id}: removed {overlap_with_kept} overlapping pixels with kept leaves")
                    remainder_area = remainder_mask.sum()
                    if remainder_area > 0:
                        remainder_dict: Dict[str, Any] = {}
                        for key, val in decoded_masks[root_id][0].items():
                            if key.startswith('_') or key == 'png':
                                continue
                            remainder_dict[key] = val
                        base_id = remainder_dict.get('id', f"root_{root_id}")
                        remainder_dict['id'] = f"{base_id}_remainder"
                        remainder_dict['png'] = encode_mask(remainder_mask)
                        remainder_dict['box'] = mask_to_box(remainder_mask)
                        remainder_dict['source'] = decoded_masks[root_id][0]['_source']
                        remainder_dict['area'] = int(remainder_area)
                        remainder_dict['reason'] = 'root_minus_leaves'
                        derived_masks.append(remainder_dict)

                if debug:
                    print(
                        f"Tree root {root_id} (source={decoded_masks[root_id][0]['_source']}, area={root_area:.0f}): "
                        f"covered {coverage:.2f} raw / {coverage_after_dilation:.2f} post-dilation by {len(leaf_ids)} leaves -> "
                        + ("DISCARD root, KEEP leaves" if coverage_after_dilation >= coverage_threshold else f"KEEP leaves, CARVE remainder (area {remainder_area:.0f})")
                    )
    
    # build final result
    result = []
    for i in range(n):
        mask_dict, mask_array, area = decoded_masks[i]
        
        # keep if:
        # 1. it's a root and not discarded, OR
        # 2. it's a leaf that was explicitly kept
        is_root = i not in parent_map
        is_discarded_root = i in discarded_roots
        is_kept_leaf = i in kept_leaves
        
        keep = False
        reason = ""
        
        if is_root and not is_discarded_root:
            keep = True
            reason = "root_without_good_leaves" if i in children_map else "unique_mask"
        elif is_kept_leaf:
            keep = True
            reason = leaf_keep_reason.get(i, "leaf_covers_root_well")
        
        if keep:
            mask_copy = mask_dict.copy()
            mask_copy['source'] = mask_dict['_source']
            mask_copy['area'] = int(area)
            mask_copy['reason'] = reason
            # remove internal tracking fields
            for key in ['_source', '_original_id', '_global_id']:
                mask_copy.pop(key, None)
            result.append(mask_copy)

    # append any derived remainder masks once at the end
    result.extend(derived_masks)
    
    # prepare debug info
    debug_info = None
    if debug:
        debug_info = {
            'iom_matrix': iom_matrix,
            'parent_map': parent_map,
            'children_map': dict(children_map),
            'roots': roots,
            'discarded_roots': list(discarded_roots),
            'kept_leaves': list(kept_leaves),
            'leaf_keep_reason': leaf_keep_reason,
            'coverage_stats': coverage_stats,
            'derived_masks': [
                {
                    'id': m.get('id'),
                    'source': m.get('source'),
                    'area': m.get('area'),
                    'reason': m.get('reason')
                }
                for m in derived_masks
            ],
            'all_masks_info': [(m['_global_id'], m['_source'], m['_original_id'], area) 
                               for m, _, area in decoded_masks]
        }
    
    return result, debug_info


def merge2(
    masks1: List[Dict[str, Any]], 
    masks2: List[Dict[str, Any]], 
    iom_threshold: float = 0.5,
    coverage_threshold: float = 0.5,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Wrapper for backward compatibility - calls merge_iom_graph
    """
    result, _ = merge_iom_graph(masks1, masks2, iom_threshold, coverage_threshold, debug)
    return result


def visualize_masks_overlay(
    image_path: str,
    masks: List[Dict[str, Any]],
    output_path: str,
    title: str = "Masks",
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None
):
    """
    Visualize actual mask overlays (not just bounding boxes)
    
    Args:
        image_path: path to original image
        masks: list of mask dicts with 'png' field
        output_path: where to save visualization
        title: title for this visualization
        color_map: optional dict mapping mask index to RGB color
    """
    img = Image.open(image_path).convert("RGBA")
    result = img.copy()
    
    # create overlay for each mask with different colors
    for idx, mask_data in enumerate(masks):
        try:
            mask_array = decode_mask(mask_data['png'])
            mask_pil = Image.fromarray(mask_array.astype('uint8') * 255, mode='L')
            
            # assign color
            if color_map and idx in color_map:
                color = color_map[idx]
            else:
                # generate distinct colors using HSV
                hue = (idx * 137) % 360  # golden angle for good color separation
                rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
                color = tuple(int(c * 255) for c in rgb)
            
            # create colored overlay
            colored_mask = Image.new("RGBA", img.size, color + (120,))  # semi-transparent
            result = Image.composite(colored_mask, result, mask_pil)
        except Exception as e:
            print(f"Warning: failed to overlay mask {idx}: {e}")
    
    result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    
    # draw labels at mask centroids
    for idx, mask_data in enumerate(masks):
        try:
            mask_array = decode_mask(mask_data['png'])
            ys, xs = np.where(mask_array)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"{idx+1}"
                
                # draw background for text
                bbox = draw.textbbox((cx, cy), label, font=font)
                draw.rectangle(bbox, fill="white")
                draw.text((cx, cy), label, fill="black", font=font)
        except Exception:
            pass
    
    result.save(output_path)
    print(f"Saved {title} mask overlay to {output_path}")


def visualize_merge_decision(
    image_path: str,
    masks1: List[Dict[str, Any]],
    masks2: List[Dict[str, Any]],
    merged: List[Dict[str, Any]],
    debug_info: Optional[Dict[str, Any]] = None,
    output_path: str = "merge_debug.png"
):
    """
    Create comprehensive merge visualization with actual masks
    
    Saves 3 images:
    1. model1 masks only
    2. model2 masks only
    3. merged result with color coding by source
    """
    base_path = output_path.replace(".png", "")
    
    # visualize model1 masks
    visualize_masks_overlay(
        image_path, masks1, 
        f"{base_path}_model1.png",
        "Model1 (no box guidance)"
    )
    
    # visualize model2 masks
    visualize_masks_overlay(
        image_path, masks2,
        f"{base_path}_model2.png",
        "Model2 (with box guidance)"
    )
    
    # visualize merged masks with unique color for each mask
    img = Image.open(image_path).convert("RGBA")
    result = img.copy()
    
    for idx, mask_data in enumerate(merged):
        try:
            mask_array = decode_mask(mask_data['png'])
            mask_pil = Image.fromarray(mask_array.astype('uint8') * 255, mode='L')
            
            # assign unique color to each mask using HSV
            hue = (idx * 137) % 360  # golden angle for good color separation
            rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
            color = tuple(int(c * 255) for c in rgb)
            
            colored_mask = Image.new("RGBA", img.size, color + (120,))  # semi-transparent
            result = Image.composite(colored_mask, result, mask_pil)
        except Exception:
            pass
    
    result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    
    # draw labels
    for idx, mask_data in enumerate(merged):
        try:
            mask_array = decode_mask(mask_data['png'])
            ys, xs = np.where(mask_array)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                source = mask_data.get('source', '?')
                # show mask number and source in label
                label = f"{idx+1}"
                
                bbox = draw.textbbox((cx, cy), label, font=font)
                draw.rectangle(bbox, fill="white")
                draw.text((cx, cy), label, fill="black", font=font)
        except Exception:
            pass
    
    result.save(f"{base_path}_merged.png")
    print(f"Saved merged visualization to {base_path}_merged.png")
    
    # print merge graph if debug info available
    if debug_info:
        print("\n" + "="*60)
        print("MERGE GRAPH STRUCTURE")
        print("="*60)
        
        parent_map = debug_info.get('parent_map', {})
        children_map = debug_info.get('children_map', {})
        roots = debug_info.get('roots', [])
        discarded_roots = debug_info.get('discarded_roots', [])
        kept_leaves = debug_info.get('kept_leaves', [])
        all_masks_info = debug_info.get('all_masks_info', [])
        
        print(f"\nTotal masks: {len(all_masks_info)}")
        print(f"  - Model1: {sum(1 for _, src, _, _ in all_masks_info if src == 'model1')}")
        print(f"  - Model2: {sum(1 for _, src, _, _ in all_masks_info if src == 'model2')}")
        
        print(f"\nTree roots: {len(roots)}")
        print(f"  - Discarded roots: {len(discarded_roots)}")
        print(f"  - Kept leaves: {len(kept_leaves)}")
        
        print("\nTree structures:")
        for root_id in roots:
            gid, src, oid, area = all_masks_info[root_id]
            is_discarded = root_id in discarded_roots
            status = "DISCARDED" if is_discarded else "KEPT"
            
            print(f"\n  Root {gid} ({src}[{oid}], area={area:.0f}) [{status}]")
            
            if root_id in children_map:
                def print_tree(node_id, indent=4):
                    if node_id in children_map:
                        for child_id in children_map[node_id]:
                            gid_c, src_c, oid_c, area_c = all_masks_info[child_id]
                            is_kept = child_id in kept_leaves
                            status_c = "KEPT" if is_kept else "discarded"
                            print(" " * indent + f"└─ Child {gid_c} ({src_c}[{oid_c}], area={area_c:.0f}) [{status_c}]")
                            print_tree(child_id, indent + 4)
                
                print_tree(root_id)
        
        print("\n" + "="*60)