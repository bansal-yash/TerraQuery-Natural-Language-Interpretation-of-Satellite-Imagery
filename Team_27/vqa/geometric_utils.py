"""
Geometric utilities for computing properties from segmentation masks.

Functions to compute:
- Orientation (angle of major axis)
- Aspect ratio
- Area (in square pixels)
- Centroid position
- Distance between masks
- Bounding box dimensions
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional


def compute_mask_properties(mask: np.ndarray) -> Dict[str, float]:
    """Compute geometric properties of a segmentation mask.
    
    Args:
        mask: Binary mask (H, W) with True/1 for object pixels
        
    Returns:
        Dictionary with:
            - area: Area in square pixels
            - centroid_x, centroid_y: Center position
            - orientation: Angle of major axis in degrees (-90 to 90)
            - aspect_ratio: Ratio of major to minor axis
            - bbox_width, bbox_height: Bounding box dimensions
            - perimeter: Perimeter length in pixels
    """
    if mask.sum() == 0:
        return {
            "area": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "orientation": 0.0,
            "aspect_ratio": 1.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "perimeter": 0.0,
        }
    
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Compute moments
    moments = cv2.moments(mask_uint8)
    
    # Area (in square pixels): count of foreground pixels
    # Note: OpenCV moments m00 equals sum of intensities (255 * pixel_count for our mask).
    # We report true pixel count for area, while still using m00 for centroid/orientation math.
    area_pixels = float(mask.astype(bool).sum())
    
    # Centroid
    m00 = moments['m00']
    if m00 > 0:
        centroid_x = moments['m10'] / m00
        centroid_y = moments['m01'] / m00
    else:
        centroid_x = centroid_y = 0.0
    
    # Orientation and aspect ratio using central moments
    # Normalize central moments by m00 (intensity-sum) for stable orientation/aspect
    mu20 = moments['mu20'] / m00 if m00 > 0 else 0
    mu02 = moments['mu02'] / m00 if m00 > 0 else 0
    mu11 = moments['mu11'] / m00 if m00 > 0 else 0
    
    # Orientation (angle of major axis)
    if mu20 != mu02:
        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        orientation_deg = np.degrees(orientation)
    else:
        orientation_deg = 0.0
    
    # Eigenvalues for aspect ratio (major/minor axis lengths)
    lambda1 = (mu20 + mu02) / 2 + np.sqrt(4 * mu11**2 + (mu20 - mu02)**2) / 2
    lambda2 = (mu20 + mu02) / 2 - np.sqrt(4 * mu11**2 + (mu20 - mu02)**2) / 2
    
    if lambda2 > 0:
        aspect_ratio = np.sqrt(lambda1 / lambda2)
    else:
        aspect_ratio = 1.0
    
    # Bounding box
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Compute bounding box over all contours (union)
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        bbox_width = float(w)
        bbox_height = float(h)
        # Sum perimeters of all contours
        perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    else:
        bbox_width = bbox_height = perimeter = 0.0
    
    return {
        "area": float(area_pixels),
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "orientation": float(orientation_deg),
        "aspect_ratio": float(aspect_ratio),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "perimeter": float(perimeter),
    }


import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_min_distance_between_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the minimum Euclidean distance between two binary masks.
    
    Distance is defined as the minimum distance between any foreground pixel
    in mask1 and any foreground pixel in mask2. If masks overlap, distance is 0.
    
    Args:
        mask1: First binary mask (non-zero = foreground)
        mask2: Second binary mask (same shape as mask1)
        
    Returns:
        Minimum Euclidean distance in pixels between the two masks.
        Returns np.inf if one of the masks has no foreground pixels.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")

    # Ensure boolean
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    # If either mask is empty, define distance as infinite
    if not m1.any() or not m2.any():
        return float(np.inf)

    # If they already overlap, distance is zero
    if np.any(m1 & m2):
        return 0.0

    # Distance transform from mask2: distance to nearest True in m2
    dt = distance_transform_edt(~m2)

    # Restrict to pixels in mask1 and take the minimum
    min_dist = dt[m1].min()

    return float(min_dist)




def get_relative_position(mask1: np.ndarray, mask2: np.ndarray) -> str:
    """Get relative position of mask2 with respect to mask1.
    
    Args:
        mask1: Reference mask
        mask2: Target mask
        
    Returns:
        String describing relative position (e.g., "right", "above", "below-left")
    """
    props1 = compute_mask_properties(mask1)
    props2 = compute_mask_properties(mask2)
    
    dx = props2["centroid_x"] - props1["centroid_x"]
    dy = props2["centroid_y"] - props1["centroid_y"]
    
    # Determine horizontal position
    if abs(dx) < 10:  # Threshold for "aligned"
        horizontal = ""
    elif dx > 0:
        horizontal = "right"
    else:
        horizontal = "left"
    
    # Determine vertical position
    if abs(dy) < 10:
        vertical = ""
    elif dy > 0:
        vertical = "below"
    else:
        vertical = "above"
    
    # Combine
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    elif vertical:
        return vertical
    elif horizontal:
        return horizontal
    else:
        return "same-position"


if __name__ == "__main__":
    # Test the utilities
    print("Testing geometric utilities...")
    
    # Create a simple test mask (rectangle)
    test_mask = np.zeros((100, 100), dtype=bool)
    test_mask[20:60, 30:80] = True  # 40x50 rectangle
    
    props = compute_mask_properties(test_mask)
    print("\nTest mask properties:")
    for key, value in props.items():
        print(f"  {key}: {value:.2f}")
    
    # Create another mask for distance testing
    test_mask2 = np.zeros((100, 100), dtype=bool)
    test_mask2[50:70, 10:30] = True
    
    distance = compute_min_distance_between_masks(test_mask, test_mask2)
    print(f"\nDistance between masks: {distance:.2f} pixels")
    
    position = get_relative_position(test_mask, test_mask2)
    print(f"Relative position: mask2 is {position} of mask1")
