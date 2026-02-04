import os
from groundingdino.util.inference import load_model
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("Loading model...")
model = load_model("GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

class VRSDataset(Dataset):
    def __init__(self, json_path, image_root, use_computed_bbox=True, target_size=(800, 1333)):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.target_width, self.target_height = target_size
        self.use_computed_bbox = use_computed_bbox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_root, sample["image"])
        caption = sample["caption"]

        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        image = image.resize((self.target_width, self.target_height), Image.BILINEAR)

        bboxes = []
        labels = []
        areas = []

        for obj in sample["objects"]:
            bbox_key = "bbox_aabb_computed" if self.use_computed_bbox else "bbox_aabb" # bbox is the rounded off computed_bbox
            x_min, y_min, x_max, y_max = obj[bbox_key]

            # Not sure about the x, y
            x_min = x_min * self.target_width
            y_min = y_min * self.target_height
            x_max = x_max * self.target_width
            y_max = y_max * self.target_height

            width = x_max - x_min
            height = y_max - y_min
            area = width * height

            bboxes.append([x_min, y_min, width, height])
            labels.append(obj["class"])
            areas.append(area)

        annotations = {
            "image_id": idx,
            "caption": caption,
            "annotations": [
                {
                    "bbox": torch.tensor(bboxes[i], dtype=torch.float32),
                    "area": float(areas[i]),
                    "category": i,
                    "iscrowd": 0,
                }
                for i in range(len(bboxes))
            ],
        }

        return image, annotations

json_path = "/home/samyak/scratch/interiit/data/VRSBench/val.json"
image_root = "/home/samyak/scratch/interiit/data/VRSBench/Images_val"

dataset = VRSDataset(
    json_path=json_path,
    image_root=image_root,
    use_computed_bbox=True,
    target_size=(800, 1333)
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda x: x
)

# def visualize_sample(image, annotations):
#     fig, ax = plt.subplots(1, figsize=(8, 6))
#     ax.imshow(image)
#     ax.set_title(annotations["caption"], fontsize=10)

#     for ann in annotations["annotations"]:
#         bbox = ann["bbox"]
#         if isinstance(bbox, torch.Tensor):
#             bbox = bbox.tolist()
#         x, y, w, h = bbox
#         rect = patches.Rectangle(
#             (x, y), w, h,
#             linewidth=2,
#             edgecolor='red',
#             facecolor='none'
#         )
#         ax.add_patch(rect)
#         label = ann["category"]
#         ax.text(x, y - 5, label, color='yellow', fontsize=8, weight='bold')

#     plt.savefig("sample_vis.png")


# for batch in loader:
#     image, annotations = batch[0] 
#     print(type(image))
#     print(annotations["image_id"])
#     print(annotations["caption"])
#     print(annotations["annotations"])
#     visualize_sample(image, annotations)
#     break 
