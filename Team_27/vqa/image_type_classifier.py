import os
import logging
from typing import Optional

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# CLASS LABELS — keep same ordering as original
CLASS_NAMES = ["sar", "rgb", "falsecolor"]


class BandClassifier:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if checkpoint_path:
            try:
                self.model = self._load_model(checkpoint_path)
                logger.info("BandClassifier loaded checkpoint: %s", checkpoint_path)
            except Exception as e:
                logger.warning("Failed to load band classifier checkpoint '%s': %s", checkpoint_path, e)
                self.model = None

    def _load_model(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, len(CLASS_NAMES))
        )

        state = torch.load(checkpoint_path, map_location=self.device)
        # tolerate some key name changes
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    # Preprocessing same as training
    _preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def classify(self, image_path: str) -> Optional[str]:
        """Return one of CLASS_NAMES or None if classification unavailable."""
        if self.model is None:
            logger.debug("BandClassifier: model not loaded")
            return None
        if not os.path.exists(image_path):
            logger.debug("BandClassifier: image path does not exist: %s", image_path)
            return None

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug("BandClassifier: failed to open image %s: %s", image_path, e)
            return None

        img_tensor = BandClassifier._preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

        pred_idx = int(np.argmax(probs))
        return CLASS_NAMES[pred_idx]


def find_checkpoint_candidate() -> Optional[str]:
    """Try to find a reasonable checkpoint file in the workspace.
    Returns path or None.
    """
    candidates = [
        "band_classifier/checkpoint/best_model_3classes_450_all_data.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None
