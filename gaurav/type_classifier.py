import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# ----------------------------------------------------
# CLASS LABELS
# ----------------------------------------------------
CLASS_NAMES = ["sar", "rgb", "falsecolor"]


# ----------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------
def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_NAMES))
    )

    state = torch.load(checkpoint_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------------------------------
# PREDICT FUNCTION
# ----------------------------------------------------
def predict(image_path, model, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# ----------------------------------------------------
# MAIN — CLI ARGUMENTS
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify SAR / RGB / False-Color images")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    pred_class, conf, full_probs = predict(args.image, model, device)

    print("\n====================================")
    print("📌 PREDICTION RESULT")
    print("====================================")
    print(f"Image Path     : {args.image}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence     : {conf:.4f}")
    print(f"All Probabilities: {full_probs}")


if __name__ == "__main__":
    main()
