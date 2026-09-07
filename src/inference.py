"""
inference.py
------------
Run the trained U-Net on every image inside inference_input/ and write a
colorized segmentation result for each one into inference_output/.

USAGE (from inside the src/ folder):

    python inference.py

That's it — no arguments needed. It automatically:
  1. Loads the model weights from ../weights/best_model.pth
  2. Reads every .jpg / .jpeg / .png file from ../inference_input/
  3. Runs each one through the model
  4. Saves a side-by-side [original | predicted mask | overlay] image for
     each input into ../inference_output/, using the same filename.

See the "Running Inference" section of the main README for the full
step-by-step setup guide (cloning the repo, downloading weights, etc).
"""

import os
import sys

import cv2
import numpy as np
import torch

import config
from model import UNet

# A fixed, distinct color for each of the 13 classes (RGB, 0-255).
# Purely for visualization -- has no effect on the model itself.
CLASS_COLORS = {
    0:  (30, 30, 30),      # background
    1:  (140, 90, 55),     # abdominal_wall
    2:  (178, 60, 60),     # liver
    3:  (230, 170, 60),    # gastrointestinal_tract
    4:  (240, 220, 130),   # fat
    5:  (80, 170, 200),    # grasper
    6:  (210, 130, 180),   # connective_tissue
    7:  (200, 30, 30),     # blood
    8:  (60, 130, 90),     # cystic_duct
    9:  (90, 100, 210),    # l_hook_electrocautery
    10: (60, 190, 90),     # gallbladder
    11: (160, 40, 130),    # hepatic_vein
    12: (110, 200, 210),   # liver_ligament
}

VALID_EXTS = (".jpg", ".jpeg", ".png")


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(weights_path):
        sys.exit(
            f"\nERROR: no weights file found at:\n  {weights_path}\n\n"
            "Download 'best_model.pth' from the Google Drive link in the "
            "README and place it in the 'weights/' folder at the repo "
            "root (next to inference_input/ and inference_output/), then "
            "run this script again.\n"
        )
    model = UNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    # Support both a raw state_dict and a full training checkpoint dict
    # (the one saved during training also stores optimizer state etc.).
    state_dict = checkpoint.get("model_state_dict", checkpoint) \
        if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(image_bgr: np.ndarray, device: torch.device):
    """Resize + normalize to match training preprocessing, return a batch tensor
    and the original image (resized to model input size, RGB) for visualization."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (config.IMG_WIDTH, config.IMG_HEIGHT),
                          interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    return tensor, resized


def mask_to_color(class_mask: np.ndarray) -> np.ndarray:
    h, w = class_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[class_mask == class_id] = color
    return color_mask


def run_inference():
    device = config.DEVICE
    print(f"Using device: {device}")

    os.makedirs(config.INFERENCE_OUTPUT_DIR, exist_ok=True)

    model = load_model(config.WEIGHTS_PATH, device)
    print(f"Loaded weights from {config.WEIGHTS_PATH}")

    if not os.path.isdir(config.INFERENCE_INPUT_DIR):
        os.makedirs(config.INFERENCE_INPUT_DIR, exist_ok=True)

    input_files = sorted(
        f for f in os.listdir(config.INFERENCE_INPUT_DIR)
        if f.lower().endswith(VALID_EXTS)
    )

    if not input_files:
        sys.exit(
            f"\nNo images found in:\n  {config.INFERENCE_INPUT_DIR}\n\n"
            "Drop some .jpg/.jpeg/.png surgical frames into that folder "
            "and run this script again.\n"
        )

    print(f"Found {len(input_files)} image(s) to process.\n")

    with torch.no_grad():
        for fname in input_files:
            in_path = os.path.join(config.INFERENCE_INPUT_DIR, fname)
            image_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"  [skip] could not read {fname}")
                continue

            tensor, resized_rgb = preprocess(image_bgr, device)
            logits = model(tensor)                          # (1, 13, H, W)
            pred_class = torch.argmax(logits, dim=1)[0]      # (H, W)
            pred_class = pred_class.cpu().numpy().astype(np.uint8)

            color_mask = mask_to_color(pred_class)
            overlay = cv2.addWeighted(resized_rgb, 0.55, color_mask, 0.45, 0)

            # Side-by-side: original | predicted mask | overlay
            combined = np.concatenate([resized_rgb, color_mask, overlay], axis=1)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

            out_name = os.path.splitext(fname)[0] + "_prediction.png"
            out_path = os.path.join(config.INFERENCE_OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, combined_bgr)
            print(f"  [done] {fname} -> {out_name}")

    print(f"\nAll results saved to: {config.INFERENCE_OUTPUT_DIR}")


if __name__ == "__main__":
    run_inference()
