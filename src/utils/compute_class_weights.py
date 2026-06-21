"""
Run this ONCE before training to compute exact class weights
from your actual dataset pixel distribution.

Usage:
    python src/utils/compute_class_weights.py

Paste the printed CLASS_WEIGHTS list into combined_loss.py
to replace the estimated values.
"""

import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

from configs.class_mapping import RAW_TO_CLASS

DATASET_ROOT = Path("data/archive")
NUM_CLASSES  = 13


def compute_weights():

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    mask_files = list(DATASET_ROOT.rglob("*_watershed_mask.png"))

    print(f"Scanning {len(mask_files)} mask files...")

    for i, mask_path in enumerate(mask_files):

        if i % 200 == 0:
            print(f"  {i}/{len(mask_files)}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        class_mask = np.zeros_like(mask)
        for raw_val, class_id in RAW_TO_CLASS.items():
            class_mask[mask == raw_val] = class_id

        for c in range(NUM_CLASSES):
            pixel_counts[c] += (class_mask == c).sum()

    total = pixel_counts.sum()
    freqs = pixel_counts / total

    # Inverse frequency, normalize to mean=1
    # Add small epsilon to avoid div by zero for absent classes
    raw_w = 1.0 / (freqs + 1e-6)
    weights = raw_w / raw_w.mean()

    print("\n===== CLASS PIXEL DISTRIBUTION =====\n")
    from configs.class_mapping import CLASS_NAMES
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        pct  = freqs[c] * 100
        w    = weights[c]
        print(f"  {c:2d} {name:22s}: {pct:6.3f}%  weight={w:.4f}")

    print("\n===== PASTE INTO combined_loss.py =====\n")
    print("CLASS_WEIGHTS = [")
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        print(f"    {weights[c]:.4f},   # {c:2d} {name}")
    print("]")


if __name__ == "__main__":
    compute_weights()
