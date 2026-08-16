"""
compute_class_weights.py
-------------------------
Scans the training set's ground-truth masks, counts total pixels per class,
and computes normalized inverse-frequency weights for use in FocalLoss /
DiceLoss. Run once, paste the printed CLASS_WEIGHTS list into config.py.

Usage:
    !python compute_class_weights.py
"""

import numpy as np
import torch
from tqdm import tqdm

import config
import splits
from dataset import CholecSegDataset, remap_watershed_mask
import cv2
import os


def count_pixels(clip_list):
    counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    dataset = CholecSegDataset(
        root_dir=config.DATA_ROOT, clip_list=clip_list,
        frame_step=config.TRAIN_FRAME_STEP, transforms=None,
    )
    print(f"Scanning {len(dataset)} training samples for class pixel counts...")
    for record in tqdm(dataset.samples):
        raw_mask_bgr = cv2.imread(record["mask_path"], cv2.IMREAD_COLOR)
        raw_mask = raw_mask_bgr[:, :, 0]
        class_mask = remap_watershed_mask(raw_mask)
        unique, class_counts = np.unique(class_mask, return_counts=True)
        for u, c in zip(unique, class_counts):
            if u < config.NUM_CLASSES:
                counts[u] += c
    return counts


pixel_counts = count_pixels(splits.TRAIN_CLIPS)

print("\nRaw pixel counts per class:")
for cid in range(config.NUM_CLASSES):
    print(f"  [{cid:>2}] {config.CLASS_NAMES[cid]:<28s}: {pixel_counts[cid]:>14,}")

# Inverse-frequency weighting, normalized so mean weight = 1.0
total = pixel_counts.sum()
freq = pixel_counts / total
# add small epsilon to avoid div-by-zero for any class with 0 pixels found
inv_freq = 1.0 / (freq + 1e-8)
# normalize so the weights average to 1.0 (keeps overall loss magnitude stable)
weights = inv_freq * (config.NUM_CLASSES / inv_freq.sum())

# Cap extreme weights so the rarest class doesn't completely dominate/destabilize
# training (common practice — without this, a class with 0.01% of pixels can get
# a weight in the thousands, which blows up gradients).
MAX_WEIGHT = 20.0
weights = np.clip(weights, a_min=None, a_max=MAX_WEIGHT)

print("\nComputed class weights (paste into config.py):")
print("CLASS_WEIGHTS = [")
for cid in range(config.NUM_CLASSES):
    print(f"    {weights[cid]:.4f},  # [{cid}] {config.CLASS_NAMES[cid]}")
print("]")