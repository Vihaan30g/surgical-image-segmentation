"""
diagnose_masks.py
------------------
Run this in Colab (same environment as train.py) BEFORE changing any
training code. It checks whether the raw pixel values actually found in
the watershed mask files match what config.RAW_TO_CLASS expects.

Usage (from your repo root, in Colab):
    !python diagnose_masks.py
"""

import os
from collections import Counter

import cv2
import numpy as np

import config


def scan_masks(video_list, max_files_per_video=20):
    all_values = Counter()
    per_video_values = {}

    for video_name in video_list:
        video_path = os.path.join(config.DATA_ROOT, video_name)
        if not os.path.isdir(video_path):
            print(f"MISSING: {video_path}")
            continue

        video_values = Counter()
        files_checked = 0

        clip_dirs = sorted(
            d for d in os.listdir(video_path)
            if os.path.isdir(os.path.join(video_path, d))
        )
        for clip_dir in clip_dirs:
            clip_path = os.path.join(video_path, clip_dir)
            mask_files = sorted(
                f for f in os.listdir(clip_path)
                if f.endswith(config.WATERSHED_MASK_SUFFIX)
            )
            for fname in mask_files:
                if files_checked >= max_files_per_video:
                    break
                mask_path = os.path.join(clip_path, fname)
                raw = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                if raw is None:
                    print(f"  Could not read: {mask_path}")
                    continue

                # Check whether channels are actually identical (grayscale
                # replicated) as the current code assumes.
                if not np.array_equal(raw[:, :, 0], raw[:, :, 1]) or \
                   not np.array_equal(raw[:, :, 1], raw[:, :, 2]):
                    print(f"  WARNING: channels differ (not grayscale-replicated) in {mask_path}")

                channel0 = raw[:, :, 0]
                unique, counts = np.unique(channel0, return_counts=True)
                for u, c in zip(unique, counts):
                    video_values[int(u)] += int(c)
                    all_values[int(u)] += int(c)

                files_checked += 1
            if files_checked >= max_files_per_video:
                break

        per_video_values[video_name] = video_values
        print(f"{video_name}: checked {files_checked} mask files, "
              f"unique raw values found: {sorted(video_values.keys())}")

    return all_values, per_video_values


print("=" * 70)
print("Scanning TRAIN_VIDEOS masks...")
print("=" * 70)
all_values, _ = scan_masks(config.TRAIN_VIDEOS, max_files_per_video=30)

print("\n" + "=" * 70)
print("Scanning VAL_VIDEOS masks...")
print("=" * 70)
val_values, _ = scan_masks(config.VAL_VIDEOS, max_files_per_video=30)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nAll unique raw pixel values found across sampled TRAIN masks:")
print(sorted(all_values.keys()))

print(f"\nconfig.RAW_TO_CLASS expects these raw values:")
print(sorted(config.RAW_TO_CLASS.keys()))

expected = set(config.RAW_TO_CLASS.keys())
found = set(all_values.keys())

matched = expected & found
missing = expected - found
unexpected = found - expected

print(f"\nMatched (expected AND found):   {sorted(matched)}")
print(f"Missing (expected but NOT found): {sorted(missing)}")
print(f"Unexpected (found but NOT in config): {sorted(unexpected)}")

print("\nPixel count per matched class (sanity check — should be nonzero")
print("and roughly proportionate to organ size, not dominated by 1-2 values):")
for raw_val in sorted(matched):
    class_id = config.RAW_TO_CLASS[raw_val]
    name = config.CLASS_NAMES[class_id]
    print(f"  raw={raw_val:>3} -> class {class_id:>2} ({name:<25s}): {all_values[raw_val]:>12,} px")

if missing:
    print("\n*** DIAGNOSIS: your RAW_TO_CLASS mapping does NOT match the actual")
    print("*** mask files. This explains why most classes show as 'absent'.")
    print("*** Compare the 'unexpected' values above against what's actually")
    print("*** in the files — those unexpected raw values are likely your")
    print("*** real class pixel values, just under a different numbering.")
else:
    print("\nMapping matches file contents. If classes still show 'absent' in")
    print("validation, the issue is likely that VAL_VIDEOS genuinely lack")
    print("certain classes as a matter of chance (small video sample) — check")
    print("if TRAIN masks have all classes but VAL's 3 videos don't.")