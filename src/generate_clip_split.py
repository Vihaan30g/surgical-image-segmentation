"""
generate_clip_split.py
------------------------
Replaces whole-video splitting with CLIP-level splitting. Each video folder
contains multiple clip subfolders; clips are treated as the atomic
splitting unit (all frames of one clip stay together in one split, since
consecutive frames at 25fps are near-duplicates and splitting those apart
would leak information). But because clips are a much finer-grained unit
than whole videos, we can deliberately ensure rare classes get coverage in
BOTH train and val, instead of a whole class disappearing because its one
video landed entirely in val.

Strategy:
  1. Scan every clip's watershed masks, record which classes appear in it
     and how many frames it has.
  2. For each class (starting with the rarest), guarantee at least one
     clip containing it is reserved for TRAIN, and at least one (if a
     second distinct clip containing it exists) is reserved for VAL.
  3. All remaining unreserved clips are shuffled and assigned to hit
     target frame-count ratios (default 70/15/15).
  4. Prints a full coverage report and writes the result to splits.py,
     which config.py / dataset.py will import.

Usage:
    !python generate_clip_split.py
"""

import os
import random
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config
from dataset import remap_watershed_mask

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = config.SEED

ALL_VIDEOS = sorted(set(config.TRAIN_VIDEOS + config.VAL_VIDEOS + config.TEST_VIDEOS))


def index_clips(video_list):
    """Returns dict: 'videoXX/clipYY' -> list of watershed mask paths."""
    clips = {}
    for video_name in video_list:
        video_path = os.path.join(config.DATA_ROOT, video_name)
        if not os.path.isdir(video_path):
            print(f"WARNING: missing {video_path}")
            continue
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
            if mask_files:
                clip_key = f"{video_name}/{clip_dir}"
                clips[clip_key] = [os.path.join(clip_path, f) for f in mask_files]
    return clips


def scan_clip_class_presence(clips):
    """
    For each clip, scans ALL its frames (not subsampled — accurate
    presence detection matters for rare classes) and records which classes
    appear and their pixel counts.
    """
    clip_info = {}
    for clip_key, mask_paths in tqdm(clips.items(), desc="Scanning clips"):
        classes_present = set()
        class_pixel_counts = defaultdict(int)
        for mask_path in mask_paths:
            raw_mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if raw_mask_bgr is None:
                continue
            raw_mask = raw_mask_bgr[:, :, 0]
            class_mask = remap_watershed_mask(raw_mask)
            unique, counts = np.unique(class_mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u < config.NUM_CLASSES:
                    classes_present.add(int(u))
                    class_pixel_counts[int(u)] += int(c)
        clip_info[clip_key] = {
            "n_frames": len(mask_paths),
            "classes": classes_present,
            "class_pixel_counts": dict(class_pixel_counts),
        }
    return clip_info


def build_stratified_split(clip_info):
    rng = random.Random(SPLIT_SEED)

    class_to_clips = defaultdict(list)
    for clip_key, info in clip_info.items():
        for cid in info["classes"]:
            class_to_clips[cid].append(clip_key)

    train_clips, val_clips, test_clips = set(), set(), set()
    reserved = set()

    classes_by_rarity = sorted(
        range(config.NUM_CLASSES),
        key=lambda cid: len(class_to_clips.get(cid, []))
    )

    print("\n" + "=" * 78)
    print("Reserving clips to guarantee class coverage (rarest classes first)")
    print("=" * 78)

    for cid in classes_by_rarity:
        name = config.CLASS_NAMES[cid]
        candidates = list(class_to_clips.get(cid, []))
        rng.shuffle(candidates)

        if len(candidates) == 0:
            print(f"  [{cid:>2}] {name:<28s}: 0 clips in ENTIRE dataset — "
                  f"cannot fix via re-splitting, class is absent from the "
                  f"raw data itself. Skipping.")
            continue

        already_train = [c for c in candidates if c in train_clips]
        already_val = [c for c in candidates if c in val_clips]

        if not already_train:
            unused = [c for c in candidates if c not in reserved]
            pick_from = unused if unused else candidates
            if pick_from:
                chosen = pick_from[0]
                train_clips.add(chosen)
                reserved.add(chosen)

        if not already_val:
            unused = [c for c in candidates if c not in reserved]
            pick_from = unused if unused else [c for c in candidates if c not in train_clips]
            if pick_from:
                chosen = pick_from[0]
                val_clips.add(chosen)
                reserved.add(chosen)
            elif len(candidates) == 1:
                print(f"  [{cid:>2}] {name:<28s}: only 1 clip in the whole "
                      f"dataset contains this class — it went to TRAIN. "
                      f"VAL will have 0 examples of this class; unavoidable "
                      f"without more source data.")

        n_train = len([c for c in candidates if c in train_clips])
        n_val = len([c for c in candidates if c in val_clips])
        print(f"  [{cid:>2}] {name:<28s}: {len(candidates):>4} clips total "
              f"-> reserved {n_train} for train, {n_val} for val")

    remaining = [c for c in clip_info if c not in reserved]
    rng.shuffle(remaining)

    total_frames = sum(info["n_frames"] for info in clip_info.values())
    target_train_frames = TRAIN_RATIO * total_frames
    target_val_frames = VAL_RATIO * total_frames

    def frames_in(clip_set):
        return sum(clip_info[c]["n_frames"] for c in clip_set)

    for clip_key in remaining:
        current_train = frames_in(train_clips)
        current_val = frames_in(val_clips)

        if current_train < target_train_frames:
            train_clips.add(clip_key)
        elif current_val < target_val_frames:
            val_clips.add(clip_key)
        else:
            test_clips.add(clip_key)

    return train_clips, val_clips, test_clips


def print_coverage_report(split_name, clip_set, clip_info):
    total_frames = sum(clip_info[c]["n_frames"] for c in clip_set)
    print(f"\n[{split_name}] {len(clip_set)} clips, {total_frames} frames")
    header = f"{'class':<28s} {'clips w/ class':>15s} {'total px (proxy)':>20s}"
    print(header)
    print("-" * len(header))
    for cid in range(config.NUM_CLASSES):
        name = config.CLASS_NAMES[cid]
        clips_with = [c for c in clip_set if cid in clip_info[c]["classes"]]
        total_px = sum(clip_info[c]["class_pixel_counts"].get(cid, 0) for c in clips_with)
        flag = "  <-- MISSING from this split" if len(clips_with) == 0 else ""
        print(f"{name:<28s} {len(clips_with):>15d} {total_px:>20,}{flag}")


def main():
    print("Indexing all clips across all videos...")
    clips = index_clips(ALL_VIDEOS)
    print(f"Found {len(clips)} clips total.")

    print("\nScanning class presence per clip (reads every mask file, "
          "will take a few minutes)...")
    clip_info = scan_clip_class_presence(clips)

    train_clips, val_clips, test_clips = build_stratified_split(clip_info)

    print("\n" + "=" * 78)
    print("FINAL SPLIT COVERAGE REPORT")
    print("=" * 78)
    print_coverage_report("TRAIN", train_clips, clip_info)
    print_coverage_report("VAL", val_clips, clip_info)
    print_coverage_report("TEST", test_clips, clip_info)

    out_path = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), "splits.py")
    with open(out_path, "w") as f:
        f.write('"""\nAuto-generated by generate_clip_split.py — DO NOT EDIT BY HAND.\n')
        f.write("Clip-level (not whole-video-level) train/val/test split, stratified\n")
        f.write("to guarantee every class has coverage in train (and val, where the\n")
        f.write('raw data has enough distinct clips containing it).\n"""\n\n')
        f.write("TRAIN_CLIPS = " + repr(sorted(train_clips)) + "\n\n")
        f.write("VAL_CLIPS = " + repr(sorted(val_clips)) + "\n\n")
        f.write("TEST_CLIPS = " + repr(sorted(test_clips)) + "\n")

    print(f"\nWrote {out_path}")
    print("Next: update config.py to import from splits.py, and use the "
          "updated dataset.py (accepts clip_list now) + train.py.")


if __name__ == "__main__":
    main()