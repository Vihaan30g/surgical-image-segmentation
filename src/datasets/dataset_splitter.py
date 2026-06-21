import random
from pathlib import Path
from collections import defaultdict


# =========================================
# DATASET ROOT
# =========================================

DATASET_ROOT = Path("data/archive")


# =========================================
# TEMPORAL STRIDE SETTINGS
# =========================================
#
# TRAIN and VAL use THE SAME stride.
# This ensures frame counts scale directly
# with clip counts, so 70/20 clip split
# produces roughly 70/20 frame split too.
#
# Why same stride for val?
# Val honesty comes from using UNSEEN CLIPS
# (guaranteed by clip-wise split), NOT from
# having sparser frames. A val clip that the
# model never trained on is already a rigorous
# test — we don't need to additionally thin it.
#
# TEST uses stride=1 (all frames) for maximum
# coverage in final evaluation. Test is only
# run once at the very end, not every epoch,
# so the larger frame count doesn't slow
# training.

TRAIN_STRIDE = 5    # 80 frames → 16 per clip
VAL_STRIDE   = 5    # same as train → honest proportional val set
TEST_STRIDE  = 1    # 80 frames → 80 per clip (full final eval)


# =========================================
# CLIP SPLIT RATIOS
# =========================================
#
# We split the 101 CLIPS (not videos) with
# these ratios. Frame counts will follow
# approximately the same ratios because
# train and val use the same stride.
#
# 70% train → ~70 clips → ~1120 frames
# 20% val   → ~21 clips → ~336  frames
# 10% test  → ~10 clips → ~800  frames (stride=1)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
# TEST gets the remainder (~0.10)


# =========================================
# RANDOM SEED
# =========================================
#
# Fixed so every run produces the identical
# split. Change only if you want to test a
# different random partition.

SPLIT_SEED = 42


# =========================================
# WHY CLIP-WISE SPLIT
# =========================================
#
# Each video (e.g. video01) has multiple
# clip subfolders (video01_00080,
# video01_00160, ...). We treat each clip
# as an independent unit and shuffle all
# 101 clips before splitting.
#
# Result: every video contributes clips to
# ALL THREE sets. The model sees every
# surgical environment during training.
# Val/test contain unseen temporal segments
# from known environments — not completely
# foreign visual domains. This eliminates
# the domain shift that caused flat val loss.


# =========================================
# STEP 1: DISCOVER ALL CLIPS
# =========================================

def discover_all_clips():
    """
    Returns list of (video_name, clip_folder_path)
    for every clip in the dataset.
    """

    all_clips = []

    video_folders = sorted([
        f for f in DATASET_ROOT.iterdir()
        if f.is_dir()
    ])

    for video_folder in video_folders:

        video_name = video_folder.name

        clip_folders = sorted([
            c for c in video_folder.iterdir()
            if c.is_dir()
        ])

        for clip_folder in clip_folders:

            all_clips.append(
                (video_name, clip_folder)
            )

    return all_clips


# =========================================
# STEP 2: SPLIT CLIPS
# =========================================

def split_clips(all_clips):
    """
    Shuffle all clips with fixed seed and
    split into train / val / test by ratio.

    Returns three lists of
    (video_name, clip_folder) tuples.
    """

    clips = list(all_clips)

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(clips)

    n = len(clips)

    train_end = int(TRAIN_RATIO * n)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * n)

    train_clips = clips[:train_end]
    val_clips   = clips[train_end:val_end]
    test_clips  = clips[val_end:]

    return train_clips, val_clips, test_clips


# =========================================
# STEP 3: COLLECT FRAMES FROM CLIPS
# =========================================

def collect_frames_from_clips(clip_list, stride):
    """
    Walk each clip folder, collect image+mask
    pairs, apply temporal stride.

    Returns:
        image_paths : list[Path]
        mask_paths  : list[Path]  (parallel)
        video_ids   : list[str]   (parallel)
            e.g. "video01" — used by
            VideoBalancedSampler to build
            cross-video training batches.
    """

    image_paths = []
    mask_paths  = []
    video_ids   = []

    for video_name, clip_folder in clip_list:

        clip_images = []
        clip_masks  = []

        image_files = sorted(
            clip_folder.glob("*_endo.png")
        )

        for image_path in image_files:

            if "mask" in image_path.name:
                continue

            mask_name = (
                image_path.stem
                + "_watershed_mask.png"
            )

            mask_path = (
                image_path.parent / mask_name
            )

            if mask_path.exists():
                clip_images.append(image_path)
                clip_masks.append(mask_path)

        # Apply temporal stride in frame order
        for i in range(0, len(clip_images), stride):
            image_paths.append(clip_images[i])
            mask_paths.append(clip_masks[i])
            video_ids.append(video_name)

    return image_paths, mask_paths, video_ids


# =========================================
# PUBLIC API
# =========================================

def create_splits():
    """
    Returns:
    {
        "train": (image_paths, mask_paths, video_ids),
        "val":   (image_paths, mask_paths, video_ids),
        "test":  (image_paths, mask_paths, video_ids),
    }
    """

    all_clips = discover_all_clips()

    train_clips, val_clips, test_clips = (
        split_clips(all_clips)
    )

    train_images, train_masks, train_vids = (
        collect_frames_from_clips(
            train_clips,
            stride=TRAIN_STRIDE
        )
    )

    val_images, val_masks, val_vids = (
        collect_frames_from_clips(
            val_clips,
            stride=VAL_STRIDE
        )
    )

    test_images, test_masks, test_vids = (
        collect_frames_from_clips(
            test_clips,
            stride=TEST_STRIDE
        )
    )

    return {
        "train": (train_images, train_masks, train_vids),
        "val":   (val_images,   val_masks,   val_vids),
        "test":  (test_images,  test_masks,  test_vids),
    }


# =========================================
# DEBUG
# =========================================

if __name__ == "__main__":

    all_clips = discover_all_clips()
    train_clips, val_clips, test_clips = split_clips(all_clips)

    print("\n===== CLIP SPLIT =====")
    print(f"Total : {len(all_clips)} clips")
    print(f"Train : {len(train_clips)} clips")
    print(f"Val   : {len(val_clips)} clips")
    print(f"Test  : {len(test_clips)} clips")

    def video_coverage(clips):
        d = defaultdict(int)
        for v, _ in clips:
            d[v] += 1
        return d

    for name, clips in [("TRAIN", train_clips), ("VAL", val_clips), ("TEST", test_clips)]:
        cov = video_coverage(clips)
        print(f"\n{name} — {len(cov)} videos represented:")
        for v, c in sorted(cov.items()):
            print(f"  {v}: {c} clips")

    splits = create_splits()

    BATCH = 8
    print("\n===== FRAME & BATCH COUNTS =====")
    for split_name, (imgs, _, vids) in splits.items():
        unique = len(set(vids))
        stride = TRAIN_STRIDE if split_name == "train" else (VAL_STRIDE if split_name == "val" else TEST_STRIDE)
        print(
            f"{split_name.upper():5s} : "
            f"{len(imgs):5d} frames | "
            f"{len(imgs)//BATCH:4d} batches | "
            f"{unique:2d} videos | "
            f"stride={stride}"
        )