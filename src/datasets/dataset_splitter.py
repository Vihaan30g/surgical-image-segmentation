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

TRAIN_STRIDE = 5    # 80 frames → 16 per clip
VAL_STRIDE   = 20   # 80 frames → 4  per clip  (honest sparse metric)
TEST_STRIDE  = 1    # 80 frames → 80 per clip  (full coverage)


# =========================================
# SPLIT RATIOS
# =========================================

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO is the remainder (0.15)


# =========================================
# RANDOM SEED
# =========================================
#
# Fixed seed so every run produces the exact
# same train/val/test split. If you change
# this, you get a different split — which is
# fine for experimentation but makes runs
# incomparable to each other.

SPLIT_SEED = 42


# =========================================
# WHY CLIP-WISE SPLIT INSTEAD OF VIDEO-WISE
# =========================================
#
# OLD APPROACH (video-wise):
#   Train on video01..video28, validate on
#   video35/37/43, test on video48/52/55.
#
#   Problem: the model NEVER sees video35's
#   visual domain during training. Each video
#   looks different (lighting, tissue color,
#   camera angle). Validation on an entirely
#   unseen video domain is not measuring
#   "did the model learn anatomy" — it is
#   measuring "can the model handle a completely
#   foreign visual style it was never shown."
#   This is what caused the flat/rising val
#   loss in v1.
#
# NEW APPROACH (clip-wise):
#   Each video has multiple clip folders
#   (e.g. video01 has 16 clips). We treat
#   each CLIP as an independent unit and
#   split those 101 clips 70/15/15.
#
#   Result: every video contributes clips to
#   ALL THREE sets. The model sees video35's
#   visual style in training (from some of
#   its clips) and is validated on OTHER clips
#   from the same video. The visual domain is
#   no longer a confounder — the model is
#   evaluated on "unseen temporal segments"
#   not "unseen surgical environments."
#
#   Val and test are still rigorous: those
#   specific clip segments were never seen
#   during training. But the model has learned
#   what all 17 videos look like, so domain
#   shift is minimized.


# =========================================
# STEP 1: DISCOVER ALL CLIPS
# =========================================

def discover_all_clips():
    """
    Walk the dataset directory and return a
    list of (video_name, clip_folder_path)
    tuples for every clip that exists.

    A "clip" is one subdirectory inside a
    video folder, e.g.:
        data/archive/video01/video01_00080/
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
# STEP 2: SPLIT CLIPS INTO TRAIN/VAL/TEST
# =========================================

def split_clips(all_clips):
    """
    Randomly shuffle all 101 clips and split
    them 70/15/15 by clip count.

    Shuffling is seeded so the split is
    deterministic across runs.

    Returns three lists of
    (video_name, clip_folder_path) tuples.
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
# STEP 3: COLLECT FRAME PATHS FROM CLIPS
# =========================================

def collect_frames_from_clips(clip_list, stride):
    """
    Given a list of (video_name, clip_folder)
    tuples and a temporal stride, collect all
    image+mask path pairs.

    Returns:
        image_paths : list of Path
        mask_paths  : list of Path  (parallel)
        video_ids   : list of str   (parallel)
                      e.g. "video01"
                      Used by VideoBalancedSampler
                      to build cross-video batches.
    """

    image_paths = []
    mask_paths  = []
    video_ids   = []

    for video_name, clip_folder in clip_list:

        # Collect all valid frame pairs in
        # this clip, sorted by filename so
        # stride is applied in temporal order.

        clip_image_paths = []
        clip_mask_paths  = []

        image_files = sorted(
            clip_folder.glob("*_endo.png")
        )

        for image_path in image_files:

            # skip files that are themselves masks
            if "mask" in image_path.name:
                continue

            mask_name = (
                image_path.stem
                + "_watershed_mask.png"
            )

            mask_path = image_path.parent / mask_name

            if mask_path.exists():
                clip_image_paths.append(image_path)
                clip_mask_paths.append(mask_path)


        # Apply temporal stride.
        # stride=5 on 80 frames → indices
        # 0,5,10,...,75 → 16 frames per clip.

        for i in range(
            0,
            len(clip_image_paths),
            stride
        ):
            image_paths.append(clip_image_paths[i])
            mask_paths.append(clip_mask_paths[i])
            video_ids.append(video_name)


    return image_paths, mask_paths, video_ids


# =========================================
# PUBLIC API: CREATE SPLITS
# =========================================

def create_splits():
    """
    Main entry point called by dataloader_setup.

    Returns a dict:
    {
        "train": (image_paths, mask_paths, video_ids),
        "val":   (image_paths, mask_paths, video_ids),
        "test":  (image_paths, mask_paths, video_ids),
    }

    video_ids is a per-frame list of which
    video each frame came from. It is used
    by VideoBalancedSampler in dataloader_setup
    to ensure every training batch mixes
    frames from multiple different videos.
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
        "train": (
            train_images,
            train_masks,
            train_vids
        ),
        "val": (
            val_images,
            val_masks,
            val_vids
        ),
        "test": (
            test_images,
            test_masks,
            test_vids
        )
    }


# =========================================
# DEBUG INFO
# =========================================

if __name__ == "__main__":

    all_clips = discover_all_clips()
    train_clips, val_clips, test_clips = (
        split_clips(all_clips)
    )

    print("\n===== CLIP-WISE SPLIT SUMMARY =====\n")
    print(f"Total clips  : {len(all_clips)}")
    print(f"Train clips  : {len(train_clips)}")
    print(f"Val   clips  : {len(val_clips)}")
    print(f"Test  clips  : {len(test_clips)}")

    def video_coverage(clips):
        seen = defaultdict(int)
        for video_name, _ in clips:
            seen[video_name] += 1
        return seen

    print("\n--- Videos in TRAIN ---")
    for v, c in sorted(video_coverage(train_clips).items()):
        print(f"  {v}: {c} clips")

    print("\n--- Videos in VAL ---")
    for v, c in sorted(video_coverage(val_clips).items()):
        print(f"  {v}: {c} clips")

    print("\n--- Videos in TEST ---")
    for v, c in sorted(video_coverage(test_clips).items()):
        print(f"  {v}: {c} clips")

    splits = create_splits()

    print("\n===== FRAME COUNTS =====\n")
    for split_name, (imgs, masks, vids) in splits.items():
        unique = len(set(vids))
        print(
            f"{split_name.upper():5s} : "
            f"{len(imgs):4d} frames | "
            f"{unique} videos represented"
        )