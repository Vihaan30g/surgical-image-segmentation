from pathlib import Path


# =========================================
# DATASET ROOT
# =========================================

DATASET_ROOT = Path("data/archive")


# =========================================
# VIDEO-WISE SPLITS
# =========================================

TRAIN_VIDEOS = [
    "video01",
    "video09",
    "video12",
    "video17",
    "video18",
    "video20",
    "video24",
    "video25",
    "video26",
    "video27",
    "video28"
]

VAL_VIDEOS = [
    "video35",
    "video37",
    "video43"
]

TEST_VIDEOS = [
    "video48",
    "video52",
    "video55"
]


# =========================================
# TEMPORAL STRIDE SETTINGS
# =========================================
#
# Each clip folder has 80 consecutive frames
# captured at 25 FPS. Frame N and Frame N+1
# are nearly identical — they differ only by
# tiny movement of the surgeon's hand. Training
# on every frame means the model sees thousands
# of near-duplicate images, making it easy to
# "memorize" each clip's visual style.
#
# TRAIN_STRIDE = 10:
#   Keep every 10th frame from each clip.
#   80 frames → 8 frames per clip.
#   Those 8 frames are spread ~0.4 seconds
#   apart — visually distinct enough to count
#   as meaningfully different samples.
#   This also speeds up training significantly.
#
# VAL_STRIDE = 20:
#   Keep every 20th frame from each val clip.
#   80 frames → 4 frames per clip.
#   The goal for validation is a HONEST metric,
#   not coverage. Near-identical val frames
#   inflate Dice score artificially — if the
#   model gets frame 40 right, it trivially gets
#   frames 39 and 41 right too. With stride 20
#   the val set is small but trustworthy.
#
# TEST_STRIDE = 1:
#   Keep all test frames. At test time we want
#   full coverage to measure real-world perf.

TRAIN_STRIDE = 10

VAL_STRIDE = 20

TEST_STRIDE = 1


# =========================================
# COLLECT IMAGE AND MASK PATHS
# =========================================

def collect_image_mask_paths(
    video_list,
    stride=1
):
    """
    Walk video_list → clip folders → frames.
    Apply temporal stride to reduce near-duplicates.

    Returns two parallel lists:
        image_paths[i] and mask_paths[i]
        always refer to the same frame.

    Also returns video_id_per_frame: a list of
    the same length where entry i is the name
    of the video that frame i came from.
    This is used by VideoBalancedSampler in
    dataloader_setup.py to build cross-video
    batches.
    """

    image_paths = []
    mask_paths = []
    video_ids = []


    for video_name in video_list:

        video_path = DATASET_ROOT / video_name

        clip_folders = sorted(video_path.iterdir())


        for clip_folder in clip_folders:


            # =================================
            # COLLECT ALL FRAMES IN THIS CLIP
            # =================================

            clip_image_paths = []
            clip_mask_paths = []

            image_files = sorted(
                clip_folder.glob("*_endo.png")
            )

            for image_path in image_files:

                # skip mask files
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

                    clip_image_paths.append(image_path)
                    clip_mask_paths.append(mask_path)


            # =================================
            # APPLY TEMPORAL STRIDE
            # =================================
            #
            # stride=10 on 80 frames picks
            # indices 0, 10, 20, 30, 40, 50,
            # 60, 70 → 8 frames per clip.

            for i in range(
                0,
                len(clip_image_paths),
                stride
            ):
                image_paths.append(
                    clip_image_paths[i]
                )
                mask_paths.append(
                    clip_mask_paths[i]
                )
                video_ids.append(video_name)


    return image_paths, mask_paths, video_ids


# =========================================
# CREATE SPLITS
# =========================================

def create_splits():
    """
    Returns a dict with keys 'train', 'val',
    'test'. Each value is a tuple:
        (image_paths, mask_paths, video_ids)

    video_ids is a per-frame list of video
    names. It is used by VideoBalancedSampler
    to build cross-video batches.
    """

    train_images, train_masks, train_vids = (
        collect_image_mask_paths(
            TRAIN_VIDEOS,
            stride=TRAIN_STRIDE
        )
    )

    val_images, val_masks, val_vids = (
        collect_image_mask_paths(
            VAL_VIDEOS,
            stride=VAL_STRIDE
        )
    )

    test_images, test_masks, test_vids = (
        collect_image_mask_paths(
            TEST_VIDEOS,
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

    splits = create_splits()

    print("\n===== DATASET SUMMARY =====\n")

    for split_name, (images, masks, vids) in splits.items():

        unique_videos = sorted(set(vids))

        print(f"{split_name.upper()}")
        print(f"  Frames  : {len(images)}")
        print(f"  Videos  : {len(unique_videos)}")
        print(f"  Sources : {unique_videos}")
        print()
