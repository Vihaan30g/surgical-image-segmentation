
from pathlib import Path


# =========================================
# DATASET ROOT
# =========================================

DATASET_ROOT = Path("data/archive")


# =========================================
# AVAILABLE VIDEOS
# =========================================

ALL_VIDEOS = sorted([

    folder.name

    for folder in DATASET_ROOT.iterdir()

    if folder.is_dir()
])


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
# COLLECT IMAGE AND MASK PATHS
# =========================================

def collect_image_mask_paths(video_list):

    image_paths = []

    mask_paths = []


    for video_name in video_list:

        video_path = DATASET_ROOT / video_name

        clip_folders = sorted(video_path.iterdir())


        for clip_folder in clip_folders:


            # =================================
            # RGB IMAGES
            # =================================

            image_files = sorted(

                clip_folder.glob("*_endo.png")
            )


            for image_path in image_files:


                # skip mask files

                if "mask" in image_path.name:

                    continue


                # =============================
                # CORRESPONDING MASK
                # =============================

                mask_name = (

                    image_path.stem
                    + "_watershed_mask.png"
                )

                mask_path = (
                    image_path.parent / mask_name
                )


                # =============================
                # STORE PAIRS
                # =============================

                if mask_path.exists():

                    image_paths.append(image_path)

                    mask_paths.append(mask_path)


    return image_paths, mask_paths


# =========================================
# CREATE SPLITS
# =========================================

def create_splits():

    train_images, train_masks = (
        collect_image_mask_paths(
            TRAIN_VIDEOS
        )
    )


    val_images, val_masks = (
        collect_image_mask_paths(
            VAL_VIDEOS
        )
    )


    test_images, test_masks = (
        collect_image_mask_paths(
            TEST_VIDEOS
        )
    )


    return {

        "train": (
            train_images,
            train_masks
        ),

        "val": (
            val_images,
            val_masks
        ),

        "test": (
            test_images,
            test_masks
        )
    }


# =========================================
# DEBUG INFO
# =========================================

if __name__ == "__main__":

    splits = create_splits()

    print("\n===== DATASET SUMMARY =====\n")

    for split_name in splits:

        images, masks = splits[split_name]

        print(f"{split_name.upper()}")

        print(f"Images: {len(images)}")

        print(f"Masks : {len(masks)}\n")
