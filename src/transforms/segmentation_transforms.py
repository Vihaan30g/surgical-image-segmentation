import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================
# WHY THESE AUGMENTATIONS
# =========================================
#
# Our core problem: the model sees training
# videos that look visually different from
# validation videos (different lighting, camera
# angle, tissue appearance per patient).
#
# Augmentation artificially creates that visual
# diversity during training. If the model learns
# to segment correctly under random brightness
# shifts and geometric distortions, it will
# generalize better to unseen surgical videos.
#
# Each augmentation is justified below.


# =========================================
# TRAIN TRANSFORMS
# =========================================

train_transforms = A.Compose([


    # =====================================
    # RESIZE
    # =====================================
    # Original resolution 854×480 → 256×256.
    # Smaller size keeps training fast and
    # memory usage low.

    A.Resize(
        height=256,
        width=256
    ),


    # =====================================
    # GEOMETRIC AUGMENTATIONS
    # =====================================
    # Applied identically to image AND mask,
    # so segmentation labels stay correct.

    # Mirror flip — endoscope can approach from
    # either side. Very common in surgical video.
    A.HorizontalFlip(p=0.5),

    # Vertical flip — less common but valid.
    A.VerticalFlip(p=0.3),

    # 90-degree rotations — camera is sometimes
    # rotated in the trocar port.
    A.RandomRotate90(p=0.5),

    # Small affine jitter: shift, scale, rotate.
    # Simulates small camera position variations
    # between surgeries.
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=0,    # zero-pad (black border)
        p=0.5
    ),

    # Elastic deformation — simulates soft tissue
    # deformation as the surgeon manipulates
    # organs. Very relevant for surgical imagery.
    A.ElasticTransform(
        alpha=60,
        sigma=6,
        p=0.3
    ),


    # =====================================
    # COLOR / LIGHTING AUGMENTATIONS
    # =====================================
    # Applied to the IMAGE ONLY — Albumentations
    # automatically skips these for the mask.

    # Brightness + contrast: endoscope lighting
    # varies a lot between cameras and surgeries.
    # This is the most important color augment.
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),

    # Hue + saturation: tissue color shifts
    # slightly between patients and camera types.
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.5
    ),

    # Gaussian blur: simulates slight
    # out-of-focus frames (common when surgeon
    # moves quickly).
    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.2
    ),

    # Gaussian noise: simulates camera sensor
    # noise in darker regions of the frame.
    A.GaussNoise(
        p=0.2
    ),

    # CLAHE (Contrast Limited Adaptive
    # Histogram Equalization): some endoscope
    # frames are very low contrast. CLAHE
    # simulates enhanced-contrast preprocessing
    # that might be applied inconsistently.
    A.CLAHE(
        clip_limit=2.0,
        p=0.2
    ),


    # =====================================
    # NORMALIZATION + TENSOR CONVERSION
    # =====================================
    # ImageNet mean/std — standard starting
    # point even for medical imagery.

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    # Converts numpy [H,W,C] → torch [C,H,W].
    # Mask goes from [H,W] numpy → [H,W] tensor.
    ToTensorV2()

])


# =========================================
# VALIDATION TRANSFORMS
# =========================================
#
# No augmentation for validation — we want a
# deterministic, stable metric. Only resize and
# normalize, identical to what the model will
# see at test/inference time.

val_transforms = A.Compose([

    A.Resize(
        height=256,
        width=256
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()

])
