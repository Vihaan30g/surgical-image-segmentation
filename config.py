"""
config.py
---------
Central configuration for the CholecSeg8k U-Net segmentation project.
All paths, dataset splits, class mappings, and hyperparameters live here so
every other module can import a single source of truth.
"""

import os
import torch

# --------------------------------------------------------------------------
# 1. PATHS
#
# Drive is now mounted OUTSIDE this codebase (a manual `drive.mount(...)`
# cell run once at the top of the Colab notebook, before cloning/running
# any of this code). This code never calls drive.mount() itself.
#
# - DATA_ROOT lives on the local Colab VM disk (fast local SSD), populated
#   by downloading the dataset directly from Kaggle inside Colab.
# - CHECKPOINT_DIR / VIS_DIR still point into the mounted Drive so training
#   progress (checkpoints + visualizations) survives runtime disconnects.
# --------------------------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/surgical-image-segmentation"

# Local (non-Drive) path where the Kaggle dataset is downloaded/extracted
# inside the Colab VM. This has moved between <repo_root>/data/archive and
# <repo_root> directly across different Colab sessions (depending on how
# the Kaggle zip extracted), so _detect_data_root() below checks several
# candidate locations automatically instead of hardcoding one that breaks
# next session. Confirmed via `!pwd` + `!ls` + `!ls ./archive/` that the
# current, correct layout is <repo_root>/data/archive/video01, video02, ...
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _detect_data_root():
    """
    The dataset's location relative to the repo root has changed between
    Colab sessions twice now (once at <repo_root>/data/archive, once
    directly at <repo_root>), presumably depending on how the Kaggle zip
    was extracted. Rather than hardcode one and have it silently break
    next session, auto-detect by checking where a known video folder
    (video01) actually is, and fall back with a clear error otherwise.
    """
    candidates = [
        _REPO_ROOT,                                  # videoXX directly in repo root
        os.path.join(_REPO_ROOT, "data", "archive"),  # videoXX under data/archive
        os.path.join(_REPO_ROOT, "data"),             # videoXX under data
        "/content/data/archive",                      # legacy absolute fallback
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "video01")):
            return candidate
    # Nothing matched — don't silently pick a wrong path; surface this
    # clearly instead of failing later with 17 confusing "MISSING" warnings.
    raise RuntimeError(
        "Could not auto-detect DATA_ROOT: no candidate path contains a "
        "'video01' folder. Checked: " + ", ".join(candidates) + ". "
        "Run `!find /content -maxdepth 4 -iname 'video01'` in Colab to "
        "locate the dataset, then set DATA_ROOT manually in config.py."
    )


DATA_ROOT = _detect_data_root()
KAGGLE_DATASET_SLUG = "newslab/cholecseg8k"  # kaggle datasets download -d <slug>

# Results (checkpoints + visualizations) persist to Drive.
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
CHECKPOINT_LATEST = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth")
VIS_DIR = os.path.join(DRIVE_ROOT, "visualizations")

# --------------------------------------------------------------------------
# 2. DATASET SPLITS
#
# NOTE: these TRAIN_VIDEOS/VAL_VIDEOS/TEST_VIDEOS lists are now used ONLY
# by generate_clip_split.py (to know which raw video folders to scan). The
# actual train/val/test assignment used by dataset.py / train.py comes from
# splits.py (TRAIN_CLIPS / VAL_CLIPS / TEST_CLIPS), which is CLIP-level, not
# whole-video-level — whole-video splitting starved several rare classes
# (liver_ligament: 0 training frames; connective_tissue, blood, hepatic_vein:
# 0 validation frames) because those classes each only occur in 1-2 of the
# 17 source videos. Regenerate splits.py by running generate_clip_split.py
# whenever these video lists change.
# --------------------------------------------------------------------------
TRAIN_VIDEOS = [
    "video01", "video12", "video18", "video20", "video24", "video26",
    "video27", "video28", "video37", "video43", "video48", "video52",
]
VAL_VIDEOS = ["video09", "video17", "video55"]
TEST_VIDEOS = ["video25", "video35"]

# --------------------------------------------------------------------------
# 3. FILE NAMING PATTERNS
# --------------------------------------------------------------------------
IMAGE_SUFFIX = "_endo.png"
COLOR_MASK_SUFFIX = "_endo_color_mask.png"
TOOL_MASK_SUFFIX = "_endo_mask.png"
WATERSHED_MASK_SUFFIX = "_endo_watershed_mask.png"  # ground truth

# --------------------------------------------------------------------------
# 4. CLASS INFO & RAW-VALUE REMAPPING
# --------------------------------------------------------------------------
CLASS_INFO = {
    0: {"name": "background", "raw_value": 50},
    1: {"name": "abdominal_wall", "raw_value": 11},
    2: {"name": "liver", "raw_value": 21},
    3: {"name": "gastrointestinal_tract", "raw_value": 13},
    4: {"name": "fat", "raw_value": 12},
    5: {"name": "grasper", "raw_value": 31},
    6: {"name": "connective_tissue", "raw_value": 23},
    7: {"name": "blood", "raw_value": 24},
    8: {"name": "cystic_duct", "raw_value": 25},
    9: {"name": "l_hook_electrocautery", "raw_value": 32},
    10: {"name": "gallbladder", "raw_value": 22},
    11: {"name": "hepatic_vein", "raw_value": 33},
    12: {"name": "liver_ligament", "raw_value": 5},
}
# raw values 0 and 255 (near-white / border / void pixels, confirmed via
# derive_mapping_v2.py to carry negligible or non-semantic content) are NOT
# listed above, so they fall through to the LUT's default of class 0
# (background) in dataset.py's _build_lookup_table — this is intentional.
#
# Mapping provenance (2024 debugging session):
#   background(50), abdominal_wall(11), liver(21), fat(12), gi_tract(13),
#   grasper(31), blood(24)            -> derived via color-mask crosswalk,
#                                          high confidence, unchanged since.
#   gallbladder(22), liver_ligament(5),
#   connective_tissue(23), hepatic_vein(33) -> corrected via direct visual
#                                          confirmation by the user.
#   cystic_duct(25), l_hook_electrocautery(32) -> inferred from pixel-area
#                                          plausibility (small vs. large),
#                                          NOT yet visually confirmed —
#                                          double-check these two.

RAW_TO_CLASS = {info["raw_value"]: class_id for class_id, info in CLASS_INFO.items()}
CLASS_NAMES = {class_id: info["name"] for class_id, info in CLASS_INFO.items()}
NUM_CLASSES = len(CLASS_INFO)

# --------------------------------------------------------------------------
# 5. TEMPORAL SUB-SAMPLING (step size per split)
# --------------------------------------------------------------------------
TRAIN_FRAME_STEP = 2
VAL_FRAME_STEP = 4
TEST_FRAME_STEP = 4

# --------------------------------------------------------------------------
# 6. IMAGE / AUGMENTATION SETTINGS
# --------------------------------------------------------------------------
IMG_WIDTH = 512
IMG_HEIGHT = 288

# --------------------------------------------------------------------------
# 7. DATALOADER / TRAINING HYPERPARAMETERS
# --------------------------------------------------------------------------
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 80  # bumped from 50 — scheduler in train.py rebuilds fresh on
                 # resume using whatever NUM_EPOCHS currently is, so this
                 # can be changed again later and re-run safely.
COSINE_T_MAX = NUM_EPOCHS  # kept for reference; train.py's build_scheduler()
                           # uses NUM_EPOCHS directly, not this constant.

# Loss weighting between Focal and Dice components
FOCAL_WEIGHT = 0.5
DICE_WEIGHT = 0.5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Per-class weights (inverse pixel frequency over the NEW clip-level TRAIN
# split, normalized so mean weight = 1.0, capped at 20.0) from
# compute_class_weights.py. Re-run that script and update this list any
# time splits.py changes (e.g. if you regenerate the clip split).
CLASS_WEIGHTS = [
    0.0025,  # [0] background
    0.0031,  # [1] abdominal_wall
    0.0035,  # [2] liver
    0.0367,  # [3] gastrointestinal_tract
    0.0044,  # [4] fat
    0.0315,  # [5] grasper
    0.0357,  # [6] connective_tissue
    0.2059,  # [7] blood
    1.8764,  # [8] cystic_duct
    0.0705,  # [9] l_hook_electrocautery
    0.0111,  # [10] gallbladder
    10.3185,  # [11] hepatic_vein
    0.4001,  # [12] liver_ligament
]

# --------------------------------------------------------------------------
# 8. MODEL ARCHITECTURE
# --------------------------------------------------------------------------
IN_CHANNELS = 3
ENCODER_CHANNELS = [64, 128, 256, 512]
BOTTLENECK_CHANNELS = 1024
GROUP_NORM_GROUPS = 8
LEAKY_RELU_SLOPE = 0.1

# --------------------------------------------------------------------------
# 9. MISC
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_VIS_SAMPLES = 3  # fixed validation samples saved every epoch