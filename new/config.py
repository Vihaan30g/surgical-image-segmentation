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
# --------------------------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/surgical-image-segmentation"

DATA_ROOT = os.path.join(DRIVE_ROOT, "data", "archive")
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
CHECKPOINT_LATEST = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pth")
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth")
VIS_DIR = os.path.join(DRIVE_ROOT, "visualizations")

# --------------------------------------------------------------------------
# 2. DATASET SPLITS (VIDEO-LEVEL, to prevent frame leakage)
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
    0: {"name": "background", "raw_value": 80},
    1: {"name": "abdominal_wall", "raw_value": 17},
    2: {"name": "liver", "raw_value": 33},
    3: {"name": "gastrointestinal_tract", "raw_value": 19},
    4: {"name": "fat", "raw_value": 18},
    5: {"name": "grasper", "raw_value": 49},
    6: {"name": "connective_tissue", "raw_value": 35},
    7: {"name": "blood", "raw_value": 36},
    8: {"name": "cystic_duct", "raw_value": 37},
    9: {"name": "l_hook_electrocautery", "raw_value": 50},
    10: {"name": "gallbladder", "raw_value": 34},
    11: {"name": "hepatic_vein", "raw_value": 51},
    12: {"name": "liver_ligament", "raw_value": 5},
}

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
NUM_EPOCHS = 50
COSINE_T_MAX = NUM_EPOCHS  # CosineAnnealingLR period

# Loss weighting between Focal and Dice components
FOCAL_WEIGHT = 0.5
DICE_WEIGHT = 0.5
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

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
