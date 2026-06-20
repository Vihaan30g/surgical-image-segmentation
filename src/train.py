import os
import torch

from torch.utils.tensorboard import SummaryWriter

from src.datasets.dataloader_setup import create_dataloaders
from src.models.unet import UNet
from src.losses.dice_loss import DiceLoss
from src.training.train_one_epoch import train_one_epoch
from src.training.validate_one_epoch import validate_one_epoch


# =====================================
# CONFIG
# =====================================

NUM_EPOCHS = 100

BATCH_SIZE = 8

LEARNING_RATE = 1e-4


# =====================================
# DRIVE PATHS
# =====================================

DRIVE_ROOT = (
    "/content/drive/MyDrive/"
    "surgical_training_results"
)

MODEL_SAVE_PATH = os.path.join(
    DRIVE_ROOT,
    "best_model.pth"
)

LOG_DIR = os.path.join(
    DRIVE_ROOT,
    "runs",
    "unet_cholecseg"
)

os.makedirs(
    DRIVE_ROOT,
    exist_ok=True
)


# =====================================
# TENSORBOARD
# =====================================

writer = SummaryWriter(
    log_dir=LOG_DIR
)


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# =====================================
# MODEL
# =====================================

model = UNet().to(device)

criterion = DiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# RESUME CHECKPOINT
# =====================================

START_EPOCH = 0

best_val_loss = float("inf")

if os.path.exists(MODEL_SAVE_PATH):

    print(
        "Checkpoint found. Loading..."
    )

    checkpoint = torch.load(
        MODEL_SAVE_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    START_EPOCH = (
        checkpoint["epoch"] + 1
    )

    best_val_loss = (
        checkpoint["best_val_loss"]
    )

    print(
        f"Resuming from epoch "
        f"{START_EPOCH}"
    )


# =====================================
# DATALOADERS
# =====================================

train_loader, val_loader, test_loader = (
    create_dataloaders(
        batch_size=BATCH_SIZE
    )
)


# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(
    START_EPOCH,
    NUM_EPOCHS
):

    (
        train_loss,
        train_dice,
        train_iou,
        train_acc
    ) = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    (
        val_loss,
        val_dice,
        val_iou,
        val_acc
    ) = validate_one_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    print(
        f"\nEpoch "
        f"[{epoch+1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Dice : "
        f"{train_dice:.4f}"
    )

    print(
        f"Train IoU  : "
        f"{train_iou:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc:.4f}"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Dice   : "
        f"{val_dice:.4f}"
    )

    print(
        f"Val IoU    : "
        f"{val_iou:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc:.4f}"
    )

    # ==========================
    # TENSORBOARD
    # ==========================

    writer.add_scalars(
        "Loss",
        {
            "Train": train_loss,
            "Val": val_loss
        },
        epoch
    )

    writer.add_scalars(
        "Dice",
        {
            "Train": train_dice,
            "Val": val_dice
        },
        epoch
    )

    writer.add_scalars(
        "IoU",
        {
            "Train": train_iou,
            "Val": val_iou
        },
        epoch
    )

    writer.add_scalars(
        "Accuracy",
        {
            "Train": train_acc,
            "Val": val_acc
        },
        epoch
    )

    # ==========================
    # SAVE EVERY EPOCH
    # ==========================

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "best_val_loss":
                best_val_loss,
        },
        MODEL_SAVE_PATH
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_val_loss":
                    best_val_loss,
            },
            MODEL_SAVE_PATH
        )

        print(
            "Best model saved."
        )

writer.close()