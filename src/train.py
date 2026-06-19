import torch

from torch.utils.tensorboard import SummaryWriter

from src.datasets.dataloader_setup import (
    create_dataloaders
)

from src.models.unet import UNet

from src.losses.dice_loss import DiceLoss

from src.training.train_one_epoch import (
    train_one_epoch
)

from src.training.validate_one_epoch import (
    validate_one_epoch
)


# =====================================
# CONFIG
# =====================================

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

BATCH_SIZE = 8

MODEL_SAVE_PATH = "best_model.pth"


# =====================================
# TENSORBOARD
# =====================================

writer = SummaryWriter(
    log_dir="runs/unet_cholecseg"
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
# DATALOADERS
# =====================================

train_loader, val_loader, test_loader = (
    create_dataloaders(
        batch_size=BATCH_SIZE
    )
)


# =====================================
# MODEL
# =====================================

model = UNet().to(device)


# =====================================
# LOSS
# =====================================

criterion = DiceLoss()


# =====================================
# OPTIMIZER
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# TRAINING LOOP
# =====================================

best_val_loss = float("inf")


for epoch in range(NUM_EPOCHS):


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


    # =================================
    # CONSOLE LOGGING
    # =================================

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Dice : {train_dice:.4f}"
    )

    print(
        f"Train IoU  : {train_iou:.4f}"
    )

    print(
        f"Train Acc  : {train_acc:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Dice   : {val_dice:.4f}"
    )

    print(
        f"Val IoU    : {val_iou:.4f}"
    )

    print(
        f"Val Acc    : {val_acc:.4f}"
    )


    # =================================
    # TENSORBOARD LOGGING
    # =================================

    writer.add_scalar(
        "Loss/Train",
        train_loss,
        epoch
    )

    writer.add_scalar(
        "Loss/Validation",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "Dice/Train",
        train_dice,
        epoch
    )

    writer.add_scalar(
        "Dice/Validation",
        val_dice,
        epoch
    )

    writer.add_scalar(
        "IoU/Train",
        train_iou,
        epoch
    )

    writer.add_scalar(
        "IoU/Validation",
        val_iou,
        epoch
    )

    writer.add_scalar(
        "Accuracy/Train",
        train_acc,
        epoch
    )

    writer.add_scalar(
        "Accuracy/Validation",
        val_acc,
        epoch
    )


    # =================================
    # SAVE BEST MODEL
    # =================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(
            f"Best model saved "
            f"(Val Loss: {val_loss:.4f})"
        )


writer.close()