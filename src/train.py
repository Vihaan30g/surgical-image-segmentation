import os
import torch
from torch.utils.tensorboard import SummaryWriter

from src.datasets.dataloader_setup import create_dataloaders
from src.models.unet import UNet
from src.losses.combined_loss import CombinedLoss
from src.training.train_one_epoch import train_one_epoch
from src.training.validate_one_epoch import validate_one_epoch


# =========================================
# CONFIG
# =========================================

NUM_EPOCHS    = 100
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4

# ReduceLROnPlateau patience:
# If val_loss doesn't improve for this many
# epochs, learning rate is halved. This helps
# escape plateaus we saw in v1 training.
LR_PATIENCE  = 10
LR_FACTOR    = 0.5
LR_MIN       = 1e-6

# Early stopping patience:
# If val_loss doesn't improve for this many
# epochs, stop training. Prevents wasted
# compute on a stuck run.
EARLY_STOP_PATIENCE = 20


# =========================================
# PATHS
# =========================================
#
# DRIVE_ROOT: where checkpoints and tensorboard
# logs are saved on Google Drive. Edit this to
# match your Drive folder.

DRIVE_ROOT = (
    "/content/drive/MyDrive/"
    "surgical_training_results_v2"
)

# best_model.pth: saved whenever val_loss
# improves. Used for final evaluation.
BEST_MODEL_PATH = os.path.join(
    DRIVE_ROOT,
    "best_model.pth"
)

# last_checkpoint.pth: saved every epoch.
# Used to resume training if Colab disconnects.
LAST_CHECKPOINT_PATH = os.path.join(
    DRIVE_ROOT,
    "last_checkpoint.pth"
)

LOG_DIR = os.path.join(
    DRIVE_ROOT,
    "runs",
    "unet_v2_cholecseg"
)

os.makedirs(DRIVE_ROOT, exist_ok=True)


# =========================================
# TENSORBOARD
# =========================================

writer = SummaryWriter(log_dir=LOG_DIR)


# =========================================
# DEVICE
# =========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# =========================================
# MODEL
# =========================================

model = UNet().to(device)

criterion = CombinedLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ReduceLROnPlateau: monitors val_loss.
# If it doesn't decrease for LR_PATIENCE
# epochs, lr = lr * LR_FACTOR.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=LR_PATIENCE,
    factor=LR_FACTOR,
    min_lr=LR_MIN
)


# =========================================
# RESUME FROM CHECKPOINT
# =========================================
#
# On Colab, the runtime can disconnect at any
# time. We save last_checkpoint.pth every epoch
# so we can resume from exactly where we left
# off. This is separate from best_model.pth
# (which only saves on val_loss improvement).
#
# Resume logic:
#   1. Check if last_checkpoint.pth exists.
#   2. If yes, load model weights, optimizer
#      state, scheduler state, epoch counter,
#      and best_val_loss tracker.
#   3. Continue training from START_EPOCH.

START_EPOCH   = 0
best_val_loss = float("inf")
epochs_no_improve = 0    # early stopping counter

if os.path.exists(LAST_CHECKPOINT_PATH):

    print("Checkpoint found. Resuming...")

    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    START_EPOCH = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["best_val_loss"]

    epochs_no_improve = checkpoint.get(
        "epochs_no_improve", 0
    )

    print(
        f"Resuming from epoch {START_EPOCH} | "
        f"Best val loss so far: {best_val_loss:.4f}"
    )

else:

    print("No checkpoint found. Starting fresh.")


# =========================================
# DATALOADERS
# =========================================
#
# Created AFTER model init so that if the
# checkpoint load fails, we haven't wasted
# time building the dataset index.

train_loader, val_loader, test_loader = (
    create_dataloaders(batch_size=BATCH_SIZE)
)

print(f"\nTrain batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")
print(f"Test  batches : {len(test_loader)}\n")


# =========================================
# TRAINING LOOP
# =========================================

for epoch in range(START_EPOCH, NUM_EPOCHS):


    # =====================================
    # TRAIN
    # =====================================

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


    # =====================================
    # VALIDATE
    # =====================================

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


    # =====================================
    # LEARNING RATE SCHEDULER STEP
    # =====================================
    #
    # Pass val_loss to ReduceLROnPlateau.
    # The scheduler will halve the LR if
    # val_loss hasn't improved for LR_PATIENCE
    # epochs.

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # =====================================
    # PRINT EPOCH SUMMARY
    # =====================================

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| LR: {current_lr:.2e}"
    )

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Dice : {train_dice:.4f}")
    print(f"Train IoU  : {train_iou:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Dice   : {val_dice:.4f}")
    print(f"Val IoU    : {val_iou:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")


    # =====================================
    # TENSORBOARD LOGGING
    # =====================================

    writer.add_scalars(
        "Loss",
        {"Train": train_loss, "Val": val_loss},
        epoch
    )

    writer.add_scalars(
        "Dice",
        {"Train": train_dice, "Val": val_dice},
        epoch
    )

    writer.add_scalars(
        "IoU",
        {"Train": train_iou, "Val": val_iou},
        epoch
    )

    writer.add_scalars(
        "Accuracy",
        {"Train": train_acc, "Val": val_acc},
        epoch
    )

    writer.add_scalar(
        "LearningRate",
        current_lr,
        epoch
    )


    # =====================================
    # SAVE LAST CHECKPOINT (EVERY EPOCH)
    # =====================================
    #
    # This is the resume checkpoint. Saved
    # every epoch so Colab disconnects lose
    # at most one epoch of work.

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss":        best_val_loss,
            "epochs_no_improve":    epochs_no_improve,
        },
        LAST_CHECKPOINT_PATH
    )


    # =====================================
    # SAVE BEST MODEL
    # =====================================
    #
    # Only saved when val_loss improves.
    # This is the model you will use for final
    # evaluation — not the last checkpoint.

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_no_improve = 0

        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss":        best_val_loss,
                "epochs_no_improve":    epochs_no_improve,
            },
            BEST_MODEL_PATH
        )

        print("Best model saved.")

    else:

        epochs_no_improve += 1


    # =====================================
    # EARLY STOPPING
    # =====================================
    #
    # If val_loss hasn't improved for
    # EARLY_STOP_PATIENCE epochs, stop.
    # Prevents wasting 100 epochs on a model
    # that peaked at epoch 30.

    if epochs_no_improve >= EARLY_STOP_PATIENCE:

        print(
            f"\nEarly stopping triggered. "
            f"No improvement for "
            f"{EARLY_STOP_PATIENCE} epochs."
        )

        break


writer.close()

print("\nTraining complete.")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")