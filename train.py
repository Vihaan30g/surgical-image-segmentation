"""
train.py
--------
End-to-end training script for the CholecSeg8k U-Net.

Run this in a Google Colab cell (GPU runtime). It will:
  1. Mount Google Drive.
  2. Build train/val datasets + video-balanced dataloaders.
  3. Build the U-Net, ComboLoss, AdamW optimizer, CosineAnnealingLR scheduler.
  4. Auto-resume from /content/drive/MyDrive/.../checkpoints/checkpoint_latest.pth
     if it exists.
  5. Train, validate, log class-wise Dice/mIoU, and export 3-way visual
     grids (image | ground truth | prediction) every epoch.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from dataset import CholecSegDataset, get_train_transforms, get_eval_transforms, build_dataloader
from model import UNet
from losses import ComboLoss
from metrics import SegmentationMetricAccumulator


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Environment checks
#
# NOTE: Google Drive is mounted OUTSIDE this script — run
#   from google.colab import drive
#   drive.mount('/content/drive')
# in its own notebook cell BEFORE running train.py (see README.md). This
# function only verifies that Drive is mounted and that the Kaggle dataset
# has already been downloaded to config.DATA_ROOT; it does not mount
# anything or download anything itself.
# --------------------------------------------------------------------------
def check_environment_ready():
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive is not mounted at /content/drive/MyDrive. "
            "Run `drive.mount('/content/drive')` in a notebook cell first "
            "(see README.md), then re-run this script."
        )

    if not os.path.isdir(config.DATA_ROOT):
        raise RuntimeError(
            f"Dataset not found at {config.DATA_ROOT}. Download/extract the "
            "Kaggle dataset into that path first (see README.md for the "
            "kaggle CLI commands)."
        )

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.VIS_DIR, exist_ok=True)
    print("Environment check passed: Drive mounted, dataset found, output dirs ready.")


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch: int, best_val_dice: float,
                     path: str = config.CHECKPOINT_LATEST):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_dice": best_val_dice,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at epoch {epoch} -> {path}")


def load_checkpoint_if_exists(model, optimizer, scheduler, path: str = config.CHECKPOINT_LATEST):
    """Returns (start_epoch, best_val_dice). start_epoch=0, best_val_dice=-inf if no checkpoint."""
    if not os.path.isfile(path):
        print("No existing checkpoint found. Starting training from scratch.")
        return 0, float("-inf")

    print(f"Found checkpoint at {path}. Resuming...")
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_dice = checkpoint["best_val_dice"]
    print(f"Resuming from epoch {start_epoch}, best_val_dice so far = {best_val_dice:.4f}")
    return start_epoch, best_val_dice


# --------------------------------------------------------------------------
# Visualization export
# --------------------------------------------------------------------------
# Deterministic RGB color palette for the 13 classes (for mask visualization).
_PALETTE = np.array(
    [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    ],
    dtype=np.uint8,
)


def colorize_mask(class_mask: np.ndarray) -> np.ndarray:
    """class_mask: (H, W) int array -> (H, W, 3) uint8 RGB image."""
    return _PALETTE[class_mask]


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """img_tensor: (3, H, W) float in [0, 1] -> (H, W, 3) uint8."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def save_visual_grid(image_tensor, gt_mask, pred_mask, save_path: str, frame_id: str):
    image_np = denormalize_image(image_tensor)
    gt_color = colorize_mask(gt_mask.cpu().numpy())
    pred_color = colorize_mask(pred_mask.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred_color)
    axes[2].set_title("Predicted Mask")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Frame: {frame_id}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{frame_id}.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Train / Validate epoch loops
# --------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [train]")
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, loss_components = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss_components["total_loss"]
        running_focal += loss_components["focal_loss"]
        running_dice += loss_components["dice_loss"]
        num_batches += 1

        progress.set_postfix(loss=loss_components["total_loss"])

    return {
        "train_loss": running_loss / max(num_batches, 1),
        "train_focal_loss": running_focal / max(num_batches, 1),
        "train_dice_loss": running_dice / max(num_batches, 1),
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch,
                        num_vis_samples=None, vis_dataset=None):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    metric_accum = SegmentationMetricAccumulator(num_classes=config.NUM_CLASSES)

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [val]")
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss, loss_components = criterion(logits, masks)
        running_loss += loss_components["total_loss"]
        num_batches += 1

        preds = torch.argmax(logits, dim=1)
        metric_accum.update(preds, masks)

        progress.set_postfix(val_loss=loss_components["total_loss"])

    dice_per_class, iou_per_class, mean_dice, mean_iou = metric_accum.compute()

    print(f"\nEpoch {epoch} Validation Summary:")
    print(f"  Val Loss:  {running_loss / max(num_batches, 1):.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Mean IoU:  {mean_iou:.4f}")
    metric_accum.print_class_wise_dice(dice_per_class)

    # --- Export RANDOM visualization samples (different every epoch) ---
    if num_vis_samples is not None and vis_dataset is not None:
        epoch_vis_dir = os.path.join(config.VIS_DIR, f"epoch_{epoch}")
        os.makedirs(epoch_vis_dir, exist_ok=True)

        # fresh random sample each epoch call — uses the global `random`
        # module (seeded once at program start in set_seed), so the
        # sequence of epochs is still reproducible run-to-run, but each
        # epoch gets a *different* set of indices from the last.
        n = min(num_vis_samples, len(vis_dataset))
        vis_indices = random.sample(range(len(vis_dataset)), k=n)

        for sample_idx in vis_indices:
            sample = vis_dataset[sample_idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["mask"]
            frame_id = sample["frame_id"]

            logits = model(image)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu()

            save_visual_grid(sample["image"], gt_mask, pred_mask, epoch_vis_dir, frame_id)

        print(f"  Saved visualizations to: {epoch_vis_dir} (indices: {vis_indices})")

    return {
        "val_loss": running_loss / max(num_batches, 1),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "dice_per_class": dice_per_class,
        "iou_per_class": iou_per_class,
    }

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    set_seed(config.SEED)
    check_environment_ready()

    device = config.DEVICE
    print(f"Using device: {device}")

    # --- Datasets ---
    train_dataset = CholecSegDataset(
        root_dir=config.DATA_ROOT,
        video_list=config.TRAIN_VIDEOS,
        frame_step=config.TRAIN_FRAME_STEP,
        transforms=get_train_transforms(),
    )
    val_dataset = CholecSegDataset(
        root_dir=config.DATA_ROOT,
        video_list=config.VAL_VIDEOS,
        frame_step=config.VAL_FRAME_STEP,
        transforms=get_eval_transforms(),
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # --- Dataloaders (video-balanced batches for train; plain shuffle-free for val) ---
    train_loader = build_dataloader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )

    # Fixed validation samples for visual tracking across epochs
    rng = random.Random(config.SEED)
    fixed_vis_indices = rng.sample(range(len(val_dataset)), k=min(config.NUM_VIS_SAMPLES, len(val_dataset)))

    # --- Model / loss / optimizer / scheduler ---
    model = UNet().to(device)
    criterion = ComboLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.COSINE_T_MAX)

    # --- Auto-resume ---
    start_epoch, best_val_dice = load_checkpoint_if_exists(model, optimizer, scheduler)

    # --- Training loop ---
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, epoch,
            num_vis_samples=config.NUM_VIS_SAMPLES, vis_dataset=val_dataset,
        )

        scheduler.step()

        print(
            f"Epoch {epoch} | train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"mean_dice={val_metrics['mean_dice']:.4f} | mean_iou={val_metrics['mean_iou']:.4f}"
        )

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_dice, path=config.CHECKPOINT_BEST)

        save_checkpoint(model, optimizer, scheduler, epoch, best_val_dice, path=config.CHECKPOINT_LATEST)

    print(f"Training complete. Best validation mean Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
