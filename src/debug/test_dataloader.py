
import torch

from src.datasets.dataloader_setup import (
    create_dataloaders
)


# =========================================
# CREATE DATALOADERS
# =========================================

train_loader, val_loader, test_loader = (
    create_dataloaders(batch_size=8)
)


# =========================================
# GET ONE TRAINING BATCH
# =========================================

images, masks = next(iter(train_loader))


# =========================================
# PRINT INFORMATION
# =========================================

print("\n===== IMAGE INFO =====")

print("Shape:", images.shape)

print("Dtype:", images.dtype)

print("Min value:", images.min().item())

print("Max value:", images.max().item())


print("\n===== MASK INFO =====")

print("Shape:", masks.shape)

print("Dtype:", masks.dtype)

print("Unique class IDs:")

print(torch.unique(masks))


# =========================================
# PRINT BATCH SIZE
# =========================================

print("\nBatch size:")

print(images.shape[0])

