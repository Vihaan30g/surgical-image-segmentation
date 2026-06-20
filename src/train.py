import torch
import os

# Enable cuDNN for performance on Colab
torch.backends.cudnn.enabled = True 

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

# Drive path for saving (set in Colab)
DRIVE_ROOT = "/content/drive/MyDrive/surgical_training_results"
MODEL_SAVE_PATH = os.path.join(DRIVE_ROOT, "best_model.pth")
LOG_DIR = os.path.join(DRIVE_ROOT, "runs/unet_cholecseg")

# Ensure directory exists
os.makedirs(DRIVE_ROOT, exist_ok=True)

# =====================================
# TENSORBOARD & DEVICE
# =====================================

writer = SummaryWriter(log_dir=LOG_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================
# DATALOADERS & MODEL
# =====================================

train_loader, val_loader, test_loader = create_dataloaders(batch_size=BATCH_SIZE)
model = UNet().to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================
# TRAINING LOOP
# =====================================

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    train_loss, train_dice, train_iou, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_dice, val_iou, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

    # Logging
    writer.add_scalars("Loss", {'Train': train_loss, 'Val': val_loss}, epoch)
    writer.add_scalars("Dice", {'Train': train_dice, 'Val': val_dice}, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Best model saved to Drive.")

writer.close()