# src/debug/test_training_step.py

import torch

torch.backends.cudnn.enabled = False

from src.datasets.dataloader_setup import create_dataloaders
from src.models.unet import UNet
from src.losses.dice_loss import DiceLoss


device = torch.device("cuda")

train_loader, _, _ = create_dataloaders(
    batch_size=2
)

images, masks = next(iter(train_loader))

images = images.to(device)
masks = masks.to(device)

model = UNet().to(device)

criterion = DiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

outputs = model(images)

loss = criterion(outputs, masks)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("Training step successful")
print("Loss:", loss.item())