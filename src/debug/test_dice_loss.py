import torch

from src.datasets.dataloader_setup import (
    create_dataloaders
)

from src.models.unet import UNet

from src.losses.dice_loss import DiceLoss


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


train_loader, _, _ = create_dataloaders(
    batch_size=2
)


images, masks = next(
    iter(train_loader)
)


model = UNet().to(device)

criterion = DiceLoss()


images = images.to(device)

masks = masks.to(device)


outputs = model(images)

loss = criterion(
    outputs,
    masks
)


print("\n===== OUTPUT SHAPE =====")
print(outputs.shape)

print("\n===== MASK SHAPE =====")
print(masks.shape)

print("\n===== DICE LOSS =====")
print(loss.item())