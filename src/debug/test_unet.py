import torch

from src.models.unet import UNet


model = UNet()

x = torch.randn(
    8,
    3,
    256,
    256
)

output = model(x)

print("\n===== INPUT SHAPE =====")
print(x.shape)

print("\n===== OUTPUT SHAPE =====")
print(output.shape)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n===== MODEL PARAMETERS =====")

print(f"Total parameters: {total_params:,}")

print(
    f"Trainable parameters: {trainable_params:,}"
)