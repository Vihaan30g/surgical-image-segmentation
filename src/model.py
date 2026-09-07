"""
model.py
--------
A standard U-Net implemented from scratch in PyTorch:
  - 4 downsampling steps (encoder: 64 -> 128 -> 256 -> 512)
  - bottleneck at 1024 channels
  - 4 upsampling steps (decoder: 512 -> 256 -> 128 -> 64) with skip
    connections via channel-wise concatenation
  - GroupNorm (num_groups=8) instead of BatchNorm for small-batch stability
  - LeakyReLU(0.1) activations
  - outputs raw logits (no softmax) with shape (N, NUM_CLASSES, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    """Two (Conv2d -> GroupNorm -> LeakyReLU) layers."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_groups: int = config.GROUP_NORM_GROUPS,
                 slope: float = config.LEAKY_RELU_SLOPE):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock followed by 2x2 max pooling. Returns (skip_features, pooled)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class UpBlock(nn.Module):
    """Transposed-conv upsample, concatenate skip connection, then ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Guard against off-by-one spatial mismatches from odd input sizes.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    4-layer U-Net.

    Input:  (N, 3, 288, 512)
    Output: (N, NUM_CLASSES, 288, 512)  -- raw logits
    """

    def __init__(self, in_channels: int = config.IN_CHANNELS,
                 num_classes: int = config.NUM_CLASSES,
                 encoder_channels: list = None,
                 bottleneck_channels: int = config.BOTTLENECK_CHANNELS):
        super().__init__()
        encoder_channels = encoder_channels or config.ENCODER_CHANNELS  # [64, 128, 256, 512]

        c1, c2, c3, c4 = encoder_channels

        # Encoder
        self.down1 = DownBlock(in_channels, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.down4 = DownBlock(c3, c4)

        # Bottleneck
        self.bottleneck = ConvBlock(c4, bottleneck_channels)

        # Decoder (mirrors encoder channel widths)
        self.up4 = UpBlock(bottleneck_channels, skip_channels=c4, out_channels=c4)
        self.up3 = UpBlock(c4, skip_channels=c3, out_channels=c3)
        self.up2 = UpBlock(c3, skip_channels=c2, out_channels=c2)
        self.up1 = UpBlock(c2, skip_channels=c1, out_channels=c1)

        # Final 1x1 conv -> raw logits, no activation
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        logits = self.out_conv(x)
        return logits


if __name__ == "__main__":
    # Quick shape sanity check.
    model = UNet()
    dummy = torch.randn(2, config.IN_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, config.NUM_CLASSES, config.IMG_HEIGHT, config.IMG_WIDTH)
    print("Shape check passed.")