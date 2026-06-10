import torch
import torch.nn as nn

from src.models.blocks import (
    DoubleConv,
    EncoderBlock,
    DecoderBlock
)


# =========================================
# UNET MODEL
# =========================================

class UNet(nn.Module):


    def __init__(
        self,
        in_channels=3,
        num_classes=13
    ):

        super().__init__()


        # =================================
        # ENCODER
        # =================================

        self.enc1 = EncoderBlock(
            in_channels,
            64
        )

        self.enc2 = EncoderBlock(
            64,
            128
        )

        self.enc3 = EncoderBlock(
            128,
            256
        )

        self.enc4 = EncoderBlock(
            256,
            512
        )


        # =================================
        # BOTTLENECK
        # =================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # =================================
        # DECODER
        # =================================

        self.dec1 = DecoderBlock(
            in_channels=1024,
            skip_channels=512,
            out_channels=512
        )

        self.dec2 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.dec3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.dec4 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )


        # =================================
        # FINAL OUTPUT LAYER
        # =================================

        self.final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1
        )


    def forward(self, x):


        # =================================
        # ENCODER
        # =================================

        skip1, x = self.enc1(x)

        skip2, x = self.enc2(x)

        skip3, x = self.enc3(x)

        skip4, x = self.enc4(x)


        # =================================
        # BOTTLENECK
        # =================================

        x = self.bottleneck(x)


        # =================================
        # DECODER
        # =================================

        x = self.dec1(x, skip4)

        x = self.dec2(x, skip3)

        x = self.dec3(x, skip2)

        x = self.dec4(x, skip1)


        # =================================
        # FINAL OUTPUT
        # =================================

        x = self.final_conv(x)


        return x
