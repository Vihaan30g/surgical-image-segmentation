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
#
# Changes from v1:
#
# 1. GroupNorm replaces BatchNorm everywhere
#    (handled inside DoubleConv in blocks.py).
#
# 2. Dropout2d added at the bottleneck (p=0.4)
#    and in the first two decoder blocks
#    (dec1 p=0.3, dec2 p=0.2). The last two
#    decoder blocks are kept clean so the model
#    can still recover fine spatial boundaries.
#
# 3. Architecture depth and channel counts are
#    unchanged — the model capacity is the same,
#    but regularization is applied selectively.

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
        #
        # No dropout in encoder blocks.
        # We want the encoder to freely extract
        # rich features from the input image.

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
        #
        # Dropout p=0.4 here.
        # The bottleneck is the most compressed
        # representation — the model is most
        # prone to memorizing video-specific
        # features here. Strong dropout forces
        # it to learn redundant, robust features.

        self.bottleneck = DoubleConv(
            512,
            1024,
            dropout_p=0.4
        )


        # =================================
        # DECODER
        # =================================
        #
        # dec1 (p=0.3): just expanded from
        # bottleneck, still high-level features.
        # Moderate dropout.
        #
        # dec2 (p=0.2): mid-level features.
        # Light dropout.
        #
        # dec3, dec4 (p=0.0): low-level spatial
        # detail for boundary recovery. No dropout
        # — we don't want to corrupt fine edges.

        self.dec1 = DecoderBlock(
            in_channels=1024,
            skip_channels=512,
            out_channels=512,
            dropout_p=0.3
        )

        self.dec2 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            dropout_p=0.2
        )

        self.dec3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            dropout_p=0.0
        )

        self.dec4 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
            dropout_p=0.0
        )


        # =================================
        # FINAL OUTPUT LAYER
        # =================================
        #
        # 1×1 conv maps 64 feature channels
        # to 13 class logits per pixel.
        # No activation — raw logits out.
        # Softmax is applied inside the loss
        # and metric functions.

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
        # skip1: [B, 64,  256, 256]
        # x:     [B, 64,  128, 128]

        skip2, x = self.enc2(x)
        # skip2: [B, 128, 128, 128]
        # x:     [B, 128,  64,  64]

        skip3, x = self.enc3(x)
        # skip3: [B, 256,  64,  64]
        # x:     [B, 256,  32,  32]

        skip4, x = self.enc4(x)
        # skip4: [B, 512,  32,  32]
        # x:     [B, 512,  16,  16]


        # =================================
        # BOTTLENECK
        # =================================

        x = self.bottleneck(x)
        # x: [B, 1024, 16, 16]


        # =================================
        # DECODER
        # =================================

        x = self.dec1(x, skip4)
        # x: [B, 512, 32, 32]

        x = self.dec2(x, skip3)
        # x: [B, 256, 64, 64]

        x = self.dec3(x, skip2)
        # x: [B, 128, 128, 128]

        x = self.dec4(x, skip1)
        # x: [B, 64, 256, 256]


        # =================================
        # FINAL OUTPUT
        # =================================

        x = self.final_conv(x)
        # x: [B, 13, 256, 256]

        return x
