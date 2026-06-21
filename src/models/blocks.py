import torch
import torch.nn as nn


# =========================================
# WHY GROUP NORM INSTEAD OF BATCH NORM
# =========================================
#
# Our batches are intentionally mixed across
# different surgical videos. Each video has a
# very different visual style (lighting, camera
# angle, tissue appearance). BatchNorm computes
# mean and variance across the whole batch, so
# mixing styles makes those statistics noisy
# and meaningless.
#
# GroupNorm computes statistics per-image,
# per-group-of-channels. It does NOT depend on
# batch composition at all. This makes it
# robust to our mixed-video batches.
#
# We use num_groups=8 throughout. Rule of thumb:
# num_groups must divide out_channels evenly.
# 64 / 8 = 8  ✓
# 128 / 8 = 16 ✓
# 256 / 8 = 32 ✓
# 512 / 8 = 64 ✓
# 1024 / 8 = 128 ✓

NUM_GROUPS = 8


# =========================================
# DOUBLE CONVOLUTION BLOCK
# =========================================

class DoubleConv(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_p=0.0
    ):
        """
        Two consecutive Conv → GroupNorm → ReLU blocks.

        dropout_p : spatial dropout probability applied
                    AFTER the second ReLU. Set to 0.0 to
                    disable (used in encoder). Set to
                    0.3–0.5 in bottleneck/decoder to
                    regularize.
        """

        super().__init__()


        # ================================
        # FIRST CONV LAYER
        # ================================

        self.conv1 = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False       # bias redundant when norm follows
            ),

            # GroupNorm: normalizes per image, per channel group
            # affine=True keeps learnable scale + shift (gamma, beta)
            nn.GroupNorm(
                num_groups=NUM_GROUPS,
                num_channels=out_channels,
                affine=True
            ),

            nn.ReLU(inplace=True)
        )


        # ================================
        # SECOND CONV LAYER
        # ================================

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(
                num_groups=NUM_GROUPS,
                num_channels=out_channels,
                affine=True
            ),

            nn.ReLU(inplace=True)
        )


        # ================================
        # SPATIAL DROPOUT
        # ================================
        #
        # Dropout2d drops entire feature maps
        # (channels), not individual pixels.
        # This is stronger regularization for
        # conv networks than standard Dropout.
        #
        # Only active if dropout_p > 0.0

        self.dropout = (
            nn.Dropout2d(p=dropout_p)
            if dropout_p > 0.0
            else nn.Identity()
        )


    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.dropout(x)

        return x


# =========================================
# ENCODER BLOCK
# =========================================

class EncoderBlock(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels
    ):
        """
        DoubleConv → save skip → MaxPool down.

        Encoder blocks do NOT use dropout.
        We want the encoder to learn rich
        features. Regularization is applied
        at the bottleneck and decoder instead.
        """

        super().__init__()


        self.conv = DoubleConv(
            in_channels,
            out_channels,
            dropout_p=0.0    # no dropout in encoder
        )


        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


    def forward(self, x):


        # =================================
        # FEATURE EXTRACTION
        # =================================

        features = self.conv(x)


        # =================================
        # DOWNSAMPLING
        # =================================

        pooled = self.pool(features)


        # features → saved as skip connection
        # pooled   → passed deeper into network

        return features, pooled


# =========================================
# DECODER BLOCK
# =========================================

class DecoderBlock(nn.Module):


    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        dropout_p=0.0
    ):
        """
        ConvTranspose2d upsample → concat skip → DoubleConv.

        dropout_p is forwarded to the internal
        DoubleConv. Decoder blocks closer to the
        bottleneck (dec1, dec2) use dropout.
        Blocks closer to output (dec3, dec4)
        don't, to preserve fine spatial detail.
        """

        super().__init__()


        # ================================
        # LEARNED UPSAMPLING
        # ================================

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )


        # ================================
        # FEATURE REFINEMENT
        # ================================

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout_p=dropout_p
        )


    def forward(
        self,
        x,
        skip_features
    ):


        # ================================
        # UPSAMPLE
        # ================================

        x = self.up(x)


        # ================================
        # CONCATENATE SKIP FEATURES
        # ================================

        x = torch.cat(
            [x, skip_features],
            dim=1
        )


        # ================================
        # REFINE FEATURES
        # ================================

        x = self.conv(x)


        return x
