
import torch
import torch.nn as nn


# =========================================
# DOUBLE CONVOLUTION BLOCK
# =========================================

class DoubleConv(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),


            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        return self.block(x)









# =========================================
# ENCODER BLOCK
# =========================================

class EncoderBlock(nn.Module):


    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.conv = DoubleConv(
            in_channels,
            out_channels
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


        return features, pooled

