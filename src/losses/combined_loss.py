import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# CLASS WEIGHTS
# =========================================
#
# Inverse-frequency weights, normalized so
# max weight = 1.0 (not mean=1.0).
#
# Why max=1.0 normalization?
#   If we normalize to mean=1.0, rare classes
#   get weights like 2.1 which makes the CE
#   gradient ~2× larger than Dice gradient.
#   At LR=1e-4 this causes gradient explosion
#   → NaN loss from epoch 1.
#
#   Normalizing to max=1.0 caps all weights
#   between 0 and 1. The relative ordering is
#   identical — background is still the lowest,
#   rare classes still the highest — but the
#   gradient magnitude stays controlled.
#
# Class order matches CLASS_INFO in class_mapping.py:
# 0:background  1:abdominal_wall  2:liver
# 3:gastrointestinal  4:fat  5:grasper
# 6:connective_tissue  7:blood  8:cystic_duct
# 9:l_hook  10:gallbladder  11:hepatic_vein
# 12:liver_ligament

CLASS_WEIGHTS = [
    0.0181,   # 0  background        — ~55% of pixels, very low weight
    0.1249,   # 1  abdominal_wall
    0.0831,   # 2  liver             — large organ, common
    0.2499,   # 3  gastrointestinal
    0.2000,   # 4  fat
    0.5002,   # 5  grasper
    0.2499,   # 6  connective_tissue
    1.0000,   # 7  blood             — rare, max weight
    1.0000,   # 8  cystic_duct       — rare, thin structure
    0.5002,   # 9  l_hook
    0.2499,   # 10 gallbladder
    1.0000,   # 11 hepatic_vein      — rare
    1.0000,   # 12 liver_ligament    — rare
]


# =========================================
# DICE LOSS COMPONENT
# =========================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=13
        ).permute(0, 3, 1, 2).float()

        intersection = (
            probs * targets_one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probs.sum(dim=(0, 2, 3))
            + targets_one_hot.sum(dim=(0, 2, 3))
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (denominator + self.smooth)

        # exclude background (index 0)
        dice_score = dice_score[1:]

        return 1.0 - dice_score.mean()


# =========================================
# COMBINED LOSS
# =========================================

class CombinedLoss(nn.Module):

    def __init__(
        self,
        ce_weight=0.3,
        smooth=1e-6
    ):
        """
        Total = DiceLoss + ce_weight * CrossEntropy(weighted)

        ce_weight=0.3:
            CE at init ≈ 1.2 (with max-normalized weights).
            Dice at init ≈ 0.92.
            Combined ≈ 0.92 + 0.3×1.2 = 1.28.
            This is stable with LR=1e-4 and gradient clipping.

            Previous ce_weight=0.5 with mean-normalized weights
            gave combined loss ≈ 2.2 at init → gradient explosion
            → NaN from epoch 1.
        """

        super().__init__()

        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth=smooth)

        # register_buffer moves weights to GPU
        # automatically when you call .to(device)
        self.register_buffer(
            "class_weights",
            torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        """
        logits  : [B, 13, H, W]  raw model output
        targets : [B, H, W]      class indices 0–12
        """

        # Dice: region overlap on foreground classes
        loss_dice = self.dice(logits, targets)

        # Weighted CE: per-pixel, rare classes penalized more
        loss_ce = F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights
        )

        return loss_dice + self.ce_weight * loss_ce