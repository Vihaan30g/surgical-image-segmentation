import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# WHY COMBINED LOSS
# =========================================
#
# Problem with DiceLoss alone:
#   Dice loss on foreground classes is not
#   enough signal to force the model away from
#   the background-prediction shortcut.
#   With ~80-90% of pixels being background,
#   the model can achieve 95%+ pixel accuracy
#   and a decent-looking loss just by predicting
#   background everywhere. Foreground Dice then
#   plateaus because the model has no strong
#   incentive to learn rare classes like blood,
#   cystic_duct, or hepatic_vein.
#
# Solution — Combined loss:
#   Total Loss = DiceLoss + λ × CE(weighted)
#
#   CrossEntropy with class weights directly
#   penalizes wrong predictions on rare classes
#   much more than wrong background predictions.
#   This forces the model to actively try to
#   segment rare structures, not just ignore them.
#
# Why keep DiceLoss too?
#   CE optimizes per-pixel classification.
#   Dice optimizes region overlap directly.
#   They are complementary — CE handles class
#   imbalance, Dice handles spatial overlap.
#   Together they are stronger than either alone.
#
# λ (CE_WEIGHT) = 0.5:
#   Equal-ish contribution from both losses.
#   DiceLoss is in range [0,1].
#   CE loss is unbounded but typically 0.5–2.0
#   early in training. λ=0.5 keeps them balanced.


# =========================================
# CLASS WEIGHTS
# =========================================
#
# Computed as inverse class frequency,
# normalized so mean weight = 1.
#
# These are estimates based on typical
# CholecSeg8k pixel distributions:
#   - Background: ~55% of pixels → weight 0.04
#   - Liver: ~12% → weight 0.18
#   - Blood, cystic_duct, hepatic_vein,
#     liver_ligament: ~1% each → weight 2.10
#
# If you want exact weights from your data,
# run: python src/utils/compute_class_weights.py
# and paste the output here.
#
# Order must match CLASS_INFO in class_mapping.py:
# 0:background, 1:abdominal_wall, 2:liver,
# 3:gastrointestinal, 4:fat, 5:grasper,
# 6:connective_tissue, 7:blood, 8:cystic_duct,
# 9:l_hook, 10:gallbladder, 11:hepatic_vein,
# 12:liver_ligament

CLASS_WEIGHTS = [
    0.0000,   #  0 background
    1.4427,   #  1 abdominal_wall
    0.0150,   #  2 liver
    1.4427,   #  3 gastrointestinal_tract
    1.4427,   #  4 fat
    1.4427,   #  5 grasper
    1.4427,   #  6 connective_tissue
    1.4427,   #  7 blood
    1.4427,   #  8 cystic_duct
    0.0000,   #  9 l_hook_electrocautery
    1.4427,   # 10 gallbladder
    1.4427,   # 11 hepatic_vein
    0.0003,   # 12 liver_ligament
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
        ce_weight=0.5,
        smooth=1e-6,
        device=None
    ):
        """
        Combined Dice + Weighted CrossEntropy loss.

        ce_weight : float
            Weight for the CrossEntropy term.
            DiceLoss weight is implicitly 1.0.
            Total = DiceLoss + ce_weight * CE.
            Default 0.5 keeps contributions balanced.

        smooth : float
            Smoothing term for Dice numerator/denominator.

        device : torch.device or None
            If None, class weights are moved to the
            same device as the input tensors at runtime.
        """

        super().__init__()

        self.ce_weight = ce_weight

        self.dice = DiceLoss(smooth=smooth)

        # Register class weights as a buffer so they
        # move to GPU automatically with .to(device).
        weights_tensor = torch.tensor(
            CLASS_WEIGHTS,
            dtype=torch.float32
        )

        self.register_buffer(
            "class_weights",
            weights_tensor
        )


    def forward(self, logits, targets):
        """
        logits  : [B, 13, H, W]  raw model output
        targets : [B, H, W]      class indices 0–12
        """

        # =================================
        # DICE LOSS
        # =================================
        #
        # Measures region overlap for foreground
        # classes. Background excluded.

        loss_dice = self.dice(logits, targets)


        # =================================
        # WEIGHTED CROSS ENTROPY LOSS
        # =================================
        #
        # Per-pixel classification loss.
        # Class weights penalize wrong predictions
        # on rare foreground classes heavily.
        # Background has weight 0.038 — wrong
        # background predictions cost almost nothing,
        # so the model must focus on foreground.

        loss_ce = F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights
        )


        # =================================
        # COMBINED
        # =================================

        total = loss_dice + self.ce_weight * loss_ce

        return total
