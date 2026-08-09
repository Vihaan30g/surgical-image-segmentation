"""
losses.py
---------
Multi-class Focal Loss + Dice Loss, combined into a single ComboLoss for
training a 13-class segmentation model with heavy class imbalance.

All losses expect:
    logits: (N, C, H, W) raw (un-normalized) model output
    targets: (N, H, W) int64 tensor of class indices in [0, C-1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017), built on top of
    per-pixel softmax cross entropy.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = config.FOCAL_GAMMA,
                 alpha: float = config.FOCAL_ALPHA, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # (N, H, W) per-pixel cross entropy, no reduction yet
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability assigned to the true class
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = self.alpha * focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss, averaged over classes present in the batch's
    logits/targets. Operates on softmax probabilities vs. one-hot targets.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # (N, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        dims = (0, 2, 3)  # reduce over batch + spatial, keep class dim
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_per_class.mean()
        return loss


class ComboLoss(nn.Module):
    """Weighted sum of FocalLoss and DiceLoss."""

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 focal_weight: float = config.FOCAL_WEIGHT,
                 dice_weight: float = config.DICE_WEIGHT,
                 focal_gamma: float = config.FOCAL_GAMMA,
                 focal_alpha: float = config.FOCAL_ALPHA):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total, {
            "focal_loss": focal_loss.item(),
            "dice_loss": dice_loss.item(),
            "total_loss": total.item(),
        }
