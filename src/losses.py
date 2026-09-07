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

    FL(p_t) = -alpha_c * (1 - p_t)^gamma * log(p_t)

    `alpha` may be a single scalar (applied uniformly, the old behavior) or
    a per-class weight tensor of shape (num_classes,) — e.g. inverse-
    frequency weights from compute_class_weights.py — so rare classes get
    a proportionally larger gradient contribution per pixel.
    """

    def __init__(self, gamma: float = config.FOCAL_GAMMA,
                 alpha=config.FOCAL_ALPHA, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)) or torch.is_tensor(alpha):
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)
            self.per_class_alpha = True
        else:
            self.alpha = alpha  # plain float, uniform weighting
            self.per_class_alpha = False

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # (N, H, W) per-pixel cross entropy, no reduction yet
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability assigned to the true class
        focal_term = (1.0 - p_t).pow(self.gamma)

        if self.per_class_alpha:
            # look up each pixel's alpha by its ground-truth class id
            alpha_per_pixel = self.alpha.to(targets.device)[targets]  # (N, H, W)
            loss = alpha_per_pixel * focal_term * ce_loss
        else:
            loss = self.alpha * focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss. Operates on softmax probabilities vs.
    one-hot targets.

    `class_weights`, if given (shape (num_classes,), e.g. from
    compute_class_weights.py), turns the plain mean over classes into a
    weighted mean, so rare classes (hepatic_vein, liver_ligament, etc.)
    contribute proportionally more to the loss instead of being drowned
    out by common classes (background, fat, liver).
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, smooth: float = 1e-5,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

        if class_weights is not None:
            weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights_tensor)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # (N, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        dims = (0, 2, 3)  # reduce over batch + spatial, keep class dim
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.class_weights is not None:
            w = self.class_weights.to(dice_per_class.device)
            loss = 1.0 - (dice_per_class * w).sum() / w.sum()
        else:
            loss = 1.0 - dice_per_class.mean()
        return loss


class ComboLoss(nn.Module):
    """Weighted sum of FocalLoss and DiceLoss."""

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 focal_weight: float = config.FOCAL_WEIGHT,
                 dice_weight: float = config.DICE_WEIGHT,
                 focal_gamma: float = config.FOCAL_GAMMA,
                 focal_alpha=config.FOCAL_ALPHA,
                 class_weights=None):
        super().__init__()
        # If class_weights are provided (per-class inverse-frequency
        # weights) and no explicit focal_alpha override was passed, reuse
        # class_weights as the focal alpha too, so both loss terms push in
        # the same direction on rare classes.
        if class_weights is not None and focal_alpha == config.FOCAL_ALPHA:
            focal_alpha = class_weights
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice = DiceLoss(num_classes=num_classes, class_weights=class_weights)
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