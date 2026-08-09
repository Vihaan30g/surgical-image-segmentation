"""
metrics.py
----------
Per-class Dice score and mean IoU (mIoU) calculation for multi-class
segmentation evaluation.

Design notes on missing classes:
    A class that is absent from BOTH the prediction and the ground truth in
    a given frame/batch is not scoreable (0/0). We mark that class's score
    as NaN for that batch rather than forcing it to 0 or 1, and use
    `nanmean` when aggregating across classes/frames so absent classes
    don't distort the reported class-wise or mean scores. A class that is
    present in the ground truth but *missed* entirely by the prediction
    correctly scores 0.0 (true positive is required for a positive score).
"""

import numpy as np
import torch

import config


@torch.no_grad()
def compute_confusion_counts(preds: torch.Tensor, targets: torch.Tensor,
                              num_classes: int = config.NUM_CLASSES):
    """
    preds, targets: (N, H, W) int64 tensors of class indices.
    Returns per-class intersection, union, pred-sum, target-sum as
    (num_classes,) float tensors, accumulated over the whole batch.
    """
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    pred_sum = torch.zeros(num_classes, dtype=torch.float64)
    target_sum = torch.zeros(num_classes, dtype=torch.float64)

    for class_id in range(num_classes):
        pred_mask = preds == class_id
        target_mask = targets == class_id

        inter = (pred_mask & target_mask).sum().double()
        pred_count = pred_mask.sum().double()
        target_count = target_mask.sum().double()
        union_count = pred_count + target_count - inter

        intersection[class_id] = inter
        union[class_id] = union_count
        pred_sum[class_id] = pred_count
        target_sum[class_id] = target_count

    return intersection, union, pred_sum, target_sum


class SegmentationMetricAccumulator:
    """
    Accumulates confusion counts across many batches (e.g. a full validation
    epoch), then produces per-class Dice/IoU and mean Dice/mIoU at the end.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 class_names: dict = None, smooth: float = 1e-8):
        self.num_classes = num_classes
        self.class_names = class_names or config.CLASS_NAMES
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pred_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self.target_sum = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds, targets: (N, H, W) int64 class-index tensors (already argmax'd)."""
        inter, union, pred_sum, target_sum = compute_confusion_counts(
            preds.cpu(), targets.cpu(), self.num_classes
        )
        self.intersection += inter
        self.union += union
        self.pred_sum += pred_sum
        self.target_sum += target_sum

    def compute(self):
        """
        Returns:
            dice_per_class: (num_classes,) numpy array, NaN where class never appeared
            iou_per_class:  (num_classes,) numpy array, NaN where class never appeared
            mean_dice: float (nanmean over classes)
            mean_iou:  float (nanmean over classes)
        """
        present = (self.target_sum + self.pred_sum) > 0  # class appears somewhere

        dice_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)
        iou_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)

        denom_dice = self.pred_sum + self.target_sum
        denom_iou = self.union

        for c in range(self.num_classes):
            if not present[c]:
                continue  # leave as NaN: class absent from both pred and target
            dice_per_class[c] = (
                (2.0 * self.intersection[c] + self.smooth) / (denom_dice[c] + self.smooth)
            ).item()
            iou_per_class[c] = (
                (self.intersection[c] + self.smooth) / (denom_iou[c] + self.smooth)
            ).item()

        mean_dice = float(np.nanmean(dice_per_class))
        mean_iou = float(np.nanmean(iou_per_class))

        return dice_per_class, iou_per_class, mean_dice, mean_iou

    def print_class_wise_dice(self, dice_per_class: np.ndarray = None):
        if dice_per_class is None:
            dice_per_class, _, _, _ = self.compute()

        print("Class-wise Dice scores:")
        for class_id in range(self.num_classes):
            name = self.class_names.get(class_id, f"class_{class_id}")
            score = dice_per_class[class_id]
            score_str = f"{score:.4f}" if not np.isnan(score) else "N/A (absent)"
            print(f"  [{class_id:>2}] {name:<28s}: {score_str}")
