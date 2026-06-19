import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):


    def __init__(
        self,
        smooth=1e-6
    ):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):


        # =====================================
        # CONVERT LOGITS TO PROBABILITIES
        # =====================================

        probs = F.softmax(
            logits,
            dim=1
        )


        # =====================================
        # ONE-HOT ENCODE TARGET MASK
        # =====================================

        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=13
        )


        targets_one_hot = targets_one_hot.permute(
            0,
            3,
            1,
            2
        ).float()


        # =====================================
        # COMPUTE INTERSECTION
        # =====================================

        intersection = (
            probs *
            targets_one_hot
        ).sum(
            dim=(0, 2, 3)
        )


        # =====================================
        # COMPUTE UNION TERM
        # =====================================

        denominator = (
            probs.sum(dim=(0, 2, 3))
            +
            targets_one_hot.sum(dim=(0, 2, 3))
        )


        # =====================================
        # DICE SCORE
        # =====================================

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )


        # =====================================
        # DICE LOSS
        # =====================================

        dice_loss = 1.0 - dice_score.mean()


        return dice_loss