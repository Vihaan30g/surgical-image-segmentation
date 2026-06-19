import torch
import torch.nn.functional as F


def dice_score(
    logits,
    targets,
    smooth=1e-6
):

    probs = F.softmax(
        logits,
        dim=1
    )

    preds = torch.argmax(
        probs,
        dim=1
    )

    preds_one_hot = F.one_hot(
        preds,
        num_classes=13
    )

    targets_one_hot = F.one_hot(
        targets.long(),
        num_classes=13
    )

    preds_one_hot = preds_one_hot.permute(
        0, 3, 1, 2
    ).float()

    targets_one_hot = targets_one_hot.permute(
        0, 3, 1, 2
    ).float()

    intersection = (
        preds_one_hot *
        targets_one_hot
    ).sum(
        dim=(0, 2, 3)
    )

    denominator = (
        preds_one_hot.sum(dim=(0, 2, 3))
        +
        targets_one_hot.sum(dim=(0, 2, 3))
    )


    dice = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

        
    valid_classes = (
        targets_one_hot.sum(
            dim=(0, 2, 3)
        ) > 0
    )

    valid_classes[0] = False



    dice = dice[valid_classes]

    return dice.mean().item()