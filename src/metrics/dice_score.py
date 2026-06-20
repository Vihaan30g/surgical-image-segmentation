import torch
import torch.nn.functional as F


NUM_CLASSES = 13


def dice_score(
    logits,
    targets,
    smooth=1e-6
):

    preds = torch.argmax(
        logits,
        dim=1
    )

    image_scores = []

    for pred, target in zip(preds, targets):

        pred_one_hot = F.one_hot(
            pred,
            num_classes=NUM_CLASSES
        ).permute(
            2,
            0,
            1
        ).float()

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=NUM_CLASSES
        ).permute(
            2,
            0,
            1
        ).float()

        intersection = (
            pred_one_hot *
            target_one_hot
        ).sum(
            dim=(1, 2)
        )

        denominator = (
            pred_one_hot.sum(dim=(1, 2))
            +
            target_one_hot.sum(dim=(1, 2))
        )

        dice = (
            2 * intersection + smooth
        ) / (
            denominator + smooth
        )

        valid_classes = (
            target_one_hot.sum(
                dim=(1, 2)
            ) > 0
        )

        # Ignore background
        valid_classes[0] = False

        dice = dice[valid_classes]

        if len(dice) > 0:

            image_scores.append(
                dice.mean()
            )

    if len(image_scores) == 0:

        return 0.0

    return torch.stack(
        image_scores
    ).mean().item()