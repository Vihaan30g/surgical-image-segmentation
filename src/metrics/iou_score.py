import torch
import torch.nn.functional as F


NUM_CLASSES = 13


def iou_score(
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

        union = (
            pred_one_hot.sum(dim=(1, 2))
            +
            target_one_hot.sum(dim=(1, 2))
            -
            intersection
        )

        iou = (
            intersection + smooth
        ) / (
            union + smooth
        )

        valid_classes = (
            target_one_hot.sum(
                dim=(1, 2)
            ) > 0
        )

        # Ignore background
        valid_classes[0] = False

        iou = iou[valid_classes]

        if len(iou) > 0:

            image_scores.append(
                iou.mean()
            )

    if len(image_scores) == 0:

        return 0.0

    return torch.stack(
        image_scores
    ).mean().item()