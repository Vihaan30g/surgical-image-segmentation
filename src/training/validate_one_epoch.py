from src.metrics.dice_score import dice_score
from src.metrics.iou_score import iou_score
from src.metrics.pixel_accuracy import pixel_accuracy

import torch


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):

    # ==============================
    # EVALUATION MODE
    # ==============================

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    # ==============================
    # NO GRADIENTS
    # ==============================

    with torch.no_grad():

        for images, masks in dataloader:

            # ======================
            # DEVICE TRANSFER
            # ======================

            images = images.to(device)

            masks = masks.to(device)

            # ======================
            # FORWARD PASS
            # ======================

            outputs = model(images)

            # ======================
            # LOSS
            # ======================

            loss = criterion(
                outputs,
                masks
            )

            # ======================
            # METRICS
            # ======================

            running_dice += dice_score(
                outputs,
                masks
            )

            running_iou += iou_score(
                outputs,
                masks
            )

            running_acc += pixel_accuracy(
                outputs,
                masks
            )

            # ======================
            # ACCUMULATE LOSS
            # ======================

            running_loss += loss.item()

    # ==============================
    # EPOCH AVERAGES
    # ==============================

    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    epoch_dice = (
        running_dice /
        len(dataloader)
    )

    epoch_iou = (
        running_iou /
        len(dataloader)
    )

    epoch_acc = (
        running_acc /
        len(dataloader)
    )

    return (
        epoch_loss,
        epoch_dice,
        epoch_iou,
        epoch_acc
    )