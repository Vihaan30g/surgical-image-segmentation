import torch
from src.metrics.metrics import calculate_metrics

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validate model for one epoch using the centralized metrics calculator.
    """

    # =====================================
    # EVAL MODE
    # =====================================

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    # =====================================
    # NO GRADIENT FOR VALIDATION
    # =====================================

    with torch.no_grad():

        # =================================
        # LOOP OVER BATCHES
        # =================================

        for images, masks in dataloader:

            # =============================
            # MOVE TO DEVICE
            # =============================

            images = images.to(device)
            masks = masks.to(device)

            # =============================
            # FORWARD PASS
            # =============================

            outputs = model(images)

            # =============================
            # LOSS
            # =============================

            loss = criterion(
                outputs,
                masks
            )

            # =============================
            # METRICS
            # =============================

            acc, dice, iou = calculate_metrics(outputs, masks)

            running_acc += acc
            running_dice += dice
            running_iou += iou

            # =============================
            # LOSS ACCUMULATION
            # =============================

            running_loss += loss.item()

    # =====================================
    # EPOCH AVERAGES
    # =====================================

    n = len(dataloader)
    epoch_loss = running_loss / n
    epoch_dice = running_dice / n
    epoch_iou = running_iou / n
    epoch_acc = running_acc / n

    return (
        epoch_loss,
        epoch_dice,
        epoch_iou,
        epoch_acc
    )