import torch
from src.metrics.metrics import calculate_metrics


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):
    """
    Run one full validation epoch.

    The val dataloader uses temporally strided
    frames (every 20th frame per clip), so each
    validation sample is meaningfully different
    from its neighbors. The Dice score reported
    here is a much more honest estimate of
    generalization than in v1.

    Returns:
        (epoch_loss, epoch_dice, epoch_iou, epoch_acc)
        All are scalar floats, averaged over all
        batches in the epoch.
    """

    # =====================================
    # EVAL MODE
    # =====================================
    #
    # model.eval() deactivates:
    # - Dropout2d (all channels active)
    # - GroupNorm uses running stats
    # We want deterministic inference here.

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    running_acc  = 0.0


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
            masks  = masks.to(device)


            # =============================
            # FORWARD PASS
            # =============================

            outputs = model(images)
            # outputs: [B, 13, 256, 256]


            # =============================
            # LOSS
            # =============================

            loss = criterion(outputs, masks)


            # =============================
            # METRICS
            # =============================

            acc, dice, iou = calculate_metrics(
                outputs, masks
            )

            running_acc  += acc
            running_dice += dice
            running_iou  += iou


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
    epoch_iou  = running_iou  / n
    epoch_acc  = running_acc  / n

    return (
        epoch_loss,
        epoch_dice,
        epoch_iou,
        epoch_acc
    )
