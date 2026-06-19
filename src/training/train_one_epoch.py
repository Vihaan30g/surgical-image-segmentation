from src.metrics.dice_score import dice_score
from src.metrics.iou_score import iou_score
from src.metrics.pixel_accuracy import pixel_accuracy


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Train model for one epoch.

    Returns:
        epoch_loss,
        epoch_dice,
        epoch_iou,
        epoch_acc
    """

    # =====================================
    # TRAIN MODE
    # =====================================

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    # =====================================
    # LOOP OVER BATCHES
    # =====================================

    for images, masks in dataloader:

        # =================================
        # MOVE TO DEVICE
        # =================================

        images = images.to(device)

        masks = masks.to(device)

        # =================================
        # CLEAR OLD GRADIENTS
        # =================================

        optimizer.zero_grad()

        # =================================
        # FORWARD PASS
        # =================================

        outputs = model(images)

        # =================================
        # LOSS
        # =================================

        loss = criterion(
            outputs,
            masks
        )

        # =================================
        # METRICS
        # =================================

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

        # =================================
        # BACKPROPAGATION
        # =================================

        loss.backward()

        # =================================
        # WEIGHT UPDATE
        # =================================

        optimizer.step()

        # =================================
        # LOSS ACCUMULATION
        # =================================

        running_loss += loss.item()

    # =====================================
    # EPOCH AVERAGES
    # =====================================

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