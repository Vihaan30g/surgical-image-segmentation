from src.metrics.metrics import calculate_metrics
import torch

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Train model for one epoch using the centralized metrics calculator.
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
        
        # Calculate all metrics in one pass to ensure consistency
        acc, dice, iou = calculate_metrics(outputs, masks)

        running_acc += acc
        running_dice += dice
        running_iou += iou

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