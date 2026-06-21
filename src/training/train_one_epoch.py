import torch
from src.metrics.metrics import calculate_metrics


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Run one full training epoch.

    The dataloader uses VideoBalancedSampler,
    so each batch already contains frames from
    multiple different surgical videos. No
    changes needed here — the diversity comes
    from the sampler upstream.

    Returns:
        (epoch_loss, epoch_dice, epoch_iou, epoch_acc)
        All are scalar floats, averaged over all
        batches in the epoch.
    """

    # =====================================
    # TRAIN MODE
    # =====================================
    #
    # model.train() activates:
    # - Dropout2d (randomly drops channels)
    # - GroupNorm in training mode
    # Both are critical for regularization.

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    running_acc  = 0.0


    # =====================================
    # LOOP OVER BATCHES
    # =====================================

    for images, masks in dataloader:


        # =================================
        # MOVE TO DEVICE
        # =================================

        images = images.to(device)
        masks  = masks.to(device)


        # =================================
        # CLEAR OLD GRADIENTS
        # =================================

        optimizer.zero_grad()


        # =================================
        # FORWARD PASS
        # =================================

        outputs = model(images)
        # outputs: [B, 13, 256, 256] — raw logits


        # =================================
        # LOSS
        # =================================

        loss = criterion(outputs, masks)


        # =================================
        # METRICS
        # =================================
        #
        # calculate_metrics uses hard argmax
        # predictions (not softmax probs) to
        # compute Dice, IoU, and pixel accuracy.
        # Computed with torch.no_grad() not
        # needed here since we detach inside,
        # but the loss.backward() call below
        # only touches the loss graph.

        acc, dice, iou = calculate_metrics(
            outputs, masks
        )

        running_acc  += acc
        running_dice += dice
        running_iou  += iou


        # =================================
        # BACKPROPAGATION
        # =================================

        loss.backward()


        # =================================
        # GRADIENT CLIPPING
        # =================================
        #
        # Clips gradient norm to 1.0.
        # Prevents occasional exploding
        # gradients that can destabilize
        # training, especially early on when
        # the model is far from convergence.

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


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
    epoch_iou  = running_iou  / n
    epoch_acc  = running_acc  / n

    return (
        epoch_loss,
        epoch_dice,
        epoch_iou,
        epoch_acc
    )
