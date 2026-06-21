import torch
from src.metrics.metrics import calculate_metrics


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou  = 0.0
    running_acc  = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):

        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        # =====================================
        # NaN GUARD
        # =====================================
        # If loss is NaN, something exploded.
        # Raise immediately with useful context
        # rather than silently accumulating NaN
        # for the whole epoch.

        if torch.isnan(loss):
            raise RuntimeError(
                f"NaN loss at batch {batch_idx}. "
                f"Check class weights, learning rate, "
                f"or gradient magnitudes. "
                f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]"
            )

        acc, dice, iou = calculate_metrics(outputs, masks)

        running_acc  += acc
        running_dice += dice
        running_iou  += iou

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

    n = len(dataloader)

    return (
        running_loss / n,
        running_dice / n,
        running_iou  / n,
        running_acc  / n,
    )