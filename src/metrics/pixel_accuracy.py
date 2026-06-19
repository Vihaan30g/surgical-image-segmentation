import torch


def pixel_accuracy(
    logits,
    targets
):

    preds = torch.argmax(
        logits,
        dim=1
    )

    correct = (
        preds == targets
    ).sum()

    total = targets.numel()

    accuracy = (
        correct.float() /
        total
    )

    return accuracy.item()