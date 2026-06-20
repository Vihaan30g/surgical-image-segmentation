import torch
import torch.nn.functional as F

def calculate_metrics(logits, targets, num_classes=13):
    """
    Calculates Accuracy, Dice, and IoU in a single pass using batch-level 
    computations to perfectly align with the DiceLoss behavior.
    
    Args:
        logits: Tensor of shape [B, 13, H, W] (raw outputs from model)
        targets: Tensor of shape [B, H, W] (class indices 0-12)
    """
    # 1. Get hard predictions (Argmax)
    preds = torch.argmax(logits, dim=1)
    
    # 2. Calculate true Pixel Accuracy
    correct = (preds == targets).sum().float()
    total = targets.numel()
    accuracy = correct / total
    
    # 3. One-Hot Encode Predictions and Targets
    # Convert [B, H, W] to [B, 13, H, W]
    preds_oh = F.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
    targets_oh = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
    
    # 4. Compute Intersections and Sums over the ENTIRE BATCH (dim=0,2,3)
    # This aggregates all pixels across all images in the batch
    intersection = (preds_oh * targets_oh).sum(dim=(0, 2, 3))
    pred_sum = preds_oh.sum(dim=(0, 2, 3))
    target_sum = targets_oh.sum(dim=(0, 2, 3))
    
    dice_scores = []
    iou_scores = []
    
    # 5. Evaluate ONLY Foreground Classes (1 through 12)
    # We loop through classes 1-12, skipping the background (class 0)
    for c in range(1, num_classes):
        
        # We evaluate the class if it exists in the Ground Truth (target_sum > 0)
        # OR if the model predicted it (pred_sum > 0). 
        # This approach avoids the 'Empty Class Epsilon Trap'.
        if target_sum[c] > 0 or pred_sum[c] > 0:
            
            dice = (2. * intersection[c]) / (pred_sum[c] + target_sum[c] + 1e-6)
            iou = intersection[c] / (pred_sum[c] + target_sum[c] - intersection[c] + 1e-6)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            
    # Calculate Mean. If no foreground objects were present in the batch,
    # return 1.0 (indicating perfect background segmentation).
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 1.0
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 1.0
    
    return accuracy.item(), mean_dice.item(), mean_iou.item()
