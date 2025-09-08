import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1])  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Flatten
        predictions_flat = predictions.reshape(-1)  # [B*C*H*W]
        targets_flat = targets_one_hot.reshape(-1)  # [B*C*H*W]

        # Calculate dice
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions_flat.sum() + targets_flat.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Weighted sum of CE and Dice losses
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        return self.ce_weight * ce + self.dice_weight * dice
