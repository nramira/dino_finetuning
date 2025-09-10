import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    The Dice coefficient is defined as:
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)

    Where X is the predicted segmentation and Y is the target segmentation.
    The loss is computed as 1 - Dice.

    Attributes:
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions (torch.Tensor): [B, C, H, W] logits from the model
            targets (torch.Tensor): [B, H, W] ground truth class indices

        Returns:
            torch.Tensor: Computed Dice loss (scalar tensor)
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
    """
    Combined Cross Entropy + Dice Loss.

    This loss function combines Cross Entropy Loss and Dice Loss to leverage the
    benefits of both loss functions. Cross Entropy Loss is effective for pixel-wise
    classification while Dice Loss is particularly good for handling class imbalance
    in segmentation tasks.

    Attributes:
        ce_weight (float): Weight factor for Cross Entropy Loss.
        dice_weight (float): Weight factor for Dice Loss.
        ce_loss: Instance of Cross Entropy Loss.
        dice_loss: Instance of Dice Loss.
    """

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            predictions (torch.Tensor): [B, C, H, W] logits from the model
            targets (torch.Tensor): [B, H, W] ground truth class indices

        Returns:
            torch.Tensor: Computed combined loss (scalar tensor)
        """
        # Weighted sum of CE and Dice losses
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        return self.ce_weight * ce + self.dice_weight * dice
