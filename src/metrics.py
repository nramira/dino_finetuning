from typing import Dict

import numpy as np
import torch

from src import config


def dice_score(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate Dice score per class for multi-class segmentation.

    The Dice coefficient measures the overlap between predicted and target segmentation masks.
    It ranges from 0 (no overlap) to 1 (perfect overlap). For each class, the Dice score
    is computed as: 2 * |intersection| / (|pred| + |target|).

    Args:
        predictions (torch.Tensor): Model predictions with shape (batch_size, num_classes, height, width).
        targets (torch.Tensor): Ground truth segmentation masks with shape (batch_size, height, width).

    Returns:
        Dict[str, float]: Dictionary containing Dice scores with the following keys:
            - "dice_background": Dice score for background class
            - "dice_glioma": Dice score for glioma class
            - "dice_meningioma": Dice score for meningioma class
            - "dice_pituitary": Dice score for pituitary class
            - "dice_mean": Average Dice score across all classes
    """
    predictions = torch.argmax(predictions, dim=1)
    dice_scores = {}

    for class_id in range(config.default_config.num_classes):
        pred_mask = (predictions == class_id).float()
        target_mask = (targets == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        if union > 0:
            dice = (2.0 * intersection) / union
            dice_value = dice.item()
        else:
            dice_value = 1.0 if pred_mask.sum() == 0 else 0.0

        class_name = config.default_config.classes_names[class_id]
        dice_scores[f"dice_{class_name}"] = dice_value

    dice_scores["dice_mean"] = np.mean(list(dice_scores.values()))

    return dice_scores
