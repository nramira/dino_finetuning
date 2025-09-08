from typing import Dict

import numpy as np
import torch


def dice_score(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
    """Calculate Dice score per class"""
    predictions = torch.argmax(predictions, dim=1)
    dice_scores = {}

    for class_id in range(num_classes):
        pred_mask = (predictions == class_id).float()
        target_mask = (targets == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        if union > 0:
            dice = (2.0 * intersection) / union
            dice_value = dice.item()
        else:
            dice_value = 1.0 if pred_mask.sum() == 0 else 0.0

        class_name = ["background", "glioma", "meningioma", "pituitary"][class_id]
        dice_scores[f"dice_{class_name}"] = dice_value

    dice_scores["dice_mean"] = np.mean(list(dice_scores.values()))

    return dice_scores
