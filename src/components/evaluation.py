from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import config, metrics
from src.logger import logging


def evaluate_on_dataloader(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, Any]:
    """
    Comprehensive evaluation on dataset

    Args:
        model: Trained segmentation model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run on
        save_examples: Whether to save prediction examples

    Returns:
        Dict with metrics
    """
    model.eval()
    total_loss = 0
    all_dice_scores = []

    logging.info("Evaluating on dataset...")

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            masks = batch["masks"].to(device)

            # Forward pass
            logits = model(pixel_values)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            # Calculate metrics
            dice_scores = metrics.dice_score(logits, masks)
            all_dice_scores.append(dice_scores)

    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)

    # Aggregate dice scores
    final_dice = {}
    for key in all_dice_scores[0].keys():
        scores = [d[key] for d in all_dice_scores]
        final_dice[key] = {"mean": np.mean(scores), "std": np.std(scores), "min": np.min(scores), "max": np.max(scores)}

    # Print results
    logging.info("=" * 60)
    logging.info("RESULTS")
    logging.info("-" * 60)
    logging.info(f"Average Loss: {avg_loss:.4f}")
    logging.info(f"Overall Dice Score: {final_dice['dice_mean']['mean']:.4f} ± {final_dice['dice_mean']['std']:.4f}")
    logging.info("Per-Class Results:")
    logging.info("-" * 60)

    for i, class_name in enumerate(config.default_config.classes_names):
        dice_key = f"dice_{class_name}"
        if dice_key in final_dice:
            stats = final_dice[dice_key]
            logging.info(
                f"  {class_name.capitalize():12}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                f"(min: {stats['min']:.3f}, max: {stats['max']:.3f})"
            )

    logging.info("=" * 60)

    return {
        "avg_loss": avg_loss,
        "dice_scores": final_dice,
    }
