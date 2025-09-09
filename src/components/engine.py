from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import metrics, utils
from src.logger import logging


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model to train
        dataloader (DataLoader): Training data loader containing batches of data
        criterion (nn.Module): Loss function for computing training loss
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        device (torch.device): Device to run computations on (CPU or GPU)

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing:
            - avg_loss (float): Average loss across all batches
            - avg_dice (Dict[str, float]): Dictionary with average dice scores
              including 'dice_mean' and per-class scores
    """

    model.train()
    total_loss = 0
    all_dice_scores = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(pixel_values)
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate dice score for each class
        with torch.inference_mode():
            dice_scores = metrics.dice_score(logits, masks)
            all_dice_scores.append(dice_scores)

    # Average metrics
    avg_dice = {}
    for key in all_dice_scores[0].keys():
        avg_dice[key] = np.mean([d[key] for d in all_dice_scores])

    avg_loss = total_loss / len(dataloader)

    return avg_loss, avg_dice


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model for one epoch.

    Args:
        model (nn.Module): The neural network model to evaluate
        dataloader (DataLoader): Validation/test data loader containing batches of data
        criterion (nn.Module): Loss function for computing validation loss
        device (torch.device): Device to run computations on (CPU or GPU)

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing:
            - avg_loss (float): Average loss across all batches
            - avg_dice (Dict[str, float]): Dictionary with average dice scores
              including 'dice_mean' and per-class scores
    """

    model.eval()
    total_loss = 0
    all_dice_scores = []

    with torch.inference_mode():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            masks = batch["masks"].to(device)

            logits = model(pixel_values)
            loss = criterion(logits, masks)

            total_loss += loss.item()

            # Calculate metrics
            dice_scores = metrics.dice_score(logits, masks)
            all_dice_scores.append(dice_scores)

    # Average metrics
    avg_dice = {}
    for key in all_dice_scores[0].keys():
        avg_dice[key] = np.mean([d[key] for d in all_dice_scores])

    avg_loss = total_loss / len(dataloader)

    return avg_loss, avg_dice


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    num_epochs: int,
    model_name: str,
) -> Dict[str, Any]:
    """
    Reusable training function for segmentation models

    Args:
        model (nn.Module): The segmentation model to train
        train_dataloader (DataLoader): Training data loader containing image-mask pairs
        test_dataloader (DataLoader): Test/validation data loader for evaluation
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, DiceLoss)
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        scheduler (ReduceLROnPlateau): Learning rate scheduler that reduces LR on plateau
        device (torch.device): Device to train on (CPU or GPU)
        num_epochs (int, optional): Number of training epochs. Defaults to 5.
        model_name (str, optional): Name for saving model checkpoints.
            Defaults to "segmentation_model".

    Returns:
        Dict[str, Any]: Dictionary containing comprehensive training history:
            - train_losses (List[float]): Training loss per epoch
            - test_losses (List[float]): Validation loss per epoch
            - train_dice_scores (List[float]): Training dice scores per epoch
            - test_dice_scores (List[float]): Validation dice scores per epoch
            - best_test_dice (float): Best validation dice score achieved
            - best_epoch (int): Epoch number where best performance occurred
    """

    # Training history
    history = {
        "train_losses": [],
        "test_losses": [],
        "train_dice_scores": [],
        "test_dice_scores": [],
        "best_test_dice": 0.0,
        "best_epoch": 0,
    }

    best_test_dice = 0
    model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 40)

        # Train
        train_loss, train_dice = train_epoch(model, train_dataloader, criterion, optimizer, device)
        history["train_losses"].append(train_loss)
        history["train_dice_scores"].append(train_dice["dice_mean"])

        # Test
        test_loss, test_dice = test_epoch(model, test_dataloader, criterion, device)
        history["test_losses"].append(test_loss)
        history["test_dice_scores"].append(test_dice["dice_mean"])

        # Step scheduler
        scheduler.step(test_loss)

        # Print epoch results
        logging.info(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        logging.info(f"Train Dice: {train_dice['dice_mean']:.4f}, Test Dice: {test_dice['dice_mean']:.4f}")
        logging.info(
            f"Per-class Dice - Glioma: {test_dice['dice_glioma']:.3f}, "
            f"Meningioma: {test_dice['dice_meningioma']:.3f}, "
            f"Pituitary: {test_dice['dice_pituitary']:.3f}"
        )

        # Save best model
        if test_dice["dice_mean"] > best_test_dice:
            best_test_dice = test_dice["dice_mean"]
            history["best_test_dice"] = best_test_dice
            history["best_epoch"] = epoch + 1

            utils.save_model(model, target_dir=Path("models"), model_name=f"{model_name}.pth")

    return history
