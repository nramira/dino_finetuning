import random
from typing import Any, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader


def plot_training_history(history: Dict[str, Any], title: str = "Training History") -> None:
    """
    Plot training curves from training history dictionary.

    Args:
        history (Dict[str, Any]): Dictionary containing training history with keys:
            - "train_losses": List of training losses per epoch
            - "test_losses": List of validation/test losses per epoch
            - "train_dice_scores": List of training dice scores per epoch
            - "test_dice_scores": List of validation/test dice scores per epoch
        title (str, optional): Overall title for the plot. Defaults to "Training History".
    Returns:
        None: Displays the training curves plot.
    """

    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label="Train Loss", marker="o")
    plt.plot(history["test_losses"], label="Test Loss", marker="s")
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Dice score plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_dice_scores"], label="Train Dice", marker="o")
    plt.plot(history["test_dice_scores"], label="Test Dice", marker="s")
    plt.title("Training and Test Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    """
    Create an overlay of segmentation mask on top of an image.

    Args:
        image (Image.Image): Original PIL image to overlay mask on
        mask (np.ndarray): 2D numpy array with integer class labels (0=background,
            1=glioma, 2=meningioma, 3=pituitary)
        alpha (float, optional): Transparency level for the overlay (0=transparent,
            1=opaque).

    Returns:
        Image.Image: PIL image with the colored mask overlay applied

    Note:
        Color mapping:
        - Class 1 (Glioma): Red [255, 0, 0]
        - Class 2 (Meningioma): Green [0, 255, 0]
        - Class 3 (Pituitary): Blue [0, 0, 255]
    """

    # Resize image to match mask size
    img = np.array(image.resize((mask.shape[1], mask.shape[0])))

    # Create colored overlay
    overlay = np.zeros_like(img)
    colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}  # Red, Green, Blue
    for class_id, color in colors.items():
        overlay[mask == class_id] = color

    # Blend manually
    result = (1 - alpha) * img + alpha * overlay

    return Image.fromarray(result.astype(np.uint8))


def visualize_batch(dataloader: DataLoader, num_samples: int = 4, alpha: float = 0.5, model: nn.Module = None) -> None:
    """
    Visualize images with overlaid segmentation masks and optionally model predictions.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset to visualize.
            The dataset should have "original_images", "masks", and "pixel_values" keys.
        num_samples (int, optional): Number of random samples to display. Defaults to 4.
        alpha (float, optional): Transparency level for mask overlays (0=transparent,
            1=opaque). Defaults to 0.5.
        model (nn.Module, optional): Trained model for generating predictions. If provided,
            a third row showing model predictions will be added. Defaults to None.

    Note:
        - Creates a 2-row plot if model is None (original + ground truth)
        - Creates a 3-row plot if model is provided (original + ground truth + predictions)
        - Includes a color legend for the tumor classes
        - Randomly samples from the dataset to show variety
    """

    # Get random samples from the dataloader's dataset
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))

    # Setup plot
    rows = 2 if model is None else 3
    fig, axes = plt.subplots(rows, num_samples, figsize=(4 * num_samples, 4 * rows))
    if num_samples == 1:
        axes = [axes]
        axes = axes.reshape(rows, 1)

    for i, idx in enumerate(random_indices):
        # Top row: Original images
        axes[0, i].imshow(dataset["original_images"][idx])
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        # Middle row: Overlaid images
        overlaid_image = overlay_mask(dataset["original_images"][idx], dataset["masks"][idx].numpy(), alpha=alpha)
        axes[1, i].imshow(overlaid_image)
        axes[1, i].set_title(f"Ground truth mask {i + 1}")
        axes[1, i].axis("off")

        if model is not None:
            # Bottom row: Model predictions
            device = next(model.parameters()).device
            model.eval()
            with torch.inference_mode():
                pixel_values = dataset["pixel_values"][idx].unsqueeze(0).to(device)
                logits = model(pixel_values)
                pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

                # Show the predicted mask
                axes[2, i].imshow(overlay_mask(dataset["original_images"][idx], pred_mask, alpha=alpha))
                axes[2, i].set_title(f"Predicted {i + 1}")
                axes[2, i].axis("off")

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="red", label="Glioma"),
        patches.Patch(facecolor="green", label="Meningioma"),
        patches.Patch(facecolor="blue", label="Pituitary"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()
