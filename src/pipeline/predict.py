import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from src import config, utils, visualizations


def process_single_image(
    image: Image.Image,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    cfg: config.TrainingConfig,
) -> Dict[str, Any]:
    """
    Perform segmentation prediction on a single PIL image.

    Args:
        image (Image.Image): Input image to segment
        model (torch.nn.Module): Trained segmentation model
        processor (AutoImageProcessor): Image processor for preprocessing
        cfg (config.TrainingConfig, optional): Configuration object containing
            model parameters and paths. Defaults to config.default_config.

    Returns:
        Dict[str, Any]: Dictionary containing prediction results with keys:
            - 'mask': numpy array of predicted segmentation mask
            - 'overlay': PIL Image with overlay visualization
            - 'class_percentages': dict of class percentages
    """
    # Preprocess image
    processed_image = processor(images=image, return_tensors="pt")
    pixels_values = processed_image["pixel_values"].to(cfg.device)

    # Pass image through model
    model.to(cfg.device)
    model.eval()
    with torch.inference_mode():
        logits = model(pixels_values)
        mask = torch.argmax(logits, dim=1).cpu().numpy()[0]  # Remove batch dimension

    # Create overlay visualization
    overlay_image = visualizations.overlay_mask(image, mask, alpha=0.5)

    # Calculate class percentages
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    class_percentages = {}

    for class_id, count in zip(unique, counts):
        class_name = cfg.classes_names[class_id]
        percentage = (count / total_pixels) * 100
        class_percentages[class_name] = round(percentage, 2)

    return {"mask": mask, "overlay": overlay_image, "class_percentages": class_percentages}


def predict(image_path: Path, cfg: config.TrainingConfig = config.default_config) -> Image.Image:
    """
    Perform segmentation prediction on a single image file using a trained model.

    Args:
        image_path (Path): Path to the input image file to segment
        cfg (config.TrainingConfig, optional): Configuration object containing
            model parameters and paths. Defaults to config.default_config.

    Returns:
        Image.Image: PIL Image with segmentation mask overlaid on the original image.
            Different tumor classes are colored: red (glioma), green (meningioma),
            blue (pituitary).
    """
    # Load model and processor
    model, processor = utils.load_model(cfg)

    # Load and predict on image
    image = Image.open(image_path)
    result = process_single_image(image, model, processor, cfg)

    return result["overlay"]


if __name__ == "__main__":
    """
    Demonstration script for running prediction on a random validation image.
    """
    # Select random validation image
    cfg = config.default_config
    validation_images = list(cfg.valid_dir).glob("*.jpg")
    random_sample = random.choice(validation_images)

    # Predict on the random sample
    masked_image = predict(random_sample, cfg)

    plt.figure(figsize=(10, 10))
    legend_elements = [
        patches.Patch(facecolor="red", label="Glioma"),
        patches.Patch(facecolor="green", label="Meningioma"),
        patches.Patch(facecolor="blue", label="Pituitary"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.imshow(masked_image)
    plt.axis("off")
    plt.show()
