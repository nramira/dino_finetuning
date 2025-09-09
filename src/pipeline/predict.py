import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src import config, utils, visualizations
from src.components import model_builder


def predict(image_path: Path, cfg: config.TrainingConfig = config.default_config) -> Image.Image:
    """
    Perform segmentation prediction on a single image using a trained model.

    Args:
        image_path (Path): Path to the input image file to segment
        cfg (config.TrainingConfig, optional): Configuration object containing
            model parameters and paths. Defaults to config.default_config.

    Returns:
        Image.Image: PIL Image with segmentation mask overlaid on the original image.
            Different tumor classes are colored: red (glioma), green (meningioma),
            blue (pituitary).
    """
    # Load base model and image processor
    processor = AutoImageProcessor.from_pretrained(cfg.base_model_name, use_fast=True)
    dino = AutoModel.from_pretrained(cfg.base_model_name)

    # Wrap base model in segmentation model
    model = model_builder.DINOSegmentation(
        dino_model=dino, num_classes=cfg.num_classes, head_hidden_dim=cfg.head_hidden_dim
    )

    # Load weights
    model_dir = cfg.model_save_dir / cfg.model_name
    assert model_dir.exists(), f"Model directory {model_dir} does not exist."
    model = utils.load_model(model=model, target_dir=model_dir, device=cfg.device)

    # Preprocess image
    image = Image.open(image_path)
    image = processor(images=image, return_tensors="pt").unsqueeze(0).to(cfg.device)

    # Pass image through model
    model.to(cfg.device)
    model.eval()
    with torch.inference_mode():
        logits = model(image["pixel_values"].to(cfg.device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    return visualizations.overlay_mask(image, preds, alpha=0.5)


if __name__ == "__main__":
    """
    Demonstration script for running prediction on a random validation image.
    """
    cfg = config.default_config
    validation_images = list(cfg.valid_dir).glob("*.jpg")
    masked_image = predict(image_path=random.choice(validation_images), cfg=cfg)

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
