from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoImageProcessor, AutoModel

from src import config
from src.components import model_builder


def save_model(model: torch.nn.Module, target_dir: Path, model_name: str) -> None:
    """
    Saves a PyTorch model's state dictionary to a target directory.

    Args:
        model (torch.nn.Module): A PyTorch model to save. The model's state_dict()
            will be saved, not the entire model object.
        target_dir (Path): A directory path for saving the model to. Will be created
            if it doesn't exist.
        model_name (str): A filename for the saved model. Must include either
            ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(cfg: config.TrainingConfig) -> Tuple[torch.nn.Module, AutoImageProcessor]:
    """
    Loads a PyTorch model's state dictionary from a file.

    Args:
        cfg (config.TrainingConfig): Configuration object containing model parameters
    Returns:
        Tuple[torch.nn.Module, AutoImageProcessor]: Loaded model and image processor
    """
    # Load base model and image processor
    processor = AutoImageProcessor.from_pretrained(cfg.base_model_name, use_fast=True)
    dino = AutoModel.from_pretrained(cfg.base_model_name, token=cfg.huggingface_token)

    # Wrap base model in segmentation model
    model = model_builder.DINOSegmentation(
        dino_model=dino,
        token_offset=cfg.token_offset,
        num_classes=cfg.num_classes,
        head_hidden_dim=cfg.head_hidden_dim,
        dropout=cfg.dropout,
        image_size=cfg.image_size,
    )

    # Load weights
    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model file not found at {cfg.model_path}")
    model.load_state_dict(torch.load(f=cfg.model_path))

    return model, processor
