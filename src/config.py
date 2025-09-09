from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.

    Attributes:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        valid_dir: Path to validation data directory
        classes_names: List of class names for segmentation
        batch_size: Number of samples per batch
        num_epochs: Number of training epochs
        head_hidden_dim: Hidden dimension size for segmentation head
        cross_entropy_weight: Weight for cross entropy loss in combined loss
        dice_weight: Weight for dice loss in combined loss
        learning_rate: Learning rate for optimizer
        base_model_name: Pretrained model name from HuggingFace
        model_save_dir: Directory to save trained models
    """

    # Data paths
    train_dir: Path = Path("data/BrainTumor/train")
    test_dir: Path = Path("data/BrainTumor/test")
    valid_dir: Path = Path("data/BrainTumor/valid")

    # Model configuration
    classes_names: List[str] = None
    base_model_name: str = "facebook/dinov2-base"
    head_hidden_dim: int = 256

    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4

    # Loss configuration
    cross_entropy_weight: float = 0.4
    dice_weight: float = 0.6

    # Output configuration
    model_save_dir: Path = Path("models")

    def __post_init__(self) -> None:
        """Initialize default values and validate configuration."""
        if self.classes_names is None:
            self.classes_names = ["background", "glioma", "meningioma", "pituitary"]

        # Validation
        if self.cross_entropy_weight + self.dice_weight != 1.0:
            raise ValueError("Cross entropy weight + dice weight must equal 1.0")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

    @property
    def num_classes(self) -> int:
        """Returns the number of classes."""
        return len(self.classes_names)

    @property
    def model_name(self) -> str:
        """Returns formatted model name for saving."""
        return f"{self.base_model_name.replace('/', '_')}_segmentation"

    @property
    def model_path(self) -> Path:
        """Returns the full path to save the trained model."""
        return self.model_save_dir / f"{self.model_name}.pth"

    @property
    def device(self) -> str:
        """Returns the device to be used for training and inference."""
        return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# Create default configuration instance
default_config = TrainingConfig()
