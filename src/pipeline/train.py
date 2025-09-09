from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

from src import visualizations
from src.components import dataloaders, engine, evaluation, loss, model_builder

# Setup
TRAIN_DIR: Path = Path("../data/BrainTumor/train")
TEST_DIR: Path = Path("../data/BrainTumor/test")
VALID_DIR: Path = Path("../data/BrainTumor/valid")
CLASSES_NAMES: list[str] = ["background", "glioma", "meningioma", "pituitary"]
BATCH_SIZE: int = 8
NUM_EPOCHS: int = 5
HEAD_HIDDEN_DIM: int = 256
CROSS_ENTROPY_WEIGHT: float = 0.4
DICE_WEIGHT: float = 0.6
LEARNING_RATE: float = 1e-4
BASE_MODEL_NAME: str = "facebook/dinov2-base"


def run_pipeline() -> None:
    """Main training function"""

    # Initialize processor and model
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    dino = AutoModel.from_pretrained(BASE_MODEL_NAME)

    # Create dataloaders
    train_dataloader, test_dataloader, validation_dataloader = dataloaders.create_dataloaders(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR, valid_dir=VALID_DIR, processor=processor, batch_size=BATCH_SIZE
    )

    # Wrap model with segmentation head
    model = model_builder.DINOSegmentation(
        dino_model=dino, num_classes=len(CLASSES_NAMES), head_hidden_dim=HEAD_HIDDEN_DIM
    )

    # Device agnostic setup
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Initialize loss and optimizer
    criterion = loss.CombinedLoss(ce_weight=CROSS_ENTROPY_WEIGHT, dice_weight=DICE_WEIGHT)
    optimizer = torch.optim.Adam(model.segmentation_head.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Training loop
    history = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        model_name=f"{BASE_MODEL_NAME}_segmentation",
    )

    # Visualize loss, dice curves and some predictions
    visualizations.plot_training_history(history, title="DINO Segmentation Training History")
    visualizations.visualize_batch(validation_dataloader, num_samples=4, alpha=0.5, model=model)

    # Final evaluation on validation set
    final_metrics = evaluation.evaluate_on_dataloader(
        model=model, dataloader=validation_dataloader, criterion=criterion, device=device
    )

    return final_metrics


if __name__ == "__main__":
    run_pipeline()
