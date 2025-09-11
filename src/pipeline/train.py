import argparse

import torch
from transformers import AutoImageProcessor, AutoModel

from src import config, visualizations
from src.components import dataloaders, engine, evaluation, loss, model_builder
from src.logger import logging

# Setup ArgParser
parser = argparse.ArgumentParser(description="Fine-tune a DINO model for medical semantic segmentation.")
parser.add_argument(
    "--epochs",
    type=int,
    default=config.default_config.num_epochs,
    help="Number of training epochs.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=config.default_config.batch_size,
    help="Batch size for data loaders.",
)
parser.add_argument(
    "--head_hidden_dim",
    type=int,
    default=config.default_config.head_hidden_dim,
    help="Hidden dimension for the segmentation head.",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=config.default_config.dropout,
    help="Dropout rate for the segmentation head.",
)
parser.add_argument(
    "--ce_weight",
    type=float,
    default=config.default_config.cross_entropy_weight,
    help="Weight for cross entropy loss.",
)
parser.add_argument(
    "--dice_weight",
    type=float,
    default=config.default_config.dice_weight,
    help="Weight for dice loss.",
)
parser.add_argument(
    "--base_model_name",
    type=str,
    default=config.default_config.base_model_name,
    help="Pretrained model name from HuggingFace.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=config.default_config.learning_rate,
    help="Learning rate for optimizer.",
)

args = parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> config.TrainingConfig:
    """
    Create a TrainingConfig instance from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        config.TrainingConfig: Configuration object with values from arguments
    """
    return config.TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        cross_entropy_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        base_model_name=args.base_model_name,
        learning_rate=args.lr,
    )


def run_pipeline(cfg: config.TrainingConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Training configuration containing all hyperparameters and paths
    """

    logging.info("Starting training pipeline...")

    processor = AutoImageProcessor.from_pretrained(cfg.base_model_name, use_fast=True)
    dino = AutoModel.from_pretrained(cfg.base_model_name, token=cfg.huggingface_token)
    logging.info(f"Loaded pre-trained model and image processor from {cfg.base_model_name}")

    train_dataloader, test_dataloader, validation_dataloader = dataloaders.create_dataloaders(
        train_dir=cfg.train_dir,
        test_dir=cfg.test_dir,
        valid_dir=cfg.valid_dir,
        processor=processor,
        batch_size=cfg.batch_size,
    )
    logging.info("Created dataloaders")

    model = model_builder.DINOSegmentation(
        dino_model=dino,
        token_offset=cfg.token_offset,
        num_classes=cfg.num_classes,
        head_hidden_dim=cfg.head_hidden_dim,
        dropout=cfg.dropout,
        image_size=cfg.image_size,
    )
    logging.info("Created segmentation model")
    logging.info(f"Using device: {cfg.device}")

    criterion = loss.CombinedLoss(ce_weight=cfg.cross_entropy_weight, dice_weight=cfg.dice_weight)
    optimizer = torch.optim.Adam(model.segmentation_head.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    logging.info("Initialized loss function, optimizer, and scheduler")

    history = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.device,
        num_epochs=cfg.num_epochs,
        model_name=cfg.model_name,
    )
    logging.info("Training complete")
    logging.info(f"Model saved to: models/{cfg.model_name}.pth")

    visualizations.plot_training_history(history, title="DINO Segmentation Training History")
    visualizations.visualize_batch(validation_dataloader, num_samples=5, alpha=0.5, model=model)
    logging.info("Plotted training history and visualized batch")

    evaluation.evaluate_on_dataloader(
        model=model, dataloader=validation_dataloader, criterion=criterion, device=cfg.device
    )
    logging.info("Training pipeline finished.")


if __name__ == "__main__":
    # Create config from command line arguments
    training_config = create_config_from_args(args)

    # Run the training pipeline with the config
    run_pipeline(training_config)
