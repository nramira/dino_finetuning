import torch
from transformers import AutoImageProcessor, AutoModel

from src import config, visualizations
from src.components import dataloaders, engine, evaluation, loss, model_builder
from src.logger import logging


def run_pipeline(cfg: config.TrainingConfig = config.default_config) -> None:
    """Main training function"""

    logging.info("Starting training pipeline...")

    processor = AutoImageProcessor.from_pretrained(cfg.base_model_name, use_fast=True)
    dino = AutoModel.from_pretrained(cfg.base_model_name)
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
        dino_model=dino, num_classes=cfg.num_classes, head_hidden_dim=cfg.head_hidden_dim
    )
    logging.info("Created segmentation model")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using device: {device}")

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
        device=device,
        num_epochs=cfg.num_epochs,
        model_name=cfg.model_name,
    )
    logging.info("Training complete")
    logging.info(f"Model saved to: models/{cfg.model_name}.pth")

    visualizations.plot_training_history(history, title="DINO Segmentation Training History")
    visualizations.visualize_batch(validation_dataloader, num_samples=5, alpha=0.5, model=model)
    logging.info("Plotted training history and visualized batch")

    evaluation.evaluate_on_dataloader(model=model, dataloader=validation_dataloader, criterion=criterion, device=device)
    logging.info("Training pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
