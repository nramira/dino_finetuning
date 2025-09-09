import torch
from transformers import AutoImageProcessor, AutoModel

from src import config, visualizations
from src.components import dataloaders, engine, evaluation, loss, model_builder


def run_pipeline(cfg: config.TrainingConfig = config.default_config) -> None:
    """Main training function"""

    # Initialize processor and model
    processor = AutoImageProcessor.from_pretrained(cfg.base_model_name, use_fast=True)
    dino = AutoModel.from_pretrained(cfg.base_model_name)

    # Create dataloaders
    train_dataloader, test_dataloader, validation_dataloader = dataloaders.create_dataloaders(
        train_dir=cfg.train_dir,
        test_dir=cfg.test_dir,
        valid_dir=cfg.valid_dir,
        processor=processor,
        batch_size=cfg.batch_size,
    )

    # Wrap model with segmentation head
    model = model_builder.DINOSegmentation(
        dino_model=dino, num_classes=cfg.num_classes, head_hidden_dim=cfg.head_hidden_dim
    )

    # Device agnostic setup
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Initialize loss and optimizer
    criterion = loss.CombinedLoss(ce_weight=cfg.cross_entropy_weight, dice_weight=cfg.dice_weight)
    optimizer = torch.optim.Adam(model.segmentation_head.parameters(), lr=cfg.learning_rate)
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
        num_epochs=cfg.num_epochs,
        model_name=cfg.model_name,
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
