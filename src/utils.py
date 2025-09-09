from pathlib import Path

import torch


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


def load_model(model: torch.nn.Module, target_dir: Path, device: torch.device) -> torch.nn.Module:
    """
    Loads a PyTorch model's state dictionary from a file and moves it to the specified device.

    Args:
        model (torch.nn.Module): A PyTorch model instance with the same architecture
            as the saved model. The state_dict will be loaded into this model.
        target_dir (Path): Path to the saved model file (.pth or .pt). This should
            be the complete file path, not just the directory.
        device (torch.device): The device to move the loaded model to (e.g., 'cpu', 'cuda').

    Returns:
        torch.nn.Module: The model with loaded weights, moved to the specified device.
    """
    # Load in the saved state_dict()
    model.load_state_dict(torch.load(f=target_dir))

    return model.to(device)
