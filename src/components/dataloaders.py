from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from src.logger import logging


class SemanticSegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation with polygon annotations.

    Args:
        directory: Path to images and labels folder
        processor: Image processor for transforming images

    Attributes:
        images_dir (Path): Path to the directory containing image files
        labels_dir (Path): Path to the directory containing label files
        processor: (AutoImageProcessor) Image processor for transforming images
        image_files (List[Path]): Sorted list of image file paths
    """

    def __init__(self, directory: Path, processor: AutoImageProcessor) -> None:
        """
        Args:
            directory: Path to images and labels folder
            processor: Image processor
        """
        super().__init__()
        self.images_dir = directory / "images"
        self.labels_dir = directory / "labels"
        self.processor = processor

        # Get all image files
        self.image_files = list(self.images_dir.glob("*.jpg"))
        self.image_files.sort()

        logging.info(f"Found {len(self.image_files)} images in {self.images_dir}")

    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in the dataset
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx: Index of the item to retrieve

        Returns:
            Dict[str, Any]: Dictionary containing pixel_values, mask, image_path, and original_image
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path)

        # Process image first to get the target size
        processed_image = self.processor(images=image, return_tensors="pt")
        pixel_values = processed_image["pixel_values"].squeeze(0)
        target_size = (pixel_values.shape[1], pixel_values.shape[2])  # (H, W)

        # Create segmentation mask at the target size
        mask = self._create_segmentation_mask(labels, target_size)
        mask_tensor = torch.from_numpy(mask).long()

        return {"pixel_values": pixel_values, "mask": mask_tensor, "image_path": img_path, "original_image": image}

    def _load_labels(self, labels_path: Path) -> List[Dict[str, Any]]:
        """
        Process label file to extract polygon annotations.

        Args:
            labels_path: Path to the labels text file

        Returns:
            List[Dict[str, Any]]: List of label dictionaries containing class_id and vertices
        """
        labels = []

        with open(labels_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            entries = line.strip().split()
            # Extract class id (first entry)
            class_id = int(entries[0])

            # Extract polygon coordinates (normalized between 0-1)
            coordinates = [float(i) for i in entries[1:]]
            x_coords = coordinates[::2]
            y_coords = coordinates[1::2]

            labels.append({"class_id": class_id, "vertices": list(zip(x_coords, y_coords))})

        return labels

    def _create_segmentation_mask(self, labels: List[Dict[str, Any]], target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create segmentation mask from polygon vertices.

        Args:
            labels: List of label dictionaries containing class_id and vertices
            target_size: Target dimensions (height, width) for the segmentation mask

        Returns:
            np.ndarray: Segmentation mask with class labels
        """
        target_height, target_width = target_size
        mask = np.zeros((target_height, target_width), dtype=np.uint8)

        for label in labels:
            class_id = label["class_id"]
            vertices = label["vertices"]

            # Convert normalized coordinates to target size pixel coordinates
            pixel_vertices = [(int(x * target_width), int(y * target_height)) for x, y in vertices]

            # Create polygon mask using PIL at target size
            mask_img = Image.new("L", (target_width, target_height), 0)
            ImageDraw.Draw(mask_img).polygon(pixel_vertices, fill=class_id + 1)  # +1 to avoid background class

            # Convert to numpy and add to main mask
            polygon_mask = np.array(mask_img)
            mask = np.maximum(mask, polygon_mask)

        return mask


def create_dataloaders(
    train_dir: Path, test_dir: Path, valid_dir: Path, processor: AutoImageProcessor = None, batch_size: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for training, testing, and validation datasets.

    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        valid_dir: Path to validation data directory
        processor: Image processor for transforming images
        batch_size: Number of samples per batch

    Returns:
        Tuple[DataLoader, DataLoader]: Train, test, and validation dataloaders
    """

    train_dataset = SemanticSegmentationDataset(train_dir, processor=processor)
    test_dataset = SemanticSegmentationDataset(test_dir, processor=processor)
    validation_dataset = SemanticSegmentationDataset(valid_dir, processor=processor)

    def collate_fn(batch) -> Dict[str, Any]:
        """
        Args:
            batch: List of dataset items to collate into a batch

        Returns:
            Dict[str, Any]: Batched data with pixel_values, masks, image_paths, and original_images
        """
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        image_paths = [item["image_path"] for item in batch]
        original_images = [item["original_image"] for item in batch]

        return {
            "pixel_values": pixel_values,
            "masks": masks,
            "image_paths": image_paths,
            "original_images": original_images,
        }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, validation_dataloader
