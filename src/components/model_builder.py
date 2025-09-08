import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOSegmentation(nn.Module):
    """
    A segmentation model that uses a pre-trained DINO model as a feature extractor
    and adds a convolutional segmentation head for pixel-level classification.

    Attributes:
        backbone: The frozen pre-trained DINO model
        num_classes: Number of segmentation classes
        feature_dim: Dimension of DINO features (hidden_size from config)
        head_hidden_dim: Hidden dimension for the segmentation head
        segmentation_head: Sequential convolutional layers for classification
    """

    def __init__(self, dino_model: nn.Module, num_classes: int = 4, head_hidden_dim: int = 256) -> None:
        """
        Initialize the DINO segmentation model.

        Args:
            dino_model: Pre-trained DINO model (e.g., from transformers library)
                       Must have a config.hidden_size attribute
            num_classes: Number of segmentation classes including background.
                        Typically 4 for foreground/background + additional classes
            head_hidden_dim: Hidden dimension for the segmentation head layers.
                           Should match DINO head dimension (256 for base model)
        """
        super().__init__()
        self.backbone = dino_model
        self.num_classes = num_classes
        self.feature_dim = dino_model.config.hidden_size
        self.head_hidden_dim = head_hidden_dim

        # Freeze DINO parameters for feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.head_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.head_hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.head_hidden_dim, self.head_hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.head_hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(self.head_hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DINO segmentation model.

        Args:
            pixel_values: Input images tensor of shape [B, C, H, W] where:
                         - B: batch size
                         - C: number of channels (typically 3 for RGB)
                         - H, W: height and width (typically 224x224)

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, H, W]
                         where each pixel has logits for each class
        """
        # Get DINO features
        outputs = self.backbone(pixel_values)

        # Get patch embeddings (exclude CLS token)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, feature_dim]

        batch_size = patch_embeddings.shape[0]
        num_patches = patch_embeddings.shape[1]
        feature_dim = patch_embeddings.shape[2]

        # Use reshape and contiguous() to handle non-contiguous tensors caused by last_hidden_state slicing
        patch_size = int(np.sqrt(num_patches))
        patch_embeddings = patch_embeddings.reshape(batch_size, patch_size, patch_size, feature_dim)
        patch_embeddings = patch_embeddings.permute(0, 3, 1, 2).contiguous()  # [B, feature_dim, H, W]

        # Apply segmentation head
        logits = self.segmentation_head(patch_embeddings)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)

        return logits
