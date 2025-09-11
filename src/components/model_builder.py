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
        dropout: Dropout rate for regularization in the segmentation head
        image_size: Input image size (assumed square, e.g., 224)
        patch_length: Number of patches along one dimension (image_size / patch_size)
        num_patches: Total number of patches (patch_length squared)
        token_offset: Number of tokens to skip (e.g., 1 for CLS token, 5 for CLS + 4 register tokens)
    """

    def __init__(
        self,
        dino_model: nn.Module,
        token_offset: int,
        num_classes: int,
        head_hidden_dim: int,
        dropout: float,
        image_size: int,
    ) -> None:
        """
        Initialize the DINO segmentation model.

        Args:
            dino_model: Pre-trained DINO model
            token_offset: Number of tokens to skip
            num_classes: Number of segmentation classes including background
            head_hidden_dim: Hidden dimension for the segmentation head layers
            dropout: Dropout rate for segmentation head
            image_size: Input image size
        """
        super().__init__()
        self.backbone = dino_model
        self.num_classes = num_classes
        self.feature_dim = dino_model.config.hidden_size
        self.head_hidden_dim = head_hidden_dim
        self.dropout = dropout
        self.image_size = image_size
        self.patch_length = self.image_size // dino_model.config.patch_size  # (dinov2: 16, dinov3: 14)
        self.num_patches = self.patch_length * self.patch_length
        self.token_offset = token_offset

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.head_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.head_hidden_dim, self.head_hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.head_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.head_hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DINO segmentation model.

        Args:
            pixel_values: Input images tensor of shape [B, C, H, W] where:
                         - B: batch size
                         - C: number of channels
                         - H, W: height and width (224x224 after processing)

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, H, W]
                         where each pixel has logits for each class
        """
        batch_size = pixel_values.shape[0]

        with torch.inference_mode():
            outputs = self.backbone(pixel_values)

        # Get patch embeddings (exclude CLS token and register tokens if present)
        hidden_state = outputs.last_hidden_state
        patch_embeddings = hidden_state.narrow(1, self.token_offset, self.num_patches)  # [B, num_patches, feature_dim]

        # Reshape to [B, feature_dim, H, W] to apply the segmentation head to the embedding vectors of each patch
        patch_embeddings = patch_embeddings.transpose(1, 2).contiguous()  # [B, feature_dim, num_patches]
        patch_embeddings = patch_embeddings.view(
            batch_size, self.feature_dim, self.patch_length, self.patch_length
        )  # [B, feature_dim, H, W]

        logits = self.segmentation_head(patch_embeddings)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        return logits
