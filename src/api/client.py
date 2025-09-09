import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests
from PIL import Image


class SegmentationAPIClient:
    """
    Client class for interacting with the brain tumor segmentation API.

    Attributes:
        base_url (str): Base URL of the API server
        timeout (int): Request timeout in seconds
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        """
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        """
        Test API health endpoint.

        Returns:
            Dict[str, Any]: Health check response
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information from the API.

        Returns:
            Dict[str, Any]: Model information response
        """
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Model info request failed: {e}")
            return {"error": str(e)}

    def predict_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Send image to API for prediction.

        Args:
            image_path: Path to image file

        Returns:
            Optional[Dict[str, Any]]: Prediction response or None if failed
        """
        try:
            with open(image_path, "rb") as f:
                files = {"file": (image_path.name, f, "image/jpeg")}
                response = requests.post(f"{self.base_url}/predict", files=files, timeout=self.timeout)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Prediction request failed: {e}")
            return None
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            return None


def display_prediction_results(result: Dict[str, Any], original_image_path: Path) -> None:
    """
    Display prediction results with original and overlay images.

    Args:
        result: Prediction result from API
        original_image_path: Path to original image for comparison
    """
    if not result or not result.get("success", False):
        print("No valid results to display")
        return

    # Load original image
    original_image = Image.open(original_image_path)

    # Decode overlay image from base64
    overlay_data = base64.b64decode(result["overlay_image"])
    overlay_image = Image.open(io.BytesIO(overlay_data))

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Overlay image
    axes[1].imshow(overlay_image)
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis("off")

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="red", label="Glioma"),
        patches.Patch(facecolor="green", label="Meningioma"),
        patches.Patch(facecolor="blue", label="Pituitary"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()

    # Print class percentages
    print("\nClass Percentages:")
    for class_name, percentage in result["class_percentages"].items():
        print(f"  {class_name}: {percentage}%")
