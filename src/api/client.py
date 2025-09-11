from pathlib import Path
from typing import Any, Dict, Optional

import requests


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
