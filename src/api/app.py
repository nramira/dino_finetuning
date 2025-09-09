import base64
import io
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src import config, utils
from src.pipeline import predict

# Global variables for model and processor
model = None
processor = None
cfg = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.

    Args:
        app: FastAPI application instance
    """
    global model, processor, cfg

    # Startup
    try:
        cfg = config.default_config
        model, processor = utils.load_model(cfg)
        print("API startup completed successfully")
    except Exception as e:
        print(f"Failed to load model during startup: {str(e)}")
        raise

    yield

    # Shutdown (cleanup if needed)
    print("API shutting down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="API for semantic segmentation of brain tumors using fine-tuned DINO model",
    version="1.0.0",
    lifespan=lifespan,
)


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image to convert

    Returns:
        str: Base64 encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API information.

    Returns:
        Dict: Basic API information
    """
    return {
        "message": "Brain Tumor Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for segmentation",
            "/health": "GET - Check API health status",
            "/model-info": "GET - Get model information",
        },
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Dict: Health status information
    """
    model_loaded = model is not None and processor is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": cfg.device if cfg else "unknown",
    }


@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    """
    Get model information.

    Returns:
        Dict: Model configuration and class information
    """
    if cfg is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "base_model": cfg.base_model_name,
        "num_classes": cfg.num_classes,
        "class_names": cfg.classes_names,
        "head_hidden_dim": cfg.head_hidden_dim,
        "device": cfg.device,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Perform brain tumor segmentation on uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        JSONResponse: Prediction results including segmentation overlay and class percentages
    """
    global model, processor, cfg

    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read and process uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Perform prediction using the pipeline function
        result = predict.process_single_image(image, model, processor, cfg)

        # Format response
        response = {
            "success": True,
            "class_percentages": result["class_percentages"],
            "overlay_image": image_to_base64(result["overlay"]),
            "image_shape": list(result["mask"].shape),
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
