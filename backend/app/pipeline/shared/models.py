"""
Model loading and management shared by both colab and backend pipelines.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def setup_models(device: str, pose_model: str = "rtmpose") -> dict:
    """
    Set up all models for the pipeline.

    Safe to call from standalone Colab/Kaggle environments: if the backend
    ``app.models`` / ``app.config.settings`` modules aren't importable, the
    function returns an empty ``models`` dict instead of raising.

    Args:
        device: Device to use ('cuda' or 'cpu')
        pose_model: Pose model to use ('rtmpose' or 'mmpose')

    Returns:
        Dictionary of loaded models
    """
    models = {}

    try:
        from app.models.yolov8 import YOLOv8Tracker
        from app.models.tracknet import TrackNetV3
        from app.models.rtmpose import RTMPoseEstimator

        from app.config.settings import settings

        yolo_path = str(settings.yolov8_model_path) if settings.yolov8_model_path else None
        models["yolov8"] = YOLOv8Tracker(yolo_path, conf_threshold=0.5, device=device)

        tracknet_path = str(settings.tracknet_model_path)
        models["tracknet"] = TrackNetV3(tracknet_path, device=device)

        rtmpose_path = str(settings.rtmpose_model_path) if settings.rtmpose_model_path else None
        models["rtmpose"] = RTMPoseEstimator(rtmpose_path, device=device)

    except ImportError as e:
        # Standalone environments (Colab/Kaggle) won't have the backend app
        # modules. Return early with empty dict — caller decides how to handle.
        logger.warning(f"Model imports not available (expected in standalone mode): {e}")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

    return models

def _download_model_from_gdown(url: str, output_path: Path) -> bool:
    """Download model from Google Drive."""
    try:
        gdown.download(id=url, output=str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def _extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False