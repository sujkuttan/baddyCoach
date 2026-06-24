"""
Model loading and management shared by both colab and backend pipelines.

Provides lazy-loading getters for each model type. Stages call these instead
of importing model files directly. Standalone (colab/kaggle) environments get
empty results and handle absence themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

_models: dict[str, Any] = {}
_model_health: dict[str, dict] = {}


def _checked_load(model, state_dict, *, core_prefixes, max_missing_frac=0.05):
    """Load a state_dict with strict key checking.

    Returns a dict with status details so callers can decide whether
    the model is usable and persist the result in model_health.json.

    Args:
        model: The nn.Module instance.
        state_dict: The loaded checkpoint dict.
        core_prefixes: Tuple of key prefixes whose absence means "not loaded".
        max_missing_frac: Maximum fraction of missing keys allowed.

    Returns:
        dict with keys: loaded (bool), missing_frac (float), n_missing (int),
        n_unexpected (int), core_missing (list).
    """
    import torch
    incompat = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(incompat.missing_keys), list(incompat.unexpected_keys)
    total = sum(1 for _ in model.state_dict())
    missing_frac = len(missing) / max(total, 1)
    core_missing = [k for k in missing if any(k.startswith(p) for p in core_prefixes)]
    status = {
        "loaded": not core_missing and missing_frac <= max_missing_frac,
        "missing_frac": round(missing_frac, 4),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "core_missing": core_missing[:10],
    }
    return status


def get_model_health() -> dict[str, dict]:
    """Return the accumulated model health report for the current process."""
    return dict(_model_health)


def record_model_health(name: str, status: dict):
    """Record health status for a model."""
    _model_health[name] = status


def _get_settings():
    from app.config.settings import settings
    return settings


def _get_device():
    return _get_settings().device


def _download_model_from_gdown(url: str, output_path: Path) -> bool:
    """Download model from Google Drive."""
    try:
        import gdown
        gdown.download(id=url, output=str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def _extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract zip file."""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def get_yolov8():
    if "yolov8" not in _models:
        try:
            from app.models.yolov8 import YOLOv8Tracker
            s = _get_settings()
            path = str(s.yolov8_model_path) if s.yolov8_model_path else None
            _models["yolov8"] = YOLOv8Tracker(path, conf_threshold=0.5, device=_get_device())
        except ImportError:
            logger.warning("YOLOv8 not available (standalone mode)")
            return None
    return _models.get("yolov8")


def get_tracknet():
    if "tracknet" not in _models:
        try:
            from app.models.tracknet import TrackNetV3
            s = _get_settings()
            path = str(s.tracknet_model_path)
            ipath = str(s.inpaintnet_model_path) if s.inpaintnet_model_path and Path(s.inpaintnet_model_path).exists() else None
            _models["tracknet"] = TrackNetV3(path, device=_get_device(), inpaintnet_path=ipath)
        except ImportError:
            logger.warning("TrackNet not available (standalone mode)")
            return None
    return _models.get("tracknet")


def get_rtmpose():
    if "rtmpose" not in _models:
        try:
            from app.models.rtmpose import RTMPoseEstimator
            s = _get_settings()
            path = str(s.rtmpose_model_path) if s.rtmpose_model_path else None
            _models["rtmpose"] = RTMPoseEstimator(path, device=_get_device())
        except ImportError:
            logger.warning("RTMPose not available (standalone mode)")
            return None
    return _models.get("rtmpose")


def get_bst():
    if "bst" not in _models:
        try:
            from app.models.bst import BSTClassifier
            s = _get_settings()
            path = str(s.bst_model_path) if s.bst_model_path else None
            _models["bst"] = BSTClassifier(path, device=_get_device())
        except ImportError:
            logger.warning("BST not available (standalone mode)")
            return None
    return _models.get("bst")


def setup_models(device: str | None = None, pose_model: str = "rtmpose") -> dict:
    """Set up all models for the pipeline.

    Safe to call from standalone Colab/Kaggle environments: if the backend
    modules aren't importable, returns an empty ``models`` dict.

    Args:
        device: Device to use ('cuda' or 'cpu'). Defaults to settings.device.
        pose_model: Pose model to use ('rtmpose' or 'mmpose').

    Returns:
        Dictionary of loaded models.
    """
    if device is None:
        device = _get_device()

    models = {}
    try:
        from app.config.settings import settings as s

        yolo = get_yolov8()
        if yolo:
            models["yolov8"] = yolo

        tn = get_tracknet()
        if tn:
            models["tracknet"] = tn

        if pose_model == "mmpose":
            hr = get_hrnet()
            if hr:
                models["pose"] = hr
        else:
            rtp = get_rtmpose()
            if rtp:
                models["pose"] = rtp

        bst = get_bst()
        if bst:
            models["bst"] = bst

    except ImportError as e:
        logger.warning(f"Model imports not available (expected in standalone mode): {e}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

    return models
