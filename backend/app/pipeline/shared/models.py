"""
Model loading and management shared by both colab and backend pipelines.

Provides lazy-loading getters for each model type. Stages call these instead
of importing model files directly. Standalone (colab/kaggle) environments get
empty results and handle absence themselves.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

_models: dict[str, Any] = {}
_model_health: dict[str, dict] = {}

CKPT_DIR = Path("ckpts")

# ─── Centralized model registry ────────────────────────────────────────────
# (local_path, gdrive_id, alt_download_url_or_None)
MODEL_REGISTRY: dict[str, tuple[Path, str | None, str | None]] = {
    "tracknet": (
        CKPT_DIR / "TrackNet_best.pt",
        "1rhKXbff1GITgrFTYptW6gAvWZ76E_qzp",
        None,
    ),
    "inpaintnet": (
        CKPT_DIR / "InpaintNet_best.pt",
        None,
        None,
    ),
    "bst": (
        CKPT_DIR / "bst" / "bst_CG_AP.pt",
        "1oM2cGM4gQRDXpcS3J5lIMDY2sBJlUvJ4",
        None,
    ),
    "bst_colab": (
        CKPT_DIR / "bst" / "bst_CG_JnB_bone_merged.pt",
        "1yHLpW4s8Rk8FYIUKF_NvC29Z8b8XuDq2",
        None,
    ),
    "rtmpose": (
        CKPT_DIR / "rtmpose" / "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx",
        None,
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    ),
    "rtmpose_colab": (
        CKPT_DIR / "rtmpose" / "rtmpose-m_8xb64-270e_coco-256x192.onnx",
        "1XjwDxz1a8i3WO6afuvaq-y3HPiFh48SN",
        None,
    ),
    "hrnet": (
        CKPT_DIR / "mmpose" / "hrnet_w32_coco_256x192.onnx",
        "1LFUEbHB-D3WCyjzf9aSJ_V_kVB8igsnr",
        None,
    ),
    "court_kprcnn": (
        CKPT_DIR / "court_kpRCNN.pth",
        "1FGKyX-NudJGXvfsmKEpjiQYojDAWONdy",
        None,
    ),
    "yolov8s": (
        Path("yolov8s.pt"),
        None,
        None,  # Ultralytics auto-downloads
    ),
}


def ensure_model(name: str, *, force: bool = False,
                 registry: dict | None = None) -> Path | None:
    """Resolve a model path with local → GDrive → repo fallback.

    Args:
        name: Key into MODEL_REGISTRY.
        force: Re-download even if local file exists.
        registry: Override registry dict (for testing).

    Returns:
        Path to the model file, or None if not found and all downloads failed.
    """
    reg = registry or MODEL_REGISTRY
    entry = reg.get(name)
    if entry is None:
        logger.warning("Unknown model: %s", name)
        return None

    local_path, gdrive_id, alt_url = entry

    # 1. Local file exists → done
    if local_path.exists() and not force:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Try GDrive
    if gdrive_id is not None:
        try:
            import gdown
            logger.info("Downloading %s from Google Drive (id=%s) ...", name, gdrive_id)
            if gdrive_id.endswith(".zip") or _is_zip_gdrive(gdrive_id):
                zip_path = local_path.with_suffix(".zip")
                gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        # Extract ALL .pt and .onnx files from the zip into
                        # local_path.parent (e.g. ckpts/ for TrackNet,
                        # ckpts/rtmpose/ for RTMPose). Many zips bundle
                        # multiple checkpoints in one archive.
                        for member in zf.namelist():
                            if member.endswith('.onnx') or member.endswith('.pt'):
                                dest = local_path.parent / Path(member).name
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(zf.read(member))
                                logger.info("  extracted -> %s", dest)
                    zip_path.unlink(missing_ok=True)
            else:
                gdown.download(id=gdrive_id, output=str(local_path), quiet=False)
            if local_path.exists():
                logger.info("  -> %s", local_path)
                return local_path
        except Exception as e:
            logger.warning("GDrive download failed for %s: %s", name, e)

    # 3. Fall back to alt URL (source repo)
    if alt_url is not None:
        try:
            logger.info("Downloading %s from %s ...", name, alt_url)
            if alt_url.endswith(".zip"):
                zip_path = local_path.with_suffix(".zip")
                urllib.request.urlretrieve(alt_url, str(zip_path))
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for member in zf.namelist():
                        if member.endswith('.onnx') or member.endswith('.pt'):
                            dest = local_path.parent / Path(member).name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(zf.read(member))
                            logger.info("  extracted -> %s", dest)
                zip_path.unlink(missing_ok=True)
            else:
                urllib.request.urlretrieve(alt_url, str(local_path))
            if local_path.exists():
                logger.info("  -> %s", local_path)
                return local_path
        except Exception as e:
            logger.warning("Alt download failed for %s: %s", name, e)

    logger.error("Could not obtain model: %s (tried local, GDrive, alt URL)", name)
    return None


_ZIP_GDRIVE_IDS: set[str] = {
    "1XjwDxz1a8i3WO6afuvaq-y3HPiFh48SN",  # rtmpose_colab (zip of ONNX)
    "1rhKXbff1GITgrFTYptW6gAvWZ76E_qzp",  # tracknet (zip of TrackNet_best.pt + InpaintNet_best.pt)
}


def _is_zip_gdrive(gdrive_id: str) -> bool:
    """Check if a GDrive file ID is known to point to a zip archive."""
    return gdrive_id in _ZIP_GDRIVE_IDS


# ─── Existing utilities ────────────────────────────────────────────────────


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


# ─── Lazy model getters (auto-ensure) ──────────────────────────────────────


def get_yolov8():
    if "yolov8" not in _models:
        try:
            from app.models.yolov8 import YOLOv8Tracker
            ensure_model("yolov8s")
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
            ensure_model("tracknet")
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
            ensure_model("rtmpose")
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
            ensure_model("bst")
            s = _get_settings()
            path = str(s.bst_model_path) if s.bst_model_path else None
            _models["bst"] = BSTClassifier(path, device=_get_device())
        except ImportError:
            logger.warning("BST not available (standalone mode)")
            return None
    return _models.get("bst")


def get_hrnet():
    if "hrnet" not in _models:
        try:
            from app.models.rtmpose import RTMPoseEstimator
            ensure_model("hrnet")
            s = _get_settings()
            path = str(s.hrnet_model_path) if s.hrnet_model_path else None
            _models["hrnet"] = RTMPoseEstimator(path, device=_get_device())
        except ImportError:
            logger.warning("HRNet not available (standalone mode)")
            return None
    return _models.get("hrnet")


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