import importlib

import pytest

from app.config import gpu_batch
from app.config.settings import settings


@pytest.fixture
def _reset_settings():
    original = {
        "bst_batch_size": settings.bst_batch_size,
        "yolo_batch_size": settings.yolo_batch_size,
        "tracknet_batch_size": settings.tracknet_batch_size,
        "rtmpose_batch_size": settings.rtmpose_batch_size,
    }
    yield
    for k, v in original.items():
        setattr(settings, k, v)
    importlib.reload(gpu_batch)


def test_override_applied_on_cpu_branch(_reset_settings):
    """Overrides must apply even when no CUDA GPU is available (the path that
    previously early-returned before applying overrides)."""
    settings.bst_batch_size = 256
    settings.yolo_batch_size = 32
    cfg = gpu_batch.get_gpu_batch_config("cuda")  # no GPU here -> base CPU tier
    assert cfg["bst_batch"] == 256
    assert cfg["yolo_batch"] == 32


def test_partial_override_keeps_auto_tier(_reset_settings):
    """Only non-None overrides should replace the auto-detected tier value."""
    settings.bst_batch_size = 512
    cfg = gpu_batch.get_gpu_batch_config("cpu")
    assert cfg["bst_batch"] == 512          # overridden
    assert cfg["yolo_batch"] == 8           # untouched tier default
    assert cfg["tracknet_chunk"] == 8       # untouched tier default


def test_no_override_returns_tier(_reset_settings):
    settings.bst_batch_size = None
    settings.yolo_batch_size = None
    settings.tracknet_batch_size = None
    settings.rtmpose_batch_size = None
    cfg = gpu_batch.get_gpu_batch_config("cpu")
    assert cfg["bst_batch"] == 16  # CPU tier default, no override


def test_bst_default_batch_size_honors_setting(_reset_settings):
    from app.models.bst import BSTClassifier

    settings.bst_batch_size = 400
    try:
        assert BSTClassifier._default_batch_size() == 400
    finally:
        settings.bst_batch_size = None
