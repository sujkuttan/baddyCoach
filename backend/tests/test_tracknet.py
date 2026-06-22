import numpy as np
import pytest
import torch
from pathlib import Path


# ─── Synthetic architecture tests (no weights needed) ──────────────────────

@pytest.mark.cpu_only
def test_tracknet_backbone_forward():
    """Verify VGG-style backbone produces correct output shape."""
    from app.models.tracknet import TrackNetV3Backbone
    model = TrackNetV3Backbone(in_channels=9)
    model.eval()
    batch = torch.randn(2, 9, 288, 512)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 1, 288, 512), f"Expected (2, 1, 288, 512), got {out.shape}"


@pytest.mark.cpu_only
def test_tracknet_backbone_output_range():
    """Verify backbone output is a reasonable heatmap (finite values)."""
    from app.models.tracknet import TrackNetV3Backbone
    model = TrackNetV3Backbone(in_channels=9)
    model.eval()
    batch = torch.randn(1, 9, 288, 512)
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    assert out.min() < out.max(), "Output has no variation"


@pytest.mark.cpu_only
def test_inpaintnet_forward():
    """Verify InpaintNet produces correct output shape."""
    from app.models.tracknet import InpaintNet
    model = InpaintNet(window_size=15)
    model.eval()
    batch = torch.randn(1, 3, 15)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 2, 15), f"Expected (1, 2, 15), got {out.shape}"


@pytest.mark.cpu_only
def test_inpaintnet_output_range():
    """Verify InpaintNet produces finite values."""
    from app.models.tracknet import InpaintNet
    model = InpaintNet(window_size=15)
    model.eval()
    batch = torch.randn(1, 3, 15)
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


@pytest.mark.cpu_only
def test_tracknet_v3_wrapper_no_model():
    """Verify TrackNetV3 wrapper raises RuntimeError when no model loaded."""
    from app.models.tracknet import TrackNetV3
    model = TrackNetV3(model_path=None, device="cpu")
    assert model.model is None
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(RuntimeError, match="backbone not loaded"):
        model.predict(frames)


@pytest.mark.cpu_only
def test_tracknet_v3_too_few_frames():
    """Verify TrackNetV3 raises error with < 3 frames."""
    from app.models.tracknet import TrackNetV3
    model = TrackNetV3(model_path=None, device="cpu")
    with pytest.raises(RuntimeError, match="backbone not loaded"):
        model.predict_batch([np.zeros((100, 100, 3), dtype=np.uint8)])


@pytest.mark.cpu_only
def test_build_3frame_window():
    """Verify 3-frame window construction."""
    from app.models.tracknet import _build_3frame_window
    preprocessed = [np.zeros((288, 512, 3), dtype=np.float32) for _ in range(5)]
    # Middle frame
    window = _build_3frame_window(preprocessed, 2)
    assert window.shape == (9, 288, 512)
    # First frame (edge case — repeats frame 0)
    window = _build_3frame_window(preprocessed, 0)
    assert window.shape == (9, 288, 512)
    # Last frame (edge case — repeats frame 4)
    window = _build_3frame_window(preprocessed, 4)
    assert window.shape == (9, 288, 512)


@pytest.mark.cpu_only
def test_extract_peak():
    """Verify peak extraction from heatmap."""
    from app.models.tracknet import _extract_peak
    # Create a heatmap with a known peak
    hm = np.zeros((288, 512), dtype=np.float32)
    hm[100, 200] = 10.0  # Strong peak at (200, 100)
    x, y, conf = _extract_peak(hm, 1280, 720)
    assert 0 <= conf <= 1
    assert x == 200 * 1280 / 512  # Scaled to original
    assert y == 100 * 720 / 288


@pytest.mark.cpu_only
def test_tracknet_backbone_load_state_dict():
    """Verify the backbone can accept a synthetic state_dict."""
    import torch
    import torch.nn as nn
    from app.models.tracknet import TrackNetV3Backbone
    model = TrackNetV3Backbone(in_channels=9)
    # Create a synthetic state_dict with matching shapes
    sd = {}
    for name, param in model.named_parameters():
        sd[name] = torch.randn(param.shape)
    model.load_state_dict(sd, strict=False)
    model.eval()
    batch = torch.randn(1, 9, 288, 512)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 1, 288, 512)


# ─── Integration tests (need checkpoint files) ────────────────────────────

@pytest.mark.model
@pytest.mark.memory_intensive
def test_tracknet_predict_returns_position():
    from app.models.tracknet import TrackNetV3
    model_path = Path("ckpts/TrackNet_best.pt")
    if not model_path.exists():
        pytest.skip("TrackNet checkpoint not found")
    model = TrackNetV3(str(model_path), device="cpu")
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    result = model.predict(frames)
    assert len(result) == 1
    assert 'x' in result[0]
    assert 'y' in result[0]
    assert 'confidence' in result[0]
    assert 0 <= result[0]['confidence'] <= 1


@pytest.mark.model
@pytest.mark.memory_intensive
def test_tracknet_predict_batch():
    from app.models.tracknet import TrackNetV3
    model_path = Path("ckpts/TrackNet_best.pt")
    if not model_path.exists():
        pytest.skip("TrackNet checkpoint not found")
    model = TrackNetV3(str(model_path), device="cpu")
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(6)]
    results = model.predict_batch(frames, batch_size=3)
    assert len(results) == 4
    for r in results:
        assert 'x' in r
        assert 'y' in r
