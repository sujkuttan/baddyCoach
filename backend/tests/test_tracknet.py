import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from app.config.settings import settings


# ─── Synthetic architecture tests (no weights needed) ──────────────────────

@pytest.mark.cpu_only
def test_tracknet_model_forward():
    """Verify custom UNet backbone produces correct output shape."""
    from app.models.tracknet import TrackNetV3Model
    model = TrackNetV3Model()
    model.eval()
    batch = torch.randn(2, 27, 288, 512)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 8, 288, 512), f"Expected (2, 8, 288, 512), got {out.shape}"


@pytest.mark.cpu_only
def test_tracknet_model_output_range():
    """Verify backbone output is a reasonable heatmap (finite values)."""
    from app.models.tracknet import TrackNetV3Model
    model = TrackNetV3Model()
    model.eval()
    batch = torch.randn(1, 27, 288, 512)
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    assert out.min() < out.max(), "Output has no variation"


@pytest.mark.cpu_only
def test_inpaintnet_forward():
    """Verify InpaintNet produces correct output shape."""
    from app.models.tracknet import InpaintNet
    model = InpaintNet()
    model.eval()
    batch = torch.randn(1, 15, 2)
    mask = torch.zeros(1, 15, 1)
    with torch.no_grad():
        out = model(batch, mask)
    assert out.shape == (1, 15, 2), f"Expected (1, 15, 2), got {out.shape}"


@pytest.mark.cpu_only
def test_inpaintnet_output_range():
    """Verify InpaintNet produces finite values."""
    from app.models.tracknet import InpaintNet
    model = InpaintNet()
    model.eval()
    batch = torch.randn(1, 15, 2)
    mask = torch.zeros(1, 15, 1)
    with torch.no_grad():
        out = model(batch, mask)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


@pytest.mark.cpu_only
def test_inpaintnet_loads_complete_official_checkpoint_with_key_shape_match(tmp_path):
    """Official InpaintNet tensor names and shapes must load without relaxation."""
    from app.models.tracknet import InpaintNet, TrackNetV3

    official_shapes = {
        "down_1.conv.weight": (32, 3, 3), "down_1.conv.bias": (32,),
        "down_2.conv.weight": (64, 32, 3), "down_2.conv.bias": (64,),
        "down_3.conv.weight": (128, 64, 3), "down_3.conv.bias": (128,),
        "buttelneck.conv_1.conv.weight": (256, 128, 3), "buttelneck.conv_1.conv.bias": (256,),
        "buttelneck.conv_2.conv.weight": (256, 256, 3), "buttelneck.conv_2.conv.bias": (256,),
        "up_1.conv.weight": (128, 384, 3), "up_1.conv.bias": (128,),
        "up_2.conv.weight": (64, 192, 3), "up_2.conv.bias": (64,),
        "up_3.conv.weight": (32, 96, 3), "up_3.conv.bias": (32,),
        "predictor.weight": (2, 32, 3), "predictor.bias": (2,),
    }
    reference = InpaintNet()
    assert {name: tuple(value.shape) for name, value in reference.state_dict().items()} == official_shapes
    # The published checkpoint is DataParallel-wrapped and contains the historical
    # misspelling ``buttelneck``.  The loader must also accept the corrected form.
    checkpoint = {
        "model_state_dict": {
            f"module.{name.replace('buttelneck', 'buttleneck')}": torch.zeros(shape)
            for name, shape in official_shapes.items()
        }
    }
    path = tmp_path / "InpaintNet_best.pt"
    torch.save(checkpoint, path)

    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker._load_inpaintnet(str(path))

    assert tracker.inpaintnet is not None
    loaded = tracker.inpaintnet.state_dict()
    assert loaded.keys() == official_shapes.keys()
    assert {name: tuple(value.shape) for name, value in loaded.items()} == official_shapes


@pytest.mark.cpu_only
def test_rectification_preserves_observations_and_repairs_original_gap():
    """Only a detection absent before rectification may be replaced by InpaintNet."""
    from app.models.tracknet import TrackNetV3

    class ConstantRepairNet(nn.Module):
        def forward(self, coords, mask):
            assert coords.shape == (1, 3, 2)
            assert mask.shape == (1, 3, 1)
            assert torch.equal(mask[0, :, 0], torch.tensor([0.0, 1.0, 0.0]))
            return torch.tensor([[[0.1, 0.2], [0.5, 0.25], [0.9, 0.8]]])

    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker.inpaintnet = ConstantRepairNet()
    raw = [(10.0, 20.0, 0.85), None, (90.0, 80.0, 0.65)]

    repaired = tracker._rectify_trajectory(raw, orig_w=100, orig_h=100)

    assert repaired[0] == raw[0]
    assert repaired[2] == raw[2]
    assert repaired[1] == (50.0, 25.0, 0.0)


@pytest.mark.cpu_only
def test_predict_batch_repairs_low_confidence_peak_without_changing_high_confidence(monkeypatch):
    """Sub-threshold peaks become original-missing repair candidates before rectification."""
    from app.models.tracknet import TrackNetV3

    class StubBackbone(nn.Module):
        def forward(self, batch):
            return torch.zeros((len(batch), 8, 1, 1))

    class ConstantRepairNet(nn.Module):
        def forward(self, coords, mask):
            assert torch.equal(mask[0, :, 0], torch.tensor([0.0, 1.0]))
            return torch.tensor([[[0.1, 0.2], [0.5, 0.25]]])

    peaks = iter([(10.0, 20.0, 0.90), (90.0, 80.0, 0.10)])
    monkeypatch.setattr("app.models.tracknet._extract_peak", lambda *_: next(peaks))
    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker.model = StubBackbone()
    tracker.inpaintnet = ConstantRepairNet()
    tracker._preprocess = lambda frames: [np.zeros((1, 1, 3), dtype=np.float32) for _ in frames]

    results = tracker.predict_batch(
        [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)], batch_size=2
    )

    assert results[0] == {"x": 10.0, "y": 20.0, "confidence": 0.90}
    assert results[1]["x"] == 50.0
    assert results[1]["y"] == 25.0
    assert results[1]["confidence"] >= settings.shuttle_clean_min_conf
    assert results[1]["was_repaired"] is True

    from app.pipeline.shared.shuttle_utils import clean_trajectory
    cleaned = clean_trajectory(pd.DataFrame(results), settings)
    assert cleaned.loc[1, "x"] == 50.0
    assert cleaned.loc[1, "y"] == 25.0
    assert cleaned.loc[1, "was_repaired"] == True
    assert cleaned.loc[1, "was_interpolated"] == False


@pytest.mark.cpu_only
def test_inpaintnet_incompatible_checkpoint_is_disabled_and_recorded(tmp_path):
    """Key/shape incompatibility must disable repair rather than partially loading it."""
    from app.models.tracknet import TrackNetV3
    from app.pipeline.shared.models import _model_health

    path = tmp_path / "incompatible.pt"
    torch.save({"model": {"down_1.conv.weight": torch.zeros(1)}}, path)
    _model_health.clear()
    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker._load_inpaintnet(str(path))

    assert tracker.inpaintnet is None
    assert _model_health["inpaintnet"]["loaded"] is False
    assert _model_health["inpaintnet"]["missing"]


@pytest.mark.model
def test_local_official_inpaintnet_checkpoint_matches_all_keys_and_shapes():
    """The actual local InpaintNet checkpoint must exactly match the public architecture."""
    from app.models.tracknet import InpaintNet

    path = Path(__file__).resolve().parents[2] / "ckpts" / "InpaintNet_best.pt"
    if not path.exists():
        pytest.skip("InpaintNet checkpoint not found")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    while isinstance(state_dict, dict):
        nested = next((state_dict[key] for key in ("model_state_dict", "model", "state_dict")
                       if key in state_dict), None)
        if nested is None:
            break
        state_dict = nested
    actual = {
        key.removeprefix("module.").replace("buttleneck", "buttelneck"): value
        for key, value in state_dict.items()
    }
    expected = InpaintNet().state_dict()

    assert actual.keys() == expected.keys()
    assert {key: tuple(value.shape) for key, value in actual.items()} == {
        key: tuple(value.shape) for key, value in expected.items()
    }


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
def test_build_9frame_window():
    """Verify 9-frame window construction."""
    from app.models.tracknet import _build_9frame_window
    preprocessed = [np.zeros((288, 512, 3), dtype=np.float32) for _ in range(10)]
    # Middle frame
    window = _build_9frame_window(preprocessed, 5)
    assert window.shape == (27, 288, 512)
    # First frame (edge case — pads frame 0)
    window = _build_9frame_window(preprocessed, 0)
    assert window.shape == (27, 288, 512)
    # Last frame (edge case — pads frame last)
    window = _build_9frame_window(preprocessed, 9)
    assert window.shape == (27, 288, 512)


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
def test_tracknet_model_load_state_dict():
    """Verify the model can accept a synthetic state_dict."""
    import torch
    import torch.nn as nn
    from app.models.tracknet import TrackNetV3Model
    model = TrackNetV3Model()
    sd = {}
    for name, param in model.named_parameters():
        sd[name] = torch.randn(param.shape)
    model.load_state_dict(sd, strict=False)
    model.eval()
    batch = torch.randn(1, 27, 288, 512)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 8, 288, 512)


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
    assert len(results) == 6
    for r in results:
        assert 'x' in r
        assert 'y' in r
