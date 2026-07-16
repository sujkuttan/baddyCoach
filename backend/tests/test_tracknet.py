import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from app.config.settings import settings


# ─── Synthetic architecture tests (no weights needed) ──────────────────────

@pytest.mark.cpu_only
def test_build_input_background_then_eight_frames():
    """The checkpoint input is background RGB followed by eight RGB frames."""
    from app.models.tracknet import _build_input

    background = np.zeros((2, 3, 3), dtype=np.float32)
    frames = [np.full((2, 3, 3), value, dtype=np.float32) for value in range(1, 9)]

    model_input = _build_input(frames, background)

    assert model_input.shape == (27, 2, 3)
    assert np.array_equal(model_input[:3], np.zeros((3, 2, 3), dtype=np.float32))
    for index, value in enumerate(range(1, 9)):
        assert np.array_equal(model_input[3 + index * 3:6 + index * 3],
                              np.full((3, 2, 3), value, dtype=np.float32))


@pytest.mark.cpu_only
def test_eight_frame_window_edge_pads_before_and_after_video():
    """Windows at both ends retain eight slots by repeating the edge frame."""
    from app.models.tracknet import _build_8frame_window

    frames = [np.full((1, 1, 3), value, dtype=np.float32) for value in range(2)]

    assert [frame[0, 0, 0] for frame in _build_8frame_window(frames, -7)] == [0.0] * 8
    assert [frame[0, 0, 0] for frame in _build_8frame_window(frames, 1)] == [1.0] * 8


@pytest.mark.cpu_only
def test_extract_largest_component_ignores_higher_isolated_pixel():
    """Decoding favors the largest valid blob, not the highest single pixel."""
    from app.models.tracknet import _extract_largest_component

    probabilities = np.zeros((8, 8), dtype=np.float32)
    probabilities[0, 7] = 0.99
    probabilities[3:6, 2:5] = 0.80

    x, y, confidence = _extract_largest_component(
        probabilities, orig_w=80, orig_h=80, threshold=0.50
    )

    assert x == pytest.approx(3 * 80 / 512)
    assert y == pytest.approx(4 * 80 / 288)
    assert confidence == pytest.approx(0.80)


@pytest.mark.cpu_only
def test_extract_component_candidates_returns_multiple_sorted_blobs():
    from app.models.tracknet import _extract_component_candidates

    probabilities = np.zeros((8, 8), dtype=np.float32)
    probabilities[0:3, 0:3] = 0.72  # area 9
    probabilities[5:7, 5:8] = 0.91  # area 6

    candidates = _extract_component_candidates(
        probabilities, orig_w=80, orig_h=80, threshold=0.50, max_components=3
    )

    assert len(candidates) == 2
    assert candidates[0][3] == 9
    assert candidates[1][3] == 6
    assert candidates[0][2] == pytest.approx(0.72)
    assert candidates[1][2] == pytest.approx(0.91)


@pytest.mark.cpu_only
def test_select_detection_candidate_prefers_motion_consistent_blob_over_larger_area():
    from app.models.tracknet import _select_detection_candidate

    candidates = [
        (75.0, 75.0, 0.90, 12),  # larger / stronger distractor
        (22.0, 20.0, 0.70, 4),   # smaller but on the expected path
    ]

    selected = _select_detection_candidate(
        candidates,
        prev_accepted=(14.0, 20.0, 0.85),
        prev_prev_accepted=(6.0, 20.0, 0.80),
        motion_weight=0.75,
        confidence_weight=0.25,
        distance_scale_px=25.0,
    )

    assert selected == candidates[1]


@pytest.mark.cpu_only
def test_triangular_overlap_aggregation_weights_central_predictions_more():
    """Overlapping eight-channel outputs are combined per frame with triangular weights."""
    from app.models.tracknet import _aggregate_overlapping_heatmaps

    first = np.zeros((8, 1, 1), dtype=np.float32)
    second = np.zeros((8, 1, 1), dtype=np.float32)
    first[4, 0, 0] = 1.0
    second[1, 0, 0] = 0.0

    aggregate = _aggregate_overlapping_heatmaps(
        [(0, first), (3, second)], n_frames=11, sequence_length=8
    )

    # Frame four receives offset 4 (weight 4) and offset 1 (weight 2),
    # proving overlapping central predictions receive the larger influence.
    assert aggregate.shape == (11, 1, 1)
    assert aggregate[4, 0, 0] == pytest.approx(4 / 6)


@pytest.mark.cpu_only
def test_court_crop_rect_expands_bbox_and_preserves_tracknet_aspect():
    from app.models.tracknet import _court_crop_rect

    corners = [(100, 620), (1180, 620), (300, 180), (980, 180)]
    crop = _court_crop_rect(
        corners,
        margins={"left": 0.10, "right": 0.05, "top": 0.20, "bottom": 0.10},
        aspect=512.0 / 288.0,
    )

    x0, y0, x1, y1 = crop
    assert x0 < 100.0
    assert x1 > 1180.0
    assert y0 < 180.0
    assert y1 > 620.0
    assert ((x1 - x0) / (y1 - y0)) == pytest.approx(512.0 / 288.0)


@pytest.mark.cpu_only
def test_map_detection_to_full_frame_undoes_crop_resize():
    from app.models.tracknet import _map_detection_to_full_frame

    crop_rect = (100.0, 50.0, 900.0, 500.0)
    full_x, full_y = _map_detection_to_full_frame(256.0, 144.0, crop_rect)

    assert full_x == pytest.approx(500.0)
    assert full_y == pytest.approx(275.0)


@pytest.mark.cpu_only
def test_gate_tracknet_spikes_removes_out_and_back_teleport():
    from app.models.tracknet import _gate_tracknet_spikes

    points = np.array([
        [10.0, 10.0],
        [20.0, 20.0],
        [300.0, 300.0],
        [30.0, 30.0],
        [40.0, 40.0],
    ], dtype=np.float64)

    gated, removed = _gate_tracknet_spikes(points, max_step_px=100.0)

    assert removed == 1
    assert np.isnan(gated[2]).all()
    assert np.allclose(gated[[0, 1, 3, 4]], points[[0, 1, 3, 4]])


@pytest.mark.cpu_only
def test_gate_tracknet_spikes_keeps_fast_monotonic_motion_and_edge_singletons():
    from app.models.tracknet import _gate_tracknet_spikes

    monotonic = np.array([
        [0.0, 0.0],
        [90.0, 0.0],
        [180.0, 0.0],
    ], dtype=np.float64)
    gated_monotonic, removed_monotonic = _gate_tracknet_spikes(monotonic, max_step_px=100.0)
    assert removed_monotonic == 0
    assert np.allclose(gated_monotonic, monotonic)

    edge_singletons = np.array([
        [300.0, 300.0],
        [20.0, 20.0],
        [30.0, 30.0],
        [400.0, 400.0],
    ], dtype=np.float64)
    gated_edges, removed_edges = _gate_tracknet_spikes(edge_singletons, max_step_px=100.0)
    assert removed_edges == 0
    assert np.allclose(gated_edges, edge_singletons)


@pytest.mark.cpu_only
def test_merge_far_tile_tracks_only_fills_missing_far_half_frames():
    from app.models.tracknet import _merge_far_tile_tracks

    primary = np.array([
        [100.0, 100.0],
        [np.nan, np.nan],
        [500.0, 500.0],
        [np.nan, np.nan],
    ], dtype=np.float64)
    far = np.array([
        [150.0, 120.0],
        [200.0, 140.0],
        [550.0, 160.0],
        [250.0, 420.0],
    ], dtype=np.float64)

    merged, filled = _merge_far_tile_tracks(primary, far, net_y=300.0)

    assert filled == 1
    assert np.allclose(merged[0], primary[0])
    assert np.allclose(merged[1], far[1])
    assert np.allclose(merged[2], primary[2])
    assert np.isnan(merged[3]).all()

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
def test_tracknet_decoder_uses_nearest_upsampling_like_reference():
    """The reference TrackNetV3 (qaz812345/TrackNetV3) upsamples with
    nn.Upsample(scale_factor=2), whose default mode is nearest-neighbor. The
    checkpoint was trained that way, so the backbone decoder must upsample with
    nearest — not bilinear — to avoid a train/inference feature mismatch.

    We assert equivalence against a from-scratch nearest reference forward pass
    (same weights) and inequivalence against a bilinear variant.
    """
    from app.models.tracknet import TrackNetV3Model

    model = TrackNetV3Model().eval()

    def _forward(up_mode: str, align_corners) -> torch.Tensor:
        x = model  # local alias
        d1 = x.down_block_1['conv_2'](x.down_block_1['conv_1'](batch))
        d1_pool = nn.functional.max_pool2d(d1, 2)
        d2 = x.down_block_2['conv_2'](x.down_block_2['conv_1'](d1_pool))
        d2_pool = nn.functional.max_pool2d(d2, 2)
        d3 = x.down_block_3['conv_3'](
            x.down_block_3['conv_2'](x.down_block_3['conv_1'](d2_pool)))
        d3_pool = nn.functional.max_pool2d(d3, 2)
        b = x.bottleneck['conv_3'](
            x.bottleneck['conv_2'](x.bottleneck['conv_1'](d3_pool)))

        def up(t, ref):
            kwargs = {"size": ref.shape[2:], "mode": up_mode}
            if align_corners is not None:
                kwargs["align_corners"] = align_corners
            return nn.functional.interpolate(t, **kwargs)

        u1 = x.up_block_1['conv_3'](x.up_block_1['conv_2'](
            x.up_block_1['conv_1'](torch.cat([up(b, d3), d3], dim=1))))
        u2 = x.up_block_2['conv_2'](
            x.up_block_2['conv_1'](torch.cat([up(u1, d2), d2], dim=1)))
        u3 = x.up_block_3['conv_2'](
            x.up_block_3['conv_1'](torch.cat([up(u2, d1), d1], dim=1)))
        return x.predictor(u3)

    torch.manual_seed(0)
    batch = torch.randn(1, 27, 288, 512)
    with torch.no_grad():
        actual = model(batch)
        nearest = _forward("nearest", None)
        bilinear = _forward("bilinear", True)

    assert torch.allclose(actual, nearest, atol=1e-6), \
        "Backbone decoder must upsample with nearest-neighbor to match the checkpoint"
    assert not torch.allclose(actual, bilinear, atol=1e-4), \
        "Backbone output unexpectedly matches bilinear upsampling"


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

    candidates = iter([[(10.0, 20.0, 0.90, 5)], [(90.0, 80.0, 0.10, 5)]])
    monkeypatch.setattr("app.models.tracknet._extract_component_candidates", lambda *_args, **_kwargs: next(candidates))
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
def test_predict_batch_drops_low_confidence_large_jump_before_repair(monkeypatch):
    """A weak, discontinuous candidate should become a repair target, not a real point."""
    from app.models.tracknet import TrackNetV3

    class StubBackbone(nn.Module):
        def forward(self, batch):
            return torch.zeros((len(batch), 8, 1, 1))

    class ConstantRepairNet(nn.Module):
        def forward(self, coords, mask):
            assert torch.equal(mask[0, :, 0], torch.tensor([0.0, 1.0, 0.0]))
            return torch.tensor([[[0.1, 0.2], [0.5, 0.25], [0.9, 0.8]]])

    candidates = iter([[(10.0, 20.0, 0.90, 5)], [(90.0, 80.0, 0.35, 5)], [(95.0, 82.0, 0.90, 5)]])
    monkeypatch.setattr("app.models.tracknet._extract_component_candidates", lambda *_args, **_kwargs: next(candidates))
    monkeypatch.setattr(settings, "tracknet_detection_min_conf", 0.45, raising=False)
    monkeypatch.setattr(settings, "tracknet_low_conf_max_jump_px", 25.0, raising=False)
    monkeypatch.setattr(settings, "tracknet_component_distance_scale_px", 25.0, raising=False)
    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker.model = StubBackbone()
    tracker.inpaintnet = ConstantRepairNet()
    tracker._preprocess = lambda frames: [np.zeros((1, 1, 3), dtype=np.float32) for _ in frames]

    results = tracker.predict_batch(
        [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)], batch_size=3
    )

    assert results[0] == {"x": 10.0, "y": 20.0, "confidence": 0.90}
    assert results[1]["x"] == 50.0
    assert results[1]["y"] == 25.0
    assert results[1]["was_repaired"] is True
    assert results[2] == {"x": 95.0, "y": 82.0, "confidence": 0.90}


@pytest.mark.cpu_only
def test_predict_batch_keeps_low_confidence_continuous_peak(monkeypatch):
    """A weak but temporally consistent candidate should stay observed."""
    from app.models.tracknet import TrackNetV3

    class StubBackbone(nn.Module):
        def forward(self, batch):
            return torch.zeros((len(batch), 8, 1, 1))

    candidates = iter([[(10.0, 20.0, 0.90, 5)], [(18.0, 24.0, 0.35, 5)]])
    monkeypatch.setattr("app.models.tracknet._extract_component_candidates", lambda *_args, **_kwargs: next(candidates))
    monkeypatch.setattr(settings, "tracknet_detection_min_conf", 0.45, raising=False)
    monkeypatch.setattr(settings, "tracknet_low_conf_max_jump_px", 25.0, raising=False)
    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker.model = StubBackbone()
    tracker.inpaintnet = None
    tracker._preprocess = lambda frames: [np.zeros((1, 1, 3), dtype=np.float32) for _ in frames]

    results = tracker.predict_batch(
        [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)], batch_size=2
    )

    assert results[0] == {"x": 10.0, "y": 20.0, "confidence": 0.90}
    assert results[1] == {"x": 18.0, "y": 24.0, "confidence": 0.35}


@pytest.mark.cpu_only
def test_predict_batch_keeps_low_confidence_motion_consistent_fast_point(monkeypatch):
    """A weak point on the predicted fast-flight path should stay observed."""
    from app.models.tracknet import TrackNetV3

    class StubBackbone(nn.Module):
        def forward(self, batch):
            return torch.zeros((len(batch), 8, 1, 1))

    candidates = iter([
        [(10.0, 20.0, 0.90, 5)],
        [(40.0, 20.0, 0.90, 5)],
        [(70.0, 20.0, 0.35, 5)],
    ])
    monkeypatch.setattr("app.models.tracknet._extract_component_candidates", lambda *_args, **_kwargs: next(candidates))
    monkeypatch.setattr(settings, "tracknet_detection_min_conf", 0.45, raising=False)
    monkeypatch.setattr(settings, "tracknet_low_conf_max_jump_px", 25.0, raising=False)
    monkeypatch.setattr(settings, "tracknet_component_distance_scale_px", 25.0, raising=False)
    tracker = TrackNetV3(model_path=None, device="cpu")
    tracker.model = StubBackbone()
    tracker.inpaintnet = None
    tracker._preprocess = lambda frames: [np.zeros((1, 1, 3), dtype=np.float32) for _ in frames]

    results = tracker.predict_batch(
        [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)], batch_size=3
    )

    assert results[0] == {"x": 10.0, "y": 20.0, "confidence": 0.90}
    assert results[1] == {"x": 40.0, "y": 20.0, "confidence": 0.90}
    assert results[2] == {"x": 70.0, "y": 20.0, "confidence": 0.35}


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
