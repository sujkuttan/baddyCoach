import numpy as np
import pandas as pd
import pytest
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import StrokeClassificationStage
from app.pipeline.strokes import _temporal_resample


def _varied_keypoints() -> np.ndarray:
    """A spatially-varied, fully-confident COCO-17 skeleton.

    Every joint sits at a distinct location so that joint normalization does
    not collapse the skeleton to zeros (which would trip the ``degenerate_joints``
    quality gate). Used by tests that need a clip to pass the input-quality
    gate on its own merits.
    """
    xs = 40.0 + np.arange(17, dtype=float) * 2.0
    ys = 30.0 + (np.arange(17, dtype=float) % 5) * 6.0
    return np.column_stack([xs, ys, np.ones(17)])


def test_stroke_classification_labels_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    hits_df = pd.DataFrame({
        "frame": [0, 10, 20, 30],
        "confidence": [0.9, 0.85, 0.92, 0.88],
    })
    store.set_parquet("hits", hits_df)

    shuttle_df = pd.DataFrame({
        "frame": list(range(40)),
        "x": np.linspace(100, 500, 40),
        "y": np.linspace(200, 100, 40),
        "confidence": [0.95] * 40,
    })
    store.set_parquet("shuttle", shuttle_df)

    pose_df = pd.DataFrame({
        "frame": list(range(40)),
        "player_id": ["player_1"] * 40,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(40)],
    })
    store.set_parquet("pose", pose_df)

    court_data = {"court_length": 13.4, "court_width": 6.10}
    store.set("court", court_data)

    stage = StrokeClassificationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert len(shots_df) == 4
    assert "stroke_type" in shots_df.columns
    assert "stroke_confidence" in shots_df.columns


def test_temporal_resample_upsample():
    """Upsample from 10 to 20 frames via linear interpolation."""
    arr = np.arange(10, dtype=np.float32).reshape(10, 1)
    result = _temporal_resample(arr, 20)
    assert result.shape == (20, 1)
    # First and last values preserved (np.interp clamps to boundaries)
    assert result[0, 0] == 0.0
    assert result[-1, 0] == 9.0
    # result[k] = np.interp(k * 9/19, [0..9], [0..9])
    assert abs(result[10, 0] - (10 * 9 / 19)) < 0.01


def test_temporal_dedup_preserves_quality_abstained_shots():
    from app.pipeline.strokes import _can_temporally_deduplicate

    assert _can_temporally_deduplicate(
        {"stroke_type": "unknown", "bst_input_eligible": False},
        {"stroke_type": "smash", "bst_input_eligible": True},
        gap=1,
        max_gap=6,
    ) is False


def test_temporal_resample_downsample():
    """Downsample from 100 to 50 frames."""
    arr = np.arange(100, dtype=np.float32).reshape(100, 1)
    result = _temporal_resample(arr, 50)
    assert result.shape == (50, 1)
    assert result[0, 0] == 0.0
    assert result[-1, 0] == 99.0
    # result[1] = np.interp(1 * 99/49, [0..99], [0..99])
    expected = 1 * 99 / 49
    assert abs(result[1, 0] - expected) < 0.01


def test_temporal_resample_identity():
    """Same length returns the same array."""
    arr = np.random.rand(50, 2, 72).astype(np.float32)
    result = _temporal_resample(arr, 50)
    np.testing.assert_array_equal(result, arr)


def test_temporal_resample_empty():
    """Empty input returns empty output."""
    arr = np.zeros((0, 2, 72), dtype=np.float32)
    result = _temporal_resample(arr, 50)
    assert result.shape == (50, 2, 72)


def test_temporal_resample_zero_is_missing():
    """zero_is_missing=True interpolates only between valid regions."""
    arr = np.zeros((10, 2), dtype=np.float32)
    arr[2, :] = [0.5, 0.3]  # valid at index 2
    arr[7, :] = [0.8, 0.4]  # valid at index 7
    result = _temporal_resample(arr, 20, zero_is_missing=True)
    assert result.shape == (20, 2)
    # Regions before the first valid source point are zero-masked
    assert np.all(result[:3, :] == 0)
    # Regions between valid points are interpolated
    mid = len(result) // 2
    assert np.any(result[mid] != 0)
    # Regions after the last valid source point are zero-masked
    assert np.all(result[-3:, :] == 0)


def test_temporal_resample_multi_dim():
    """Multi-dimensional trailing dims are handled."""
    arr = np.random.rand(30, 2, 72).astype(np.float32)
    result = _temporal_resample(arr, 100)
    assert result.shape == (100, 2, 72)


def test_stroke_classification_empty_hits(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    hits_df = pd.DataFrame({"frame": [], "confidence": []})
    store.set_parquet("hits", hits_df)

    stage = StrokeClassificationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert result.metadata["shot_count"] == 0


def test_build_clip_resolution_mode_keeps_court_rejected_pixel_shuttle(monkeypatch):
    from app.pipeline.strokes import _build_clip, settings

    monkeypatch.setattr(settings, "bst_shuttle_norm", "resolution")
    monkeypatch.setattr(settings, "bst_shuttle_require_raw_observation", False)
    frames = [0, 1, 2]
    shuttle = pd.DataFrame({
        "frame": frames,
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 300.0],
        "confidence": [0.9, 0.9, 0.9],
        "was_interpolated": [False, True, False],
        "court_rejected": [False, True, False],
    })
    shuttle_raw = pd.DataFrame({
        "frame": frames,
        "x": [100.0, np.nan, 300.0],
        "y": [100.0, np.nan, 300.0],
        "confidence": [0.9, 0.0, 0.9],
        "was_repaired": [False, True, False],
    })
    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    pose = pd.DataFrame([
        {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}
        for frame in frames for player in ("player_1", "player_2")
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames
        ]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames
        ]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,
        player_detections=players, player_ids=["player_1", "player_2"],
        shuttle_raw=shuttle_raw,
    )

    # Court-rejected frame still contributes resolution-normalized pixels
    np.testing.assert_allclose(clip["shuttle"][1], [200.0 / 640.0, 200.0 / 480.0], atol=1e-6)
    assert clip["_bst_provenance"]["shuttle_court_rejected"] == [False, True, False]
    assert clip["_bst_provenance"]["shuttle_observed"] == [True, False, True]


def test_build_clip_court_mode_zeros_court_rejected_shuttle(monkeypatch):
    from app.pipeline.strokes import _build_clip, settings

    monkeypatch.setattr(settings, "bst_shuttle_norm", "court")
    frames = [0, 1, 2]
    shuttle = pd.DataFrame({
        "frame": frames,
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 300.0],
        "confidence": [0.9, 0.9, 0.9],
        "was_interpolated": [False, False, False],
        "court_rejected": [False, True, False],
    })
    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    pose = pd.DataFrame([
        {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}
        for frame in frames for player in ("player_1", "player_2")
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames
        ]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames
        ]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,
        player_detections=players, player_ids=["player_1", "player_2"],
        homography=np.eye(3),
    )

    np.testing.assert_array_equal(clip["shuttle"][1], [0.0, 0.0])
    assert clip["_bst_provenance"]["shuttle_court_rejected"][1] is True


def test_build_clip_skips_repaired_and_interpolated_when_require_raw(monkeypatch):
    from app.pipeline.strokes import _build_clip, settings

    monkeypatch.setattr(settings, "bst_shuttle_norm", "resolution")
    monkeypatch.setattr(settings, "bst_shuttle_require_raw_observation", True)
    frames = [0, 1, 2]
    # Frame 1: cleaned has xy + interpolated, raw repaired -> tensor stays 0
    shuttle = pd.DataFrame({
        "frame": frames,
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 300.0],
        "confidence": [0.9, 0.9, 0.9],
        "was_interpolated": [False, True, False],
        "court_rejected": [False, False, False],
    })
    shuttle_raw = pd.DataFrame({
        "frame": frames,
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 300.0],
        "confidence": [0.9, 0.0, 0.9],
        "was_repaired": [False, True, False],
    })
    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    pose = pd.DataFrame([
        {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}
        for frame in frames for player in ("player_1", "player_2")
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames
        ]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames
        ]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,
        player_detections=players, player_ids=["player_1", "player_2"],
        shuttle_raw=shuttle_raw,
    )

    # Frame 1: repaired (InpaintNet) is fed to the BST tensor by default
    np.testing.assert_allclose(clip["shuttle"][1], [200.0 / 640.0, 200.0 / 480.0], atol=1e-6)
    assert clip["_bst_provenance"]["shuttle_repaired"][1] is True
    assert clip["_bst_provenance"]["shuttle_interpolated"][1] is True
    # Frames 0 and 2: raw observed -> tensor written
    np.testing.assert_allclose(clip["shuttle"][0], [100.0 / 640.0, 100.0 / 480.0], atol=1e-6)
    np.testing.assert_allclose(clip["shuttle"][2], [300.0 / 640.0, 300.0 / 480.0], atol=1e-6)


def test_build_clip_skips_repaired_when_use_repaired_false(monkeypatch):
    from app.pipeline.strokes import _build_clip, settings

    monkeypatch.setattr(settings, "bst_shuttle_norm", "resolution")
    monkeypatch.setattr(settings, "bst_shuttle_require_raw_observation", True)
    monkeypatch.setattr(settings, "bst_shuttle_use_repaired", False)
    frames = [0, 1, 2]
    shuttle = pd.DataFrame({
        "frame": frames,
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 300.0],
        "confidence": [0.9, 0.9, 0.9],
        "was_interpolated": [False, True, False],
        "court_rejected": [False, False, False],
    })
    shuttle_raw = pd.DataFrame({
        "frame": frames,
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 300.0],
        "confidence": [0.9, 0.0, 0.9],
        "was_repaired": [False, True, False],
    })
    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    pose = pd.DataFrame([
        {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}
        for frame in frames for player in ("player_1", "player_2")
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames
        ]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames
        ]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,
        player_detections=players, player_ids=["player_1", "player_2"],
        shuttle_raw=shuttle_raw,
    )

    # Frame 1: repaired but use_repaired=False -> no tensor written
    np.testing.assert_array_equal(clip["shuttle"][1], [0.0, 0.0])
    assert clip["_bst_provenance"]["shuttle_repaired"][1] is True
    # Frames 0 and 2: raw observed -> tensor written
    np.testing.assert_allclose(clip["shuttle"][0], [100.0 / 640.0, 100.0 / 480.0], atol=1e-6)
    np.testing.assert_allclose(clip["shuttle"][2], [300.0 / 640.0, 300.0 / 480.0], atol=1e-6)


class _QualityGateClassifier:
    seq_len = 20

    def __init__(self):
        self.received = []

    def predict_from_clips(self, clips, **kwargs):
        self.received = clips
        results = [("smash", 0.9, 3, 0.5, 0.0, 0.0) for _ in clips]
        probs = np.zeros((len(clips), 25), dtype=np.float32)
        if len(clips):
            probs[:, 3] = 1.0
        return results, probs


class _AimAlphaProbeClassifier:
    seq_len = 20

    def __init__(self):
        self.received = []

    def predict_from_clips(self, clips, **kwargs):
        self.received.extend(clips)
        results = []
        probs = np.zeros((len(clips), 25), dtype=np.float32)
        for clip in clips:
            offset = clip.get("_alpha_probe_offset", 0)
            alpha = 0.8 if offset >= 0 else 0.2
            results.append(("smash", 0.9, 3, alpha, alpha, 1.0 - alpha))
            probs[len(results) - 1, 3] = 1.0
        return results, probs


def test_stroke_stage_skips_ineligible_clip_and_persists_quality(monkeypatch, tmp_job_dir):
    from app.pipeline.shared import models

    classifier = _QualityGateClassifier()
    monkeypatch.setattr(models, "get_bst", lambda: classifier)
    monkeypatch.setattr("app.pipeline.strokes.settings.fusion_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.hierarchical_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.confusion_pair_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.physics_gate_enabled", False)
    # Court-space shuttle normalization so the court-rejected hard gate applies.
    monkeypatch.setattr("app.pipeline.strokes.settings.bst_shuttle_norm", "court")

    store = ArtifactStore(tmp_job_dir)
    # Hits at 0 and 30. Under the default midpoint clip boundary these span
    # [0:15] (clean) and [15:40] (covers the court-rejected 30-40 window).
    store.set_parquet("hits", pd.DataFrame({"frame": [0, 30], "confidence": [0.9, 0.9]}))
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": list(range(50)), "x": [100.0] * 50, "y": [100.0] * 50,
        "confidence": [0.9] * 50, "was_interpolated": [False] * 50,
        "court_rejected": [False] * 30 + [True] * 10 + [False] * 10,
    }))
    store.set_parquet("shuttle_raw", pd.DataFrame({
        "frame": list(range(50)), "x": [100.0] * 50, "y": [100.0] * 50,
        "confidence": [0.9] * 50, "was_repaired": [False] * 50,
    }))
    keypoints = _varied_keypoints()
    store.set_parquet("pose", pd.DataFrame([
        {"frame": f, "player_id": p, "keypoints": keypoints.tolist()}
        for f in range(50) for p in ("player_1", "player_2")
    ]))
    store.set("court", {
        "court_length": 13.4, "court_width": 6.1,
        "valid": True, "homography": np.eye(3).tolist(),
    })
    store.set("players", {"players": [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": f, "bbox": [0, 0, 100, 100]} for f in range(50)
        ]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": f, "bbox": [200, 0, 300, 100]} for f in range(50)
        ]},
    ]})

    result = StrokeClassificationStage().run(store, StageConfig(debug_level=1))
    shots = store.get_parquet("shots").sort_values("frame").reset_index(drop=True)

    assert result.status == "success"
    assert len(classifier.received) == 1
    assert shots.loc[0, "bst_input_route"] == "bst"
    assert shots.loc[1, "bst_input_route"] == "quality_abstain"
    assert shots.loc[1, "stroke_type"] == "unknown"
    assert bool(shots.loc[1, "is_bst_fallback"]) is True
    assert "court_rejected_shuttle" in shots.loc[1, "bst_input_quality_reasons"]
    assert store.get_parquet("debug_bst_input_quality") is not None


def test_stroke_stage_soft_quality_tier_keeps_prediction_discounted(monkeypatch, tmp_job_dir):
    """Spec 3: a clip on the soft quality tier (score between min and soft) is
    still admitted for a BST prediction but tagged low_quality_bst and its
    confidence is down-weighted rather than dumped to a hard 'unknown'."""
    from app.pipeline.shared import models

    classifier = _QualityGateClassifier()
    monkeypatch.setattr(models, "get_bst", lambda: classifier)
    monkeypatch.setattr("app.pipeline.strokes.settings.fusion_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.hierarchical_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.confusion_pair_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.physics_gate_enabled", False)

    import app.pipeline.strokes as strokes_mod
    from app.pipeline.shared.bst_input_quality import evaluate_bst_clip_quality

    _calls = {"n": 0}

    def _fake_quality(provenance):
        base = evaluate_bst_clip_quality(provenance)
        # Force the second clip (frame-30 hit) into the soft tier.
        if _calls["n"] == 1:
            base = dict(base)
            base["eligible"] = True
            base["soft"] = True
            base["score"] = 0.62
        _calls["n"] += 1
        return base

    monkeypatch.setattr(strokes_mod, "evaluate_bst_clip_quality", _fake_quality)

    store = ArtifactStore(tmp_job_dir)
    store.set_parquet("hits", pd.DataFrame({"frame": [0, 30], "confidence": [0.9, 0.9]}))
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": list(range(50)), "x": [100.0] * 50, "y": [100.0] * 50,
        "confidence": [0.9] * 50, "was_interpolated": [False] * 50,
        "court_rejected": [False] * 50,
    }))
    store.set_parquet("shuttle_raw", pd.DataFrame({
        "frame": list(range(50)), "x": [100.0] * 50, "y": [100.0] * 50,
        "confidence": [0.9] * 50, "was_repaired": [False] * 50,
    }))
    keypoints = _varied_keypoints()
    store.set_parquet("pose", pd.DataFrame([
        {"frame": f, "player_id": p, "keypoints": keypoints.tolist()}
        for f in range(50) for p in ("player_1", "player_2")
    ]))
    store.set("court", {
        "court_length": 13.4, "court_width": 6.1,
        "valid": True, "homography": np.eye(3).tolist(),
    })
    store.set("players", {"players": [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": f, "bbox": [0, 0, 100, 100]} for f in range(50)]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": f, "bbox": [200, 0, 300, 100]} for f in range(50)]},
    ]})

    result = StrokeClassificationStage().run(store, StageConfig(debug_level=1))
    shots = store.get_parquet("shots").sort_values("frame").reset_index(drop=True)

    assert result.status == "success"
    assert shots.loc[0, "bst_input_route"] == "bst"
    # Soft-tier clip keeps its prediction (not 'unknown') and is tagged.
    assert shots.loc[1, "bst_input_route"] == "low_quality_bst"
    assert shots.loc[1, "stroke_source"] == "low_quality_bst"
    assert shots.loc[1, "stroke_type"] != "unknown"
    # Confidence discounted by bst_low_quality_discount (0.8).
    raw = 0.9  # _QualityGateClassifier returns 0.9
    assert shots.loc[1, "stroke_confidence"] == pytest.approx(raw * 0.8, abs=1e-4)


def test_build_clip_masks_low_confidence_joints_in_hip_centered_mode(monkeypatch):
    from app.pipeline.strokes import _build_clip

    monkeypatch.setattr("app.pipeline.strokes.settings.bst_joint_norm", "hip_centered")
    frames = [0]
    shuttle = pd.DataFrame({"frame": frames, "x": [100.0], "y": [100.0], "confidence": [0.9]})
    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    keypoints[10] = [999.0, 999.0, 0.1]
    pose = pd.DataFrame([
        {"frame": 0, "player_id": player, "keypoints": keypoints.tolist()}
        for player in ("player_1", "player_2")
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [{"frame": 0, "bbox": [0, 0, 100, 100]}]},
        {"id": "player_2", "side": "far", "detections": [{"frame": 0, "bbox": [200, 0, 300, 100]}]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 1,
        player_detections=players, player_ids=["player_1", "player_2"],
    )

    np.testing.assert_array_equal(clip["JnB"][0, 0, 20:22], [0.0, 0.0])
    np.testing.assert_array_equal(clip["JnB"][0, 1, 20:22], [0.0, 0.0])


def test_build_clip_masks_low_confidence_joints_in_court_mode(monkeypatch):
    from app.pipeline.strokes import _build_clip

    monkeypatch.setattr("app.pipeline.strokes.settings.bst_joint_norm", "court")
    frames = [0]
    shuttle = pd.DataFrame({"frame": frames, "x": [100.0], "y": [100.0], "confidence": [0.9]})
    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    keypoints[10] = [999.0, 999.0, 0.1]
    pose = pd.DataFrame([
        {"frame": 0, "player_id": player, "keypoints": keypoints.tolist()}
        for player in ("player_1", "player_2")
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [{"frame": 0, "bbox": [0, 0, 100, 100]}]},
        {"id": "player_2", "side": "far", "detections": [{"frame": 0, "bbox": [200, 0, 300, 100]}]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 1,
        player_detections=players, player_ids=["player_1", "player_2"],
        homography=np.eye(3),
    )

    np.testing.assert_array_equal(clip["JnB"][0, 0, 20:22], [0.0, 0.0])
    np.testing.assert_array_equal(clip["JnB"][0, 1, 20:22], [0.0, 0.0])


def test_build_clip_marks_sparse_keypoints_absent(monkeypatch):
    from app.pipeline.strokes import _build_clip

    frames = [0]
    shuttle = pd.DataFrame({"frame": frames, "x": [100.0], "y": [100.0], "confidence": [0.9]})

    # Far player (player_2, p_idx=0): only 2 of 17 keypoints valid.
    far_kps = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
    far_kps[2:, 0] = np.nan
    far_kps[2:, 1] = np.nan
    far_kps[2:, 2] = 0.0
    # Near player (player_1, p_idx=1): all 17 joints valid.
    near_kps = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])

    pose = pd.DataFrame([
        {"frame": 0, "player_id": "player_2", "keypoints": far_kps.tolist()},
        {"frame": 0, "player_id": "player_1", "keypoints": near_kps.tolist()},
    ])
    players = [
        {"id": "player_1", "side": "near", "detections": [{"frame": 0, "bbox": [0, 0, 100, 100]}]},
        {"id": "player_2", "side": "far", "detections": [{"frame": 0, "bbox": [200, 0, 300, 100]}]},
    ]

    clip = _build_clip(
        frames, shuttle, pose, 640, 480, 13.4, 6.1, 1,
        player_detections=players, player_ids=["player_1", "player_2"],
    )

    assert clip["_bst_provenance"]["pose_present_far"][0] is False
    assert np.allclose(clip["JnB"][0, 0, :17], 0.0)


def test_temporal_smoothing_marks_quality_abstention_as_downstream_override(monkeypatch, tmp_job_dir):
    from app.pipeline.shared import models

    classifier = _QualityGateClassifier()
    monkeypatch.setattr(models, "get_bst", lambda: classifier)
    monkeypatch.setattr("app.pipeline.strokes.settings.fusion_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.hierarchical_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.confusion_pair_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.physics_gate_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.stroke_smoothing_window", 1)
    monkeypatch.setattr("app.pipeline.strokes.settings.stroke_smoothing_majority_count", 1)
    # Court-space shuttle normalization so the court-rejected hard gate applies.
    monkeypatch.setattr("app.pipeline.strokes.settings.bst_shuttle_norm", "court")

    store = ArtifactStore(tmp_job_dir)
    # Hits at 0, 30, 60. Under the default midpoint clip boundary the middle
    # hit (30) spans [15:45], covering the court-rejected 30-40 window so it
    # abstains and is recovered by temporal smoothing from its neighbours.
    store.set_parquet("hits", pd.DataFrame({"frame": [0, 30, 60], "confidence": [0.9] * 3}))
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": list(range(80)), "x": [100.0] * 80, "y": [100.0] * 80,
        "confidence": [0.9] * 80, "was_interpolated": [False] * 80,
        "court_rejected": [False] * 30 + [True] * 10 + [False] * 40,
    }))
    store.set_parquet("shuttle_raw", pd.DataFrame({
        "frame": list(range(80)), "x": [100.0] * 80, "y": [100.0] * 80,
        "confidence": [0.9] * 80, "was_repaired": [False] * 80,
    }))
    keypoints = _varied_keypoints()
    store.set_parquet("pose", pd.DataFrame([
        {"frame": f, "player_id": p, "keypoints": keypoints.tolist()}
        for f in range(80) for p in ("player_1", "player_2")
    ]))
    store.set("court", {
        "court_length": 13.4, "court_width": 6.1,
        "valid": True, "homography": np.eye(3).tolist(),
    })
    store.set("players", {"players": [
        {"id": "player_1", "side": "near", "detections": [
            {"frame": f, "bbox": [0, 0, 100, 100]} for f in range(80)
        ]},
        {"id": "player_2", "side": "far", "detections": [
            {"frame": f, "bbox": [200, 0, 300, 100]} for f in range(80)
        ]},
    ]})

    StrokeClassificationStage().run(store, StageConfig())
    shots = store.get_parquet("shots").sort_values("frame").reset_index(drop=True)

    assert shots.loc[1, "stroke_type"] == "smash"
    assert shots.loc[1, "bst_input_route"] == "downstream_override"
    assert shots.loc[1, "stroke_source"] == "temporal_smoothing"
    assert shots.loc[1, "bst_input_override_source"] == "temporal_smoothing"


def test_stroke_stage_marks_aim_alpha_unreliable_when_probe_offsets_flip(monkeypatch, tmp_job_dir):
    from app.pipeline.shared import models

    classifier = _AimAlphaProbeClassifier()
    monkeypatch.setattr(models, "get_bst", lambda: classifier)
    monkeypatch.setattr("app.pipeline.strokes.settings.fusion_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.hierarchical_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.confusion_pair_enabled", False)
    monkeypatch.setattr("app.pipeline.strokes.settings.physics_gate_enabled", False)
    monkeypatch.setattr(
        "app.pipeline.strokes.evaluate_aim_alpha_quality",
        lambda provenance: {
            "reliable": True,
            "score": 1.0,
            "reasons": [],
            "contact_window_valid": True,
            "pose_balance_score": 1.0,
            "identity_stable": True,
            "contact_separation": 0.5,
        },
    )

    store = ArtifactStore(tmp_job_dir)
    store.set_parquet("hits", pd.DataFrame({"frame": [10], "confidence": [0.9]}))
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": list(range(40)),
        "x": [100.0] * 40,
        "y": [100.0] * 40,
        "confidence": [0.9] * 40,
        "was_interpolated": [False] * 40,
        "court_rejected": [False] * 40,
    }))
    store.set_parquet("shuttle_raw", pd.DataFrame({
        "frame": list(range(40)),
        "x": [100.0] * 40,
        "y": [100.0] * 40,
        "confidence": [0.9] * 40,
        "was_repaired": [False] * 40,
    }))
    # Non-collapsed skeleton (distinct joint coordinates) so the clip's joints
    # are not all-zero after normalization — otherwise it is correctly rejected
    # as degenerate before reaching the classifier.
    kp_x = np.arange(17, dtype=float)
    kp_y = np.arange(17, dtype=float) * 2.0 + 10.0
    keypoints = np.column_stack([kp_x, kp_y, np.ones(17)])
    store.set_parquet("pose", pd.DataFrame([
        {"frame": f, "player_id": p, "keypoints": keypoints.tolist()}
        for f in range(40) for p in ("player_1", "player_2")
    ]))
    store.set("court", {"court_length": 13.4, "court_width": 6.1})
    store.set("players", {"players": [
        {"id": "player_1", "side": "near", "detections": [{"frame": f, "bbox": [0, 0, 100, 100]} for f in range(40)]},
        {"id": "player_2", "side": "far", "detections": [{"frame": f, "bbox": [200, 0, 300, 100]} for f in range(40)]},
    ]})

    StrokeClassificationStage().run(store, StageConfig(debug_level=1))
    shots = store.get_parquet("shots").sort_values("frame").reset_index(drop=True)

    assert len(classifier.received) == 3
    assert bool(shots.loc[0, "aim_alpha_reliable"]) is False
    assert shots.loc[0, "aim_alpha_route"] == "alpha_abstain_instability"
    assert shots.loc[0, "aim_alpha_stability_span"] == pytest.approx(0.6)
    assert "probe_direction_flip" in shots.loc[0, "aim_alpha_quality_reasons"]
