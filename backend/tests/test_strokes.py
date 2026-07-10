import numpy as np
import pandas as pd
import pytest
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import StrokeClassificationStage
from app.pipeline.strokes import _temporal_resample


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


def test_build_clip_zeros_court_rejected_shuttle_and_records_provenance():
    from app.pipeline.strokes import _build_clip

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

    np.testing.assert_array_equal(clip["shuttle"][1], [0.0, 0.0])
    assert clip["_bst_provenance"]["shuttle_observed"] == [True, False, True]
    assert clip["_bst_provenance"]["shuttle_repaired"] == [False, True, False]
    assert clip["_bst_provenance"]["shuttle_interpolated"] == [False, True, False]
    assert clip["_bst_provenance"]["shuttle_court_rejected"] == [False, True, False]
