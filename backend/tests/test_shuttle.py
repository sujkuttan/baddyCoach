import numpy as np
import pandas as pd
import pytest
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import ShuttleTrackingStage
from app.pipeline.shuttle import _add_court_space_columns


def test_shuttle_tracking_stores_parquet(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shuttle_data = [
        {"frame": 0, "x": 100.0, "y": 200.0, "confidence": 0.95},
        {"frame": 1, "x": 150.0, "y": 180.0, "confidence": 0.92},
        {"frame": 2, "x": 200.0, "y": 250.0, "confidence": 0.88},
    ]

    stage = ShuttleTrackingStage()
    result = stage.run(store, config, shuttle_data=shuttle_data)

    assert result.status == "success"
    assert "shuttle" in result.artifacts
    df = store.get_parquet("shuttle")
    assert len(df) == 3
    assert "frame" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert "velocity" in df.columns
    assert "acceleration" in df.columns
    assert "curvature" in df.columns


def test_shuttle_tracking_empty_data(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = ShuttleTrackingStage()
    result = stage.run(store, config, shuttle_data=[])

    assert result.status == "error"


def test_court_enrichment_rejects_out_of_bounds_and_impossible_speed(monkeypatch):
    """Court-space validation must not clamp or mutate raw detections."""
    from app.pipeline import shuttle

    monkeypatch.setattr(shuttle.settings, "shuttle_oob_margin_meters", 0.25, raising=False)
    monkeypatch.setattr(shuttle.settings, "shuttle_max_speed_mps", 20.0, raising=False)
    detections = pd.DataFrame([
        {"frame": 0, "x": 13.5, "y": 1.0, "confidence": 0.9},
        {"frame": 1, "x": 14.0, "y": 1.0, "confidence": 0.9},
        {"frame": 2, "x": 1.2, "y": 1.0, "confidence": 0.9},
        {"frame": 3, "x": 5.2, "y": 1.0, "confidence": 0.9},
    ])

    enriched = _add_court_space_columns(detections, np.eye(3), fps=30.0)

    assert enriched["court_rejected"].tolist() == [False, True, False, True]
    assert enriched.loc[0, "x_court"] == 13.5
    assert enriched.loc[1, ["x", "y"]].tolist() == [14.0, 1.0]
    assert enriched.loc[3, ["x", "y"]].tolist() == [5.2, 1.0]
    assert np.isnan(enriched.loc[1, ["x_court", "y_court", "speed_court", "direction_x", "direction_y"]]).all()
    assert np.isnan(enriched.loc[3, ["x_court", "y_court", "speed_court", "direction_x", "direction_y"]]).all()
    assert enriched.loc[2, "x_court"] == 1.2


def test_shuttle_in_court_fraction_and_reliability_flag():
    from app.pipeline.shuttle import compute_shuttle_in_court_fraction

    # Identity H: pixel == court metres. Points at x=20 are OOB for length 13.4.
    df = pd.DataFrame({
        "x": [1.0, 2.0, 20.0, 3.0],
        "y": [1.0, 1.0, 1.0, 1.0],
        "confidence": [0.9, 0.9, 0.9, 0.2],  # last ignored by conf gate
    })
    frac = compute_shuttle_in_court_fraction(df, np.eye(3), min_conf=0.5, oob_margin=1.0)
    # 2 of 3 high-conf points in bounds
    assert frac == pytest.approx(2 / 3, abs=1e-6)
