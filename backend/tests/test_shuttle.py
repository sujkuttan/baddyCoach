import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.shuttle import ShuttleTrackingStage


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
    assert list(df.columns) == ["frame", "x", "y", "confidence"]


def test_shuttle_tracking_empty_data(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = ShuttleTrackingStage()
    result = stage.run(store, config, shuttle_data=[])

    assert result.status == "error"
