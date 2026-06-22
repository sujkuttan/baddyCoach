import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import StrokeClassificationStage


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

    court_data = {"court_length": 13.4, "court_width": 5.18}
    store.set("court", court_data)

    stage = StrokeClassificationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert len(shots_df) == 4
    assert "stroke_type" in shots_df.columns
    assert "stroke_confidence" in shots_df.columns


def test_stroke_classification_empty_hits(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    hits_df = pd.DataFrame({"frame": [], "confidence": []})
    store.set_parquet("hits", hits_df)

    stage = StrokeClassificationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert result.metadata["shot_count"] == 0
