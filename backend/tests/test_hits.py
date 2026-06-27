import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import HitFrameLocalizationStage


def test_hit_detection_finds_trajectory_changes(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    n = 40
    frames = list(range(n))
    x = [100.0 + t * 5.0 for t in range(20)] + [200.0 - (t - 20) * 8.0 for t in range(20, n)]
    y = [200.0 - t * 2.0 for t in range(20)] + [160.0 + (t - 20) * 4.0 for t in range(20, n)]
    shuttle_df = pd.DataFrame({
        "frame": frames,
        "x": x,
        "y": y,
        "confidence": [0.95] * n,
    })
    store.set_parquet("shuttle_raw", shuttle_df)

    pose_df = pd.DataFrame({
        "frame": list(range(20)),
        "player_id": ["player_1"] * 20,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(20)],
    })
    store.set_parquet("pose", pose_df)

    stage = HitFrameLocalizationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "hits" in result.metadata
    assert result.metadata["hit_count"] > 0
