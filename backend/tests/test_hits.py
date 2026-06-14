import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.hits import HitFrameLocalizationStage


def test_hit_detection_finds_trajectory_changes(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shuttle_df = pd.DataFrame({
        "frame": list(range(20)),
        "x": [100, 120, 140, 160, 180, 170, 150, 130, 110, 100,
              120, 140, 160, 180, 170, 150, 130, 110, 100, 120],
        "y": [200, 190, 180, 170, 160, 170, 180, 190, 200, 210,
              200, 190, 180, 170, 180, 190, 200, 210, 220, 210],
        "confidence": [0.95] * 20,
    })
    store.set_parquet("shuttle", shuttle_df)

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
