import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.pose import PoseEstimationStage


def test_pose_estimation_stores_keypoints(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    pose_data = []
    for frame in range(3):
        for player_id in ["player_1", "player_2"]:
            keypoints = np.random.rand(17, 3).tolist()
            pose_data.append({
                "frame": frame,
                "player_id": player_id,
                "keypoints": keypoints,
            })

    stage = PoseEstimationStage()
    result = stage.run(store, config, pose_data=pose_data)

    assert result.status == "success"
    df = store.get_parquet("pose")
    assert len(df) == 6
    assert "frame" in df.columns
    assert "player_id" in df.columns
    assert "keypoints" in df.columns
