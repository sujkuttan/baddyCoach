import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import PoseEstimationStage


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


def test_fallback_bbox_interpolates_between_same_player_detections():
    lookup = {
        "player_1": {
            0: [0.0, 0.0, 10.0, 10.0],
            2: [10.0, 10.0, 20.0, 20.0],
        },
    }

    bbox = PoseEstimationStage._find_fallback_bbox(1, "player_1", lookup, range_limit=2)

    assert bbox == [5.0, 5.0, 15.0, 15.0]


def test_fallback_bbox_uses_nearest_same_player_box_at_video_edge():
    lookup = {"player_1": {3: [10.0, 20.0, 30.0, 40.0]}}

    bbox = PoseEstimationStage._find_fallback_bbox(0, "player_1", lookup, range_limit=3)

    assert bbox == [10.0, 20.0, 30.0, 40.0]
