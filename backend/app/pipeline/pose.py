import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PoseEstimationStage:
    name = "pose_estimation"
    input_keys = ["players"]
    output_keys = ["pose"]

    def run(self, artifacts: ArtifactStore, config: StageConfig, pose_data: list[dict] | None = None) -> StageResult:
        if not pose_data:
            return StageResult.from_error("No pose data provided")

        records = []
        for entry in pose_data:
            records.append({
                "frame": entry["frame"],
                "player_id": entry["player_id"],
                "keypoints": entry["keypoints"],
            })

        df = pd.DataFrame(records)
        artifacts.set_parquet("pose", df)

        return StageResult.success(
            artifacts={"pose": artifacts.path("pose")},
            metadata={
                "total_frames": df["frame"].nunique(),
                "players": df["player_id"].unique().tolist(),
                "keypoints_per_player": 17,
            }
        )


def smooth_keypoints(keypoints: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    smoothed = np.copy(keypoints)
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * keypoints[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed
