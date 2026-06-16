import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PoseEstimationStage:
    name = "pose_estimation"
    input_keys = ["players"]
    output_keys = ["pose"]

    def run(
        self,
        artifacts: ArtifactStore,
        config: StageConfig,
        frames: list[np.ndarray] | None = None,
        pose_data: list[dict] | None = None
    ) -> StageResult:
        """Run pose estimation.

        If frames provided, runs RTMPose inference.
        If pose_data provided, uses pre-computed data.
        """
        if pose_data:
            return self._store_data(artifacts, pose_data)

        if frames:
            pose_data = self._run_rtmpose(frames, artifacts)
            return self._store_data(artifacts, pose_data)

        return StageResult.from_error("No frames or pose data provided")

    def _run_rtmpose(self, frames: list[np.ndarray], artifacts: ArtifactStore) -> list[dict]:
        """Run RTMPose on video frames using player detections."""
        from app.models.rtmpose import RTMPoseEstimator
        from app.config.settings import settings

        model_path = str(settings.rtmpose_model_path) if settings.rtmpose_model_path else None
        estimator = RTMPoseEstimator(model_path, device=settings.device)

        players = artifacts.get("players")
        if not players:
            return []

        player_list = players.get("players", [])
        if not player_list:
            return []

        pose_data = []
        for frame_idx, frame in enumerate(frames):
            for player in player_list:
                player_id = player["id"]

                # Find detection for this specific frame
                bbox = None
                for det in player.get("detections", []):
                    if det.get("frame") == frame_idx:
                        bbox = det.get("bbox")
                        break
                if bbox is None:
                    # Fallback: use closest detection in time
                    dets = player.get("detections", [])
                    if dets:
                        closest = min(dets, key=lambda d: abs(d.get("frame", 0) - frame_idx))
                        bbox = closest.get("bbox", (100, 100, 300, 400))
                    else:
                        bbox = (100, 100, 300, 400)

                keypoints = estimator.estimate(frame, bbox)

                pose_data.append({
                    "frame": frame_idx,
                    "player_id": player_id,
                    "keypoints": keypoints.tolist(),
                })

        return pose_data

    def _store_data(self, artifacts: ArtifactStore, pose_data: list[dict]) -> StageResult:
        """Store pose estimation data."""
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
