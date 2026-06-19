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
        Supports hybrid mode: MMPose primary + RTMPose secondary.
        """
        if pose_data:
            return self._store_data(artifacts, pose_data)

        if frames:
            pose_data = self._run_pose(frames, artifacts)
            return self._store_data(artifacts, pose_data)

        return StageResult.from_error("No frames or pose data provided")

    def _run_pose(self, frames: list[np.ndarray], artifacts: ArtifactStore) -> list[dict]:
        """Run pose estimation based on configured model."""
        from app.config.settings import settings

        pose_model = settings.pose_model

        if pose_model == "hybrid":
            return self._run_hybrid(frames, artifacts)
        elif pose_model == "mmpose":
            return self._run_mmpose(frames, artifacts)
        else:
            return self._run_rtmpose(frames, artifacts)

    def _run_hybrid(self, frames: list[np.ndarray], artifacts: ArtifactStore) -> list[dict]:
        """Run hybrid mode: MMPose primary (strokes) + RTMPose secondary (hits)."""
        from app.models.rtmpose import RTMPoseEstimator
        from app.config.settings import settings

        # Primary: MMPose HRNet (for stroke classification)
        hrnet_path = str(settings.hrnet_model_path) if settings.hrnet_model_path else None
        primary_estimator = RTMPoseEstimator(hrnet_path, device=settings.device)

        # Secondary: RTMPose (for hit confidence and fitness)
        rtmpose_path = str(settings.rtmpose_model_path) if settings.rtmpose_model_path else None
        secondary_estimator = RTMPoseEstimator(rtmpose_path, device=settings.device)

        pose_data = self._estimate_with_estimator(frames, artifacts, primary_estimator)

        # Store secondary pose data for fitness analytics
        secondary_pose = self._estimate_with_estimator(frames, artifacts, secondary_estimator)
        artifacts.set("pose_secondary", secondary_pose)

        return pose_data

    def _run_mmpose(self, frames: list[np.ndarray], artifacts: ArtifactStore) -> list[dict]:
        """Run MMPose HRNet for pose estimation."""
        from app.models.rtmpose import RTMPoseEstimator
        from app.config.settings import settings

        hrnet_path = str(settings.hrnet_model_path) if settings.hrnet_model_path else None
        estimator = RTMPoseEstimator(hrnet_path, device=settings.device)
        return self._estimate_with_estimator(frames, artifacts, estimator)

    def _run_rtmpose(self, frames: list[np.ndarray], artifacts: ArtifactStore) -> list[dict]:
        """Run RTMPose on video frames using player detections."""
        from app.models.rtmpose import RTMPoseEstimator
        from app.config.settings import settings

        model_path = str(settings.rtmpose_model_path) if settings.rtmpose_model_path else None
        estimator = RTMPoseEstimator(model_path, device=settings.device)
        return self._estimate_with_estimator(frames, artifacts, estimator)

    def _estimate_with_estimator(self, frames: list[np.ndarray], artifacts: ArtifactStore, estimator) -> list[dict]:
        """Run pose estimation with given estimator."""
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
