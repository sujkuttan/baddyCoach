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
        """Run pose estimation with given estimator, using same-side fallback."""
        players_data = artifacts.get("players")
        if not players_data:
            return []

        player_list = players_data.get("players", [])
        if not player_list:
            return []

        side_map = {p["id"]: p.get("side", "near") for p in player_list}
        det_lookup = {}
        for p in player_list:
            pid = p["id"]
            det_lookup[pid] = {d["frame"]: d["bbox"] for d in p.get("detections", [])}

        pose_data = []
        for frame_idx, frame in enumerate(frames):
            crops_for_frame = []

            for player in player_list:
                player_id = player["id"]
                bbox = det_lookup.get(player_id, {}).get(frame_idx)

                if bbox is None:
                    bbox = self._find_fallback_bbox(
                        frame_idx, player_id, side_map, det_lookup, range_limit=10
                    )

                if bbox is None:
                    pose_data.append({
                        "frame": frame_idx,
                        "player_id": player_id,
                        "keypoints": np.zeros((17, 3), dtype=np.float32).tolist(),
                    })
                    continue

                crops_for_frame.append((player_id, bbox))

            if crops_for_frame:
                bboxes = [c[1] for c in crops_for_frame]
                keypoints_list = estimator.estimate_batch(frame, bboxes)
                for (player_id, _), kps in zip(crops_for_frame, keypoints_list):
                    pose_data.append({
                        "frame": frame_idx,
                        "player_id": player_id,
                        "keypoints": kps.tolist(),
                    })

        return pose_data

    @staticmethod
    def _find_fallback_bbox(frame_idx: int, player_id: str, side_map: dict,
                            det_lookup: dict, range_limit: int = 10):
        """Find fallback bbox: same-side first, then any side."""
        my_side = side_map.get(player_id, "near")
        same_side_pids = [pid for pid, s in side_map.items() if s == my_side and pid != player_id]
        other_pids = [pid for pid, s in side_map.items() if s != my_side]

        for delta in range(1, range_limit + 1):
            for pid in same_side_pids:
                dets = det_lookup.get(pid, {})
                for offset in [frame_idx + delta, frame_idx - delta]:
                    if offset in dets:
                        return dets[offset]
            for pid in other_pids:
                dets = det_lookup.get(pid, {})
                for offset in [frame_idx + delta, frame_idx - delta]:
                    if offset in dets:
                        return dets[offset]

        return None

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
