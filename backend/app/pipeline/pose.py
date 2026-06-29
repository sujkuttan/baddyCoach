import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger


COCO_R_SHOULDER, COCO_R_ELBOW, COCO_R_WRIST = 6, 8, 10
COCO_L_SHOULDER, COCO_L_ELBOW, COCO_L_WRIST = 5, 7, 9
COCO_R_HIP, COCO_L_HIP = 12, 11


def _vec(a, b):
    return np.array(b) - np.array(a)


def _angle_at_joint(a, b, c):
    """Interior angle (degrees) at b formed by a→b and b→c."""
    ba = _vec(b, a)
    bc = _vec(c, b)
    dot = float(np.dot(ba, bc))
    norm = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if norm < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def _angle_from_horizontal(a, b):
    """Angle (degrees) of vector a→b relative to horizontal (0 = rightward)."""
    d = _vec(a, b)
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _torso_angle_from_vertical(shoulder_mid, hip_mid):
    """Angle (degrees) of torso centreline from vertical."""
    d = _vec(hip_mid, shoulder_mid)
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    vertical = np.array([0.0, -1.0])
    dot = float(np.dot(d, vertical))
    norm = float(np.linalg.norm(d))
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def _compute_pose_angles(kps: np.ndarray) -> dict:
    """Compute derived joint angles from 17×3 COCO keypoints.

    Returns dict with right-side primary angles and left-side for symmetry.
    Returns 0.0 for any unavailable keypoint (confidence < 0.1 or all-zeros).
    """
    result = {"elbow_angle": 0.0, "shoulder_angle": 0.0, "torso_angle": 0.0}

    def _valid(idx):
        if idx >= len(kps):
            return False
        x, y, c = kps[idx]
        return bool(c > 0.1) and not (x == 0 and y == 0)

    def _pt(idx):
        return kps[idx, :2]

    # Right-arm angles (dominant hand assumed right)
    if all(_valid(i) for i in (COCO_R_SHOULDER, COCO_R_ELBOW, COCO_R_WRIST)):
        result["elbow_angle"] = _angle_at_joint(
            _pt(COCO_R_SHOULDER), _pt(COCO_R_ELBOW), _pt(COCO_R_WRIST)
        )
        result["shoulder_angle"] = _angle_from_horizontal(
            _pt(COCO_R_SHOULDER), _pt(COCO_R_ELBOW)
        )

    # Torso angle from vertical (shoulder midpoint → hip midpoint)
    if all(_valid(i) for i in (COCO_R_SHOULDER, COCO_L_SHOULDER, COCO_R_HIP, COCO_L_HIP)):
        shoulder_mid = (_pt(COCO_R_SHOULDER) + _pt(COCO_L_SHOULDER)) / 2.0
        hip_mid = (_pt(COCO_R_HIP) + _pt(COCO_L_HIP)) / 2.0
        result["torso_angle"] = _torso_angle_from_vertical(shoulder_mid, hip_mid)

    # Left-arm angles (secondary, when right-side is unavailable)
    if result["elbow_angle"] == 0.0 and all(_valid(i) for i in (COCO_L_SHOULDER, COCO_L_ELBOW, COCO_L_WRIST)):
        result["elbow_angle"] = _angle_at_joint(
            _pt(COCO_L_SHOULDER), _pt(COCO_L_ELBOW), _pt(COCO_L_WRIST)
        )
        result["shoulder_angle"] = _angle_from_horizontal(
            _pt(COCO_L_SHOULDER), _pt(COCO_L_ELBOW)
        )

    return result


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
        """Run pose estimation based on configured model.

        Model loading is local to this stage (not via shared.models.setup_models)
        to keep the colab pipeline's self-contained model loading approach intact.
        """
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
                        frame_idx, player_id, det_lookup, range_limit=10
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
    def _find_fallback_bbox(frame_idx: int, player_id: str,
                            det_lookup: dict, range_limit: int = 10):
        """Find fallback bbox via temporal interpolation within same player's own detections."""
        my_dets = det_lookup.get(player_id, {})
        for delta in range(1, range_limit + 1):
            for offset in (frame_idx + delta, frame_idx - delta):
                if offset in my_dets:
                    return my_dets[offset]
        return None

    def _store_data(self, artifacts: ArtifactStore, pose_data: list[dict]) -> StageResult:
        """Store pose estimation data with computed joint angles."""
        records = []
        angle_records = []
        for entry in pose_data:
            records.append({
                "frame": entry["frame"],
                "player_id": entry["player_id"],
                "keypoints": entry["keypoints"],
            })
            # Compute derived angles per frame/player
            kps = np.array(entry["keypoints"], dtype=np.float64)
            angles = _compute_pose_angles(kps)
            angle_records.append({
                "frame": entry["frame"],
                "player_id": entry["player_id"],
                "elbow_angle": angles["elbow_angle"],
                "shoulder_angle": angles["shoulder_angle"],
                "torso_angle": angles["torso_angle"],
            })

        df = pd.DataFrame(records)
        if angle_records:
            angle_df = pd.DataFrame(angle_records)
            for col in ("elbow_angle", "shoulder_angle", "torso_angle"):
                df[col] = angle_df[col].values
        artifacts.set_parquet("pose", df)

        logger.info(f"Stored {len(df)} pose records across {df['player_id'].nunique()} players")

        return StageResult.success(
            artifacts={"pose": artifacts.path("pose")},
            metadata={
                "total_frames": df["frame"].nunique(),
                "players": df["player_id"].unique().tolist(),
                "keypoints_per_player": 17,
            }
        )
