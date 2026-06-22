import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult

COURT_WIDTH_M = 5.18
COURT_LENGTH_M = 13.4


def _pixel_to_meter_scale(court: dict) -> float:
    """Estimate pixels-per-meter from court corner pixel coordinates.

    Uses the average of near-side and far-side pixel widths (court is 5.18m wide).
    Falls back to 1.0 if corners unavailable.
    """
    corners = court.get("corners_pixel", [])
    if len(corners) < 4:
        return 1.0
    bl, br, tl, tr = corners[:4]
    near_w = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    far_w = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    avg_px = (near_w + far_w) / 2.0
    if avg_px < 1.0:
        return 1.0
    return avg_px / COURT_WIDTH_M


class FootworkAnalyticsStage:
    name = "footwork_analytics"
    input_keys = ["pose", "court", "rallies", "shots"]
    output_keys = ["footwork_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        # Use secondary pose (RTMPose) for footwork if available (hybrid mode)
        pose_df = artifacts.get_parquet("pose")
        secondary_pose = artifacts.get("pose_secondary")
        if secondary_pose and isinstance(secondary_pose, list) and len(secondary_pose) > 0:
            import pandas as pd
            pose_df = pd.DataFrame(secondary_pose)

        court = artifacts.get("court")
        rallies_df = artifacts.get_parquet("rallies")
        shots_df = artifacts.get_parquet("shots")

        if pose_df is None or court is None:
            return StageResult.from_error("Pose and court data required")

        court_length = court["court_length"]
        court_width = court["court_width"]

        px_per_m = _pixel_to_meter_scale(court)

        metrics = {}
        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            com_trajectory = self._extract_com(player_poses)

            if len(com_trajectory) > 0:
                base_position = np.median(com_trajectory, axis=0)
            else:
                base_position = np.array([court_width / 2, court_length / 2])

            distance_px = self._compute_distance(com_trajectory)
            distance_m = distance_px / px_per_m if px_per_m > 0 else distance_px
            recovery_times = self._compute_recovery_times(player_poses, shots_df, base_position) if shots_df is not None else []

            metrics[player_id] = {
                "distance_covered": float(distance_m),
                "recovery_times": recovery_times,
                "avg_recovery": float(np.mean(recovery_times)) if recovery_times else 0,
            }

        artifacts.set("footwork_analytics", metrics)

        return StageResult.success(
            artifacts={"footwork_analytics": artifacts.path("footwork_analytics")},
            metadata={
                "distance_covered": {k: v["distance_covered"] for k, v in metrics.items()},
                "recovery_times": {k: v["avg_recovery"] for k, v in metrics.items()},
            }
        )

    @staticmethod
    def _extract_com(player_poses: pd.DataFrame) -> np.ndarray:
        com_points = []
        for _, row in player_poses.iterrows():
            kps = np.array(row["keypoints"].tolist())
            if kps.shape != (17, 3):
                kps = np.array(kps.tolist())
            if kps.shape == (17, 3):
                left_hip = kps[11][:2]
                right_hip = kps[12][:2]
                com = (left_hip + right_hip) / 2
                com_points.append(com)
        return np.array(com_points) if com_points else np.zeros((0, 2))

    @staticmethod
    def _compute_distance(com_trajectory: np.ndarray) -> float:
        if len(com_trajectory) < 2:
            return 0.0
        # Filter out identity switches: large jumps (>500px) indicate pose switching between players
        filtered = [com_trajectory[0]]
        for i in range(1, len(com_trajectory)):
            jump = np.sqrt(np.sum((com_trajectory[i] - com_trajectory[i-1])**2))
            if jump < 500:
                filtered.append(com_trajectory[i])
        if len(filtered) < 2:
            return 0.0
        filtered = np.array(filtered)
        diffs = np.diff(filtered, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return float(np.sum(distances))

    @staticmethod
    def _compute_recovery_times(pose_df: pd.DataFrame, shots_df: pd.DataFrame, base_position: np.ndarray) -> list[float]:
        recovery_times = []
        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            after_shots = pose_df[pose_df["frame"] > frame].head(30)
            if len(after_shots) == 0:
                continue

            com_points = FootworkAnalyticsStage._extract_com(after_shots)
            if len(com_points) == 0:
                continue

            distances = np.sqrt(np.sum((com_points - base_position) ** 2, axis=1))
            threshold = 0.3
            returned = np.where(distances < threshold)[0]
            if len(returned) > 0:
                recovery_times.append(float(returned[0]))

        return recovery_times
