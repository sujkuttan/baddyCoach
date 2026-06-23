import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court
from app.pipeline.shared.logging import logger
from app.config.settings import settings

COURT_WIDTH_M = COURT_WIDTH
COURT_LENGTH_M = COURT_LENGTH


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

            # Convert pixel coordinates to court coordinates using homography
            homography = court.get("homography")
            distance = self._compute_distance(com_trajectory, homography)
            # When homography is available, _compute_distance returns meters directly
            distance_m = distance if homography is not None else (distance / px_per_m if px_per_m > 0 else distance)
            fps = float(config.processing_fps or settings.fps)
            recovery_times = self._compute_recovery_times(player_poses, shots_df, base_position, fps, homography, px_per_m) if shots_df is not None else []

            metrics[player_id] = {
                "distance_covered": float(distance_m),
                "recovery_times": recovery_times,
                "avg_recovery": float(np.mean(recovery_times)) if recovery_times else 0,
            }

        logger.info(f"Computed footwork analytics for {len(metrics)} players")

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
    def _compute_distance(com_trajectory: np.ndarray, homography: np.ndarray | None = None) -> float:
        if len(com_trajectory) < 2:
            return 0.0
        
        # Convert pixel coordinates to court coordinates using homography
        if homography is not None:
            H = np.array(homography)
            court_trajectory = []
            for point in com_trajectory:
                court_x, court_y = image_to_court(H, point)
                court_trajectory.append((court_x, court_y))
            court_pts = [np.array(pt) for pt in court_trajectory]
            filtered = [court_pts[0]]
            for i in range(1, len(court_pts)):
                jump = np.sqrt(np.sum((court_pts[i] - filtered[-1])**2))
                if jump < 2.0:
                    filtered.append(court_pts[i])
            if len(filtered) < 2:
                return 0.0
            filtered = np.array(filtered)
            diffs = np.diff(filtered, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            return float(np.sum(distances))
        else:
            # Fallback to pixel-based distance calculation
            filtered = [com_trajectory[0]]
            for i in range(1, len(com_trajectory)):
                jump = np.sqrt(np.sum((com_trajectory[i] - com_trajectory[i-1])**2))
                if jump < settings.footwork_jump_filter_pixels:
                    filtered.append(com_trajectory[i])
            if len(filtered) < 2:
                return 0.0
            filtered = np.array(filtered)
            diffs = np.diff(filtered, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            return float(np.sum(distances))

    @staticmethod
    def _compute_recovery_times(pose_df: pd.DataFrame, shots_df: pd.DataFrame, base_position: np.ndarray, fps: float, homography: np.ndarray | None = None, px_per_m: float = 1.0) -> list[float]:
        recovery_times = []
        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            player_id = shot.get("player_id")
            
            # Only compute recovery for the player who hit the shot
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            after_shots = player_poses[player_poses["frame"] > frame].head(settings.footwork_recovery_lookahead_frames)
            if len(after_shots) == 0:
                continue

            com_points = FootworkAnalyticsStage._extract_com(after_shots)
            if len(com_points) == 0:
                continue

            threshold_m = settings.footwork_recovery_threshold_meters

            if homography is not None:
                H = np.array(homography)
                court_coms = [image_to_court(H, pt) for pt in com_points]
                court_base = image_to_court(H, base_position)
                distances = np.sqrt(np.sum((np.array(court_coms) - court_base) ** 2, axis=1))
            else:
                threshold_m *= px_per_m
                distances = np.sqrt(np.sum((com_points - base_position) ** 2, axis=1))

            returned = np.where(distances < threshold_m)[0]
            if len(returned) > 0:
                recovery_times.append(float(returned[0]) / fps)

        return recovery_times
