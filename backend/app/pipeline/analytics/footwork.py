import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class FootworkAnalyticsStage:
    name = "footwork_analytics"
    input_keys = ["pose", "court", "rallies", "shots"]
    output_keys = ["footwork_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court")
        rallies_df = artifacts.get_parquet("rallies")
        shots_df = artifacts.get_parquet("shots")

        if pose_df is None or court is None:
            return StageResult.from_error("Pose and court data required")

        court_length = court["court_length"]
        court_width = court["court_width"]
        base_position = np.array([court_width / 2, court_length / 2])

        metrics = {}
        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            com_trajectory = self._extract_com(player_poses)

            distance = self._compute_distance(com_trajectory)
            recovery_times = self._compute_recovery_times(player_poses, shots_df, base_position) if shots_df is not None else []

            metrics[player_id] = {
                "distance_covered": float(distance),
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
        diffs = np.diff(com_trajectory, axis=0)
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
