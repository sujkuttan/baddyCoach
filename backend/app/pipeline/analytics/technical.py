import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class TechnicalAnalyticsStage:
    name = "technical_analytics"
    input_keys = ["shots", "pose", "shuttle", "court"]
    output_keys = ["technical_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court")

        if shots_df is None or pose_df is None:
            return StageResult.from_error("Shot and pose data required")

        technical = {}
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            player_poses = pose_df[pose_df["player_id"] == player_id]

            assessments = {}
            for stroke_type in player_shots["stroke_type"].unique():
                type_shots = player_shots[player_shots["stroke_type"] == stroke_type]
                scores = []
                for _, shot in type_shots.iterrows():
                    frame = int(shot["frame"])
                    pose_row = player_poses[player_poses["frame"] == frame]
                    if len(pose_row) > 0:
                        score = self._evaluate_shot(shot["stroke_type"], pose_row.iloc[0])
                        scores.append(score)

                assessments[stroke_type] = {
                    "avg_score": float(np.mean(scores)) if scores else 0,
                    "shot_count": len(type_shots),
                    "scores": scores,
                }

            technical[player_id] = assessments

        artifacts.set("technical_analytics", technical)

        return StageResult.success(
            artifacts={"technical_analytics": artifacts.path("technical_analytics")},
            metadata={"technical_assessment": technical}
        )

    @staticmethod
    def _evaluate_shot(stroke_type: str, pose_row: pd.Series) -> float:
        kps = np.array(pose_row["keypoints"].tolist())
        if kps.shape != (17, 3):
            return 0.5

        if stroke_type in ("smash", "clear"):
            shoulder = kps[5][:2]
            wrist = kps[9][:2]
            height_diff = shoulder[1] - wrist[1]
            return min(1.0, max(0.0, height_diff / 100.0 + 0.3))

        elif stroke_type == "net_shot":
            knee = kps[13][:2]
            hip = kps[11][:2]
            lunge_depth = abs(knee[1] - hip[1])
            return min(1.0, max(0.0, lunge_depth / 80.0 + 0.2))

        return 0.5
