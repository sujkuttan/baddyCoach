import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class FitnessAnalyticsStage:
    name = "fitness_analytics"
    input_keys = ["footwork_analytics", "rallies", "shots"]
    output_keys = ["fitness_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        footwork = artifacts.get("footwork_analytics")
        rallies_df = artifacts.get_parquet("rallies")
        shots_df = artifacts.get_parquet("shots")

        if footwork is None:
            return StageResult.from_error("Footwork analytics required")

        fitness = {}
        for player_id, fw_data in footwork.items():
            rally_intensities = []
            if rallies_df is not None and shots_df is not None:
                for _, rally in rallies_df.iterrows():
                    rally_shots = shots_df[
                        (shots_df["frame"] >= rally["start_frame"]) &
                        (shots_df["frame"] <= rally["end_frame"]) &
                        (shots_df["player_id"] == player_id)
                    ]
                    intensity = len(rally_shots) / max((rally["end_frame"] - rally["start_frame"]) / 30, 1)
                    rally_intensities.append(float(intensity))

            fatigue_trend = self._compute_fatigue_trend(fw_data.get("recovery_times", []))

            fitness[player_id] = {
                "rally_intensity": float(np.mean(rally_intensities)) if rally_intensities else 0,
                "rally_intensities": rally_intensities,
                "fatigue_trend": fatigue_trend,
                "avg_recovery": fw_data.get("avg_recovery", 0),
                "total_distance": fw_data.get("distance_covered", 0),
            }

        artifacts.set("fitness_analytics", fitness)

        return StageResult.success(
            artifacts={"fitness_analytics": artifacts.path("fitness_analytics")},
            metadata={
                "rally_intensity": {k: v["rally_intensity"] for k, v in fitness.items()},
                "fatigue_trend": {k: v["fatigue_trend"] for k, v in fitness.items()},
            }
        )

    @staticmethod
    def _compute_fatigue_trend(recovery_times: list[float]) -> str:
        if len(recovery_times) < 3:
            return "insufficient_data"

        first_half = recovery_times[:len(recovery_times) // 2]
        second_half = recovery_times[len(recovery_times) // 2:]

        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)

        if avg_second > avg_first * 1.2:
            return "declining"
        elif avg_second < avg_first * 0.8:
            return "improving"
        return "stable"
