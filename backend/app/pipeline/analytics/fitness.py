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
                    
                    duration_frames = rally["end_frame"] - rally["start_frame"]
                    duration_seconds = max(duration_frames / 30, 0.1)
                    intensity = len(rally_shots) / duration_seconds
                    rally_intensities.append(float(intensity))

            fatigue_trend = self._compute_fatigue_trend(rally_intensities)
            
            avg_intensity = float(np.mean(rally_intensities)) if rally_intensities else 0
            peak_intensity = float(np.max(rally_intensities)) if rally_intensities else 0
            intensity_std = float(np.std(rally_intensities)) if rally_intensities else 0
            late_rally_fatigue = self._compute_late_rally_fatigue(rally_intensities)

            fitness[player_id] = {
                "rally_intensity": avg_intensity,
                "rally_intensities": rally_intensities,
                "fatigue_trend": fatigue_trend,
                "avg_recovery": fw_data.get("avg_recovery", 0),
                "total_distance": fw_data.get("distance_covered", 0),
                "peak_intensity": peak_intensity,
                "intensity_std": intensity_std,
                "late_rally_fatigue": late_rally_fatigue,
                "rally_count": len(rally_intensities),
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
    def _compute_fatigue_trend(rally_intensities: list[float]) -> str:
        """Analyze rally intensity over time to detect fatigue.
        
        Uses quarter comparison + linear regression slope.
        For rally intensity: higher = more active = less fatigued.
        """
        if len(rally_intensities) < 5:
            return "insufficient_data"
        
        n = len(rally_intensities)
        
        q1 = rally_intensities[:n//4]
        q4 = rally_intensities[3*n//4:]
        
        avg_q1 = np.mean(q1) if q1 else 0
        avg_q4 = np.mean(q4) if q4 else 0
        
        x = np.arange(len(rally_intensities))
        slope = np.polyfit(x, rally_intensities, 1)[0]
        
        avg_intensity = np.mean(rally_intensities)
        normalized_slope = slope / avg_intensity if avg_intensity > 0 else 0
        
        # For rally intensity: declining means Q4 < Q1 (fatigued)
        if avg_q4 < avg_q1 * 0.8 and normalized_slope < -0.01:
            return "declining"
        elif avg_q4 > avg_q1 * 1.2 and normalized_slope > 0.01:
            return "improving"
        return "stable"
    
    @staticmethod
    def _compute_late_rally_fatigue(rally_intensities: list[float]) -> float:
        """Compute fatigue factor from late rallies vs early rallies."""
        if len(rally_intensities) < 6:
            return 0.0
        
        first_half = rally_intensities[:len(rally_intensities)//2]
        second_half = rally_intensities[len(rally_intensities)//2:]
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        if avg_first == 0:
            return 0.0
        
        return float((avg_first - avg_second) / avg_first)
