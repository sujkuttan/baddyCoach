import json
from pathlib import Path
from typing import Any
import numpy as np

import pandas as pd

from app.storage.artifacts import ArtifactStore


def _clean_nan(obj: Any) -> Any:
    """Recursively replace NaN/Infinity with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


def _assign_rally_ids(shots_df: pd.DataFrame, rallies_df: pd.DataFrame) -> pd.DataFrame:
    shots_df = shots_df.copy()
    shots_df["rally_id"] = None
    for _, rally in rallies_df.iterrows():
        mask = (shots_df["frame"] >= rally["start_frame"]) & (shots_df["frame"] <= rally["end_frame"])
        shots_df.loc[mask, "rally_id"] = int(rally["rally_id"])
    return shots_df


class ReportGenerator:
    def generate(self, job_dir: Path) -> dict[str, Any]:
        artifacts = ArtifactStore(job_dir)

        report = {}

        court_analytics = artifacts.get("court_analytics")
        if court_analytics:
            report["court_analytics"] = court_analytics

        footwork = artifacts.get("footwork_analytics")
        if footwork:
            report["footwork"] = footwork

        fitness = artifacts.get("fitness_analytics")
        if fitness:
            report["fitness"] = fitness

        tactical = artifacts.get("tactical_analytics")
        if tactical:
            report["tactical"] = tactical
            for player_id, data in tactical.items():
                report.setdefault("shot_distribution", {}).update(data.get("shot_distribution", {}))

        technical = artifacts.get("technical_analytics")
        if technical:
            report["technical"] = technical

        coach = artifacts.get("report")
        if coach:
            report.update(coach)

        # Prefer structured quality.json from DataQualityStage
        quality_from_stage = artifacts.get("data_quality")
        if quality_from_stage and quality_from_stage.get("quality_score") is not None:
            report["data_quality"] = quality_from_stage
        else:
            data_quality = self._generate_data_quality_report(artifacts)
            report["data_quality"] = data_quality

        # Add per-stage timings
        stage_timings = artifacts.get("stage_timings")
        if stage_timings:
            report["stage_timings"] = stage_timings

        rallies_df = artifacts.get_parquet("rallies")
        if rallies_df is not None:
            report["rallies"] = rallies_df.to_dict(orient="records")

        shots_df = artifacts.get_parquet("shots")
        if shots_df is not None and rallies_df is not None:
            shots_df = _assign_rally_ids(shots_df, rallies_df)
            artifacts.set_parquet("shots", shots_df)
            report["shots"] = shots_df.to_dict(orient="records")
            report["shot_count"] = len(shots_df)
        elif shots_df is not None:
            report["shot_count"] = len(shots_df)

        report_path = job_dir / "report.json"
        report_path.write_text(json.dumps(_clean_nan(report), indent=2, default=str))

        return report

    def _generate_data_quality_report(self, artifacts: ArtifactStore) -> dict:
        """Generate data quality report for the pipeline run."""
        quality_flags = []
        
        # Check for synthetic detections
        players_data = artifacts.get("players") or {}
        synthetic_players = [p for p in players_data.get("players", []) if p.get("is_synthetic")]
        if synthetic_players:
            all_synthetic = len(synthetic_players) == len(players_data.get("players", []))
            quality_flags.append({
                "type": "synthetic_data",
                "severity": "high" if all_synthetic else "medium",
                "message": (
                    "All player detections were synthetically generated (YOLO failed to detect players)"
                    if all_synthetic else
                    f"{len(synthetic_players)} player(s) have synthetic detections mixed in"
                ),
                "all_synthetic": all_synthetic,
            })
        
        # Check for rule-based stroke classification
        shots_df = artifacts.get_parquet("shots")
        if shots_df is not None and len(shots_df) > 0:
            if 'is_rule_based' in shots_df.columns:
                rule_based_shots = shots_df[shots_df['is_rule_based'] == True]
                if len(rule_based_shots) > 0:
                    pct = len(rule_based_shots) / len(shots_df) * 100
                    quality_flags.append({
                        "type": "rule_based_classification",
                        "severity": "high" if pct > 50 else "medium",
                        "message": f"{len(rule_based_shots)}/{len(shots_df)} ({pct:.0f}%) strokes used rule-based fallback (BST model unavailable or returned unknown)",
                        "pct_rule_based": round(pct, 1),
                    })
        
        # Check for BST model availability
        bst_path = artifacts.get("bst_model_path")
        if bst_path is None:
            quality_flags.append({
                "type": "bst_model_missing",
                "severity": "medium",
                "message": "BST model checkpoint not found — all strokes may use rule-based fallback",
            })
        
        # Check court validity
        court = artifacts.get("court") or {}
        if not court.get("valid", True):
            quality_flags.append({
                "type": "invalid_court",
                "severity": "high",
                "message": "Court detection failed — analytics depending on homography may be inaccurate",
            })
        
        # Compute a meaningful 0-100 score
        base_score = 100
        for flag in quality_flags:
            if flag["severity"] == "high":
                base_score -= 30
            elif flag["severity"] == "medium":
                base_score -= 15
            elif flag["severity"] == "low":
                base_score -= 5
        data_quality_score = max(0, base_score)

        return {
            "data_quality_score": data_quality_score,
            "flags": quality_flags,
            "warnings": [flag["message"] for flag in quality_flags]
        }