import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.storage.artifacts import ArtifactStore


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
        report_path.write_text(json.dumps(report, indent=2, default=str))

        return report