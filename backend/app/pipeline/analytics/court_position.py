import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH
from app.pipeline.shared.logging import logger

ZONE_NAMES = [
    "front_left", "front_center", "front_right",
    "mid_left", "mid_center", "mid_right",
    "rear_left", "rear_center", "rear_right",
]


class CourtPositionAnalyticsStage:
    name = "court_position_analytics"
    input_keys = ["court", "shots"]
    output_keys = ["court_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        court = artifacts.get("court")
        if court is None:
            return StageResult.from_error("Court data required")

        if not court.get("valid", False):
            return StageResult.from_error("Court detection is invalid, cannot compute court position analytics")

        court_length = court.get("court_length", COURT_LENGTH)
        court_width = court.get("court_width", COURT_WIDTH)

        shots_df = artifacts.get_parquet("shots")

        zone_transitions = []

        if shots_df is not None and len(shots_df) > 0:
            for _, shot in shots_df.iterrows():
                court_x = shot.get("court_x")
                court_y = shot.get("court_y")
                if pd.isna(court_x) or pd.isna(court_y):
                    continue
                zone = self._get_zone_from_court(
                    float(court_x), float(court_y),
                    court_length, court_width,
                )
                zone_transitions.append({
                    "frame": int(shot["frame"]),
                    "zone": zone,
                    "player_id": shot.get("player_id", "unknown"),
                })

        analytics_data = {
            "zone_transitions": zone_transitions,
            "court_dimensions": {
                "length": court_length,
                "width": court_width,
            },
        }

        artifacts.set("court_analytics", analytics_data)

        n_players = len(set(t["player_id"] for t in zone_transitions))
        logger.info(f"Computed court position analytics: {len(zone_transitions)} zone transitions across {n_players} players")

        return StageResult.success(
            artifacts={"court_analytics": artifacts.path("court_analytics")},
            metadata={
                "zone_transitions": len(zone_transitions),
                "zones_used": list(set(t["zone"] for t in zone_transitions)),
            }
        )

    @staticmethod
    def _get_zone_from_court(court_x: float, court_y: float, court_length: float, court_width: float) -> str:
        col = min(int(court_x / court_length * 3), 2)
        row = min(int(court_y / court_width * 3), 2)
        return ZONE_NAMES[row * 3 + col]
