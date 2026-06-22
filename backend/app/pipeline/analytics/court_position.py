import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court
from app.pipeline.shared.logging import logger

ZONE_NAMES = [
    "front_left", "front_center", "front_right",
    "mid_left", "mid_center", "mid_right",
    "rear_left", "rear_center", "rear_right",
]


class CourtPositionAnalyticsStage:
    name = "court_position_analytics"
    input_keys = ["court", "shots", "shuttle"]
    output_keys = ["court_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        court = artifacts.get("court")
        if court is None:
            return StageResult.from_error("Court data required")
        
        # Check if court is valid
        if not court.get("valid", False):
            return StageResult.from_error("Court detection is invalid, cannot compute court position analytics")

        court_length = court.get("court_length", COURT_LENGTH)
        court_width = court.get("court_width", COURT_WIDTH)

        shuttle_df = artifacts.get_parquet("shuttle")
        shots_df = artifacts.get_parquet("shots")

        vid_w, vid_h = 1280, 720
        video_res = artifacts.get("video_resolution")
        if video_res:
            vid_w = float(video_res.get("width", vid_w))
            vid_h = float(video_res.get("height", vid_h))
        elif shuttle_df is not None and len(shuttle_df) > 0:
            vid_w = max(float(shuttle_df["x"].max()), 640)
            vid_h = max(float(shuttle_df["y"].max()), 480)

        zone_transitions = []
        homography = court.get("homography")
        
        if shuttle_df is not None and shots_df is not None:
            for _, shot in shots_df.iterrows():
                frame = int(shot["frame"])
                shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
                if len(shuttle_row) > 0:
                    x = float(shuttle_row.iloc[0]["x"])
                    y = float(shuttle_row.iloc[0]["y"])
                    
                    # Convert pixel coordinates to court coordinates using homography
                    if homography is not None:
                        H = np.array(homography)
                        court_x, court_y = image_to_court(H, (x, y))
                        
                        # Use court coordinates for zone calculation
                        zone = self._get_zone_from_court(court_x, court_y, court_length, court_width)
                    else:
                        # Fallback to pixel-based zone calculation
                        zone = self._get_zone(x, y, vid_w, vid_h)
                    
                    zone_transitions.append({
                        "frame": frame,
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

        logger.info(f"Computed court position analytics: {len(zone_transitions)} zone transitions across {len(set(t['player_id'] for t in zone_transitions))} players")

        return StageResult.success(
            artifacts={"court_analytics": artifacts.path("court_analytics")},
            metadata={
                "zone_transitions": len(zone_transitions),
                "zones_used": list(set(t["zone"] for t in zone_transitions)),
            }
        )

    @staticmethod
    def _get_zone(x: float, y: float, width: float, height: float) -> str:
        col = min(int(x / width * 3), 2)
        row = min(int(y / height * 3), 2)
        return ZONE_NAMES[row * 3 + col]

    @staticmethod
    def _get_zone_from_court(court_x: float, court_y: float, court_length: float, court_width: float) -> str:
        # Convert court coordinates to zone indices
        col = min(int(court_x / court_length * 3), 2)
        row = min(int(court_y / court_width * 3), 2)
        return ZONE_NAMES[row * 3 + col]
