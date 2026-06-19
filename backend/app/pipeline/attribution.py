import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PlayerAttributionStage:
    name = "player_attribution"
    input_keys = ["shots", "shuttle", "players", "court"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"attributed": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        players_data = artifacts.get("players")
        court = artifacts.get("court") or {}

        if players_data is None:
            return StageResult.from_error("Player data required for attribution")

        players = {p["id"]: p for p in players_data["players"]}

        court_corners = court.get("corners_pixel", [])
        if court_corners and len(court_corners) >= 3:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = 360

        player_ids = list(players.keys())
        attributed = []

        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            player_id = self._assign_player(frame, shuttle_df, players, court_mid_y)
            attributed.append(player_id)

        shots_df["player_id"] = attributed
        artifacts.set_parquet("shots", shots_df)

        counts = shots_df["player_id"].value_counts().to_dict()
        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"attributed": len(shots_df), "distribution": counts, "court_mid_y": court_mid_y}
        )

    def _assign_player(self, frame: int, shuttle_df: pd.DataFrame | None, players: dict, court_mid_y: float) -> str:
        if shuttle_df is None or len(players) == 0:
            return list(players.keys())[0] if players else "unknown"

        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        if len(shuttle_row) == 0:
            return list(players.keys())[0]

        shuttle_y = float(shuttle_row.iloc[0]["y"])

        player_list = list(players.values())
        if len(player_list) == 2:
            sides = [p["side"] for p in player_list]
            if shuttle_y > court_mid_y and "near" in sides:
                return next(p["id"] for p in player_list if p["side"] == "near")
            elif shuttle_y <= court_mid_y and "far" in sides:
                return next(p["id"] for p in player_list if p["side"] == "far")

        return player_list[0]["id"]
