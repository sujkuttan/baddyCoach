import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PlayerAttributionStage:
    name = "player_attribution"
    input_keys = ["shots", "shuttle", "players"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"attributed": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        players_data = artifacts.get("players")

        if players_data is None:
            return StageResult.from_error("Player data required for attribution")

        players = {p["id"]: p for p in players_data["players"]}

        player_ids = list(players.keys())
        attributed = []

        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            player_id = self._assign_player(frame, shuttle_df, players)
            attributed.append(player_id)

        shots_df["player_id"] = attributed
        artifacts.set_parquet("shots", shots_df)

        counts = shots_df["player_id"].value_counts().to_dict()
        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"attributed": len(shots_df), "distribution": counts}
        )

    def _assign_player(self, frame: int, shuttle_df: pd.DataFrame | None, players: dict) -> str:
        if shuttle_df is None or len(players) == 0:
            return list(players.keys())[0] if players else "unknown"

        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        if len(shuttle_row) == 0:
            return list(players.keys())[0]

        shuttle_y = float(shuttle_row.iloc[0]["y"])

        player_list = list(players.values())
        if len(player_list) == 2:
            sides = [p["side"] for p in player_list]
            if shuttle_y > 300 and "near" in sides:
                return next(p["id"] for p in player_list if p["side"] == "near")
            elif shuttle_y <= 300 and "far" in sides:
                return next(p["id"] for p in player_list if p["side"] == "far")

        return player_list[0]["id"]
