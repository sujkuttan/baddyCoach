from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MatchModel:
    match_id: str
    rallies: pd.DataFrame
    shots: pd.DataFrame
    hits: pd.DataFrame
    shuttle: pd.DataFrame
    positions: pd.DataFrame
    pose: pd.DataFrame | None
    player_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame], match_id: str = "") -> "MatchModel":
        shots = tables.get("shots", pd.DataFrame())
        player_ids = sorted(shots["player_id"].unique().tolist()) if "player_id" in shots.columns and len(shots) > 0 else []

        return cls(
            match_id=match_id,
            rallies=tables.get("rallies", pd.DataFrame()),
            shots=shots,
            hits=tables.get("hits", pd.DataFrame()),
            shuttle=tables.get("shuttle", pd.DataFrame()),
            positions=tables.get("player_detections", pd.DataFrame()),
            pose=tables.get("pose"),
            player_ids=player_ids,
        )

    def shots_of(self, player_id: str) -> pd.DataFrame:
        if "player_id" not in self.shots.columns:
            return pd.DataFrame()
        return self.shots[self.shots["player_id"] == player_id]

    def positions_of(self, player_id: str) -> pd.DataFrame:
        if "player_id" not in self.positions.columns:
            return pd.DataFrame()
        return self.positions[self.positions["player_id"] == player_id]
