from dataclasses import dataclass, field

import pandas as pd

from app.pipeline.shared.ownership_quality import confident_owner_shots


@dataclass
class MatchModel:
    match_id: str
    rallies: pd.DataFrame
    shots: pd.DataFrame
    hits: pd.DataFrame
    shuttle: pd.DataFrame
    positions: pd.DataFrame
    pose: pd.DataFrame | None
    owner_shots: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame], match_id: str = "") -> "MatchModel":
        shots = tables.get("shots", pd.DataFrame())
        owner_shots = confident_owner_shots(shots)
        if "player_id" in owner_shots.columns and len(owner_shots) > 0:
            player_ids = sorted(pid for pid in owner_shots["player_id"].unique().tolist() if pd.notna(pid))
        else:
            player_ids = []

        return cls(
            match_id=match_id,
            rallies=tables.get("rallies", pd.DataFrame()),
            shots=shots,
            hits=tables.get("hits", pd.DataFrame()),
            shuttle=tables.get("shuttle", pd.DataFrame()),
            positions=tables.get("player_detections", pd.DataFrame()),
            pose=tables.get("pose"),
            owner_shots=owner_shots,
            player_ids=player_ids,
        )

    def shots_of(self, player_id: str) -> pd.DataFrame:
        if "player_id" not in self.owner_shots.columns:
            return pd.DataFrame()
        return self.owner_shots[self.owner_shots["player_id"] == player_id]

    def positions_of(self, player_id: str) -> pd.DataFrame:
        if "player_id" not in self.positions.columns:
            return pd.DataFrame()
        return self.positions[self.positions["player_id"] == player_id]
