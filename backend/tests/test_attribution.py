import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.attribution import PlayerAttributionStage


def test_attribution_assigns_player_to_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "stroke_type": ["clear", "smash", "drop"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "x": [200, 400, 300],
        "y": [300, 200, 250],
        "confidence": [0.95, 0.92, 0.88],
    })
    store.set_parquet("shuttle", shuttle_df)

    players_data = {
        "players": [
            {"id": "player_1", "side": "near"},
            {"id": "player_2", "side": "far"},
        ]
    }
    store.set("players", players_data)

    stage = PlayerAttributionStage()
    result = stage.run(store, config)

    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert "player_id" in shots_df.columns
    assert shots_df["player_id"].notna().all()
