import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import PlayerAttributionStage


def test_attribution_assigns_player_to_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
        "court_length": 13.4,
        "court_width": 6.10,
    }
    store.set("court", court_data)

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


def test_bst_alpha_attribution_respects_alpha(tmp_job_dir):
    """Verify AimPlayer alpha drives player assignment in BST Top/Bottom attribution."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
        "court_length": 13.4,
        "court_width": 6.10,
    }
    store.set("court", court_data)

    # Shots with shuttleset_class_id but varying alpha values
    shots_df = pd.DataFrame({
        "frame": [0, 10, 20, 30],
        "stroke_type": ["smash", "clear", "unknown", "lift"],
        "stroke_confidence": [0.6, 0.4, 0.2, 0.7],
        "shuttleset_class_id": [3, 5, 0, 15],
        "aimplayer_alpha": [0.85, 0.55, 0.12, 0.72],
    })
    store.set_parquet("shots", shots_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20, 30],
        "x": [200, 400, 300, 500],
        "y": [300, 200, 250, 350],
        "confidence": [0.95, 0.92, 0.88, 0.9],
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

    # alpha=0.85 (>0.5+0.15) → far → player_2
    assert shots_df.loc[0, "player_id"] == "player_2"
    # alpha=0.55 (within 0.15 of 0.5) → uncertain → falls through to class_id for Top_smash (conf=0.6 ≥ 0.3) → far → player_2
    assert shots_df.loc[1, "player_id"] == "player_2"
    # alpha=0.12 (<0.5-0.15) → near → player_1, even with class_id=0
    assert shots_df.loc[2, "player_id"] == "player_1"
    # alpha=0.72 (>0.5+0.15) → far → player_2, class_id=15 (Bottom_lift) also agrees
    assert shots_df.loc[3, "player_id"] == "player_2"


def test_attention_owner_match_alpha_far(tmp_job_dir):
    """Alpha > 0.5, side set by Tier 1 alpha → attention_owner_match=True."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {"valid": True, "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)]})
    shots_df = pd.DataFrame({
        "frame": [0],
        "stroke_type": ["clear"],
        "stroke_confidence": [0.9],
        "shuttleset_class_id": [0],
        "aimplayer_alpha": [0.85],
    })
    store.set_parquet("shots", shots_df)
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0], "x": [200], "y": [300], "confidence": [0.95]}))
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    stage = PlayerAttributionStage()
    result = stage.run(store, config)
    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert shots_df.loc[0, "player_id"] == "player_2"
    assert shots_df.loc[0, "attention_alpha_owner"] == "far"
    assert shots_df.loc[0, "attention_owner_match"] == True


def test_attention_owner_match_alpha_near(tmp_job_dir):
    """Alpha < 0.5, side set by Tier 1 alpha → attention_owner_match=True."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {"valid": True, "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)]})
    shots_df = pd.DataFrame({
        "frame": [0],
        "stroke_type": ["clear"],
        "stroke_confidence": [0.9],
        "shuttleset_class_id": [0],
        "aimplayer_alpha": [0.12],
    })
    store.set_parquet("shots", shots_df)
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0], "x": [200], "y": [300], "confidence": [0.95]}))
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    stage = PlayerAttributionStage()
    result = stage.run(store, config)
    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert shots_df.loc[0, "player_id"] == "player_1"
    assert shots_df.loc[0, "attention_alpha_owner"] == "near"
    assert shots_df.loc[0, "attention_owner_match"] == True


def test_attention_owner_match_alpha_ambiguous(tmp_job_dir):
    """Alpha == 0.5 exactly → alpha_owner=None → match=None."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {"valid": True, "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)]})
    shots_df = pd.DataFrame({
        "frame": [0],
        "stroke_type": ["clear"],
        "stroke_confidence": [0.9],
        "shuttleset_class_id": [0],
        "aimplayer_alpha": [0.5],
    })
    store.set_parquet("shots", shots_df)
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0], "x": [200], "y": [300], "confidence": [0.95]}))
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    stage = PlayerAttributionStage()
    result = stage.run(store, config)
    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert shots_df.loc[0, "attention_alpha_owner"] is None
    assert shots_df.loc[0, "attention_owner_match"] is None


def test_attention_owner_match_no_alpha(tmp_job_dir):
    """No aimplayer_alpha column → match=None for all shots."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {"valid": True, "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)]})
    shots_df = pd.DataFrame({
        "frame": [0, 10],
        "stroke_type": ["clear", "smash"],
        "stroke_confidence": [0.9, 0.85],
    })
    store.set_parquet("shots", shots_df)
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0, 10], "x": [200, 400], "y": [300, 200], "confidence": [0.95, 0.92]}))
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    stage = PlayerAttributionStage()
    result = stage.run(store, config)
    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert "attention_owner_match" in shots_df.columns
    assert shots_df["attention_owner_match"].isna().all()
    assert shots_df["attention_alpha_owner"].isna().all()
