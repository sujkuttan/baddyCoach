import numpy as np
import pandas as pd
import pytest
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import PlayerAttributionStage
from app.pipeline.shared.ownership_scorer import OwnershipScorer


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


def test_attribution_continues_with_invalid_court_geometry(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {
        "valid": False,
        "corners_pixel": [(100, 500), (1180, 500), (100, 150), (1180, 150)],
        "court_length": 13.4,
        "court_width": 6.10,
    })
    store.set_parquet("shots", pd.DataFrame({
        "frame": [0, 10],
        "stroke_type": ["clear", "smash"],
        "stroke_confidence": [0.9, 0.85],
    }))
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": [0, 10],
        "x": [200, 400],
        "y": [300, 200],
        "confidence": [0.95, 0.92],
    }))
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})

    result = PlayerAttributionStage().run(store, config)

    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert "player_id" in shots_df.columns
    assert shots_df["player_id"].notna().all()


def test_bst_alpha_attribution_respects_alpha(tmp_job_dir):
    """Verify BST aimplayer_alpha and class_id feed into bst_near/bst_far ownership sub-scores."""
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
        "frame": [0, 10, 20, 30],
        "stroke_type": ["smash", "clear", "unknown", "lift"],
        "stroke_confidence": [0.6, 0.4, 0.2, 0.7],
        "shuttleset_class_id": [3, 5, 0, 15],
        "aimplayer_alpha": [0.85, 0.55, 0.12, 0.72],
    })
    store.set_parquet("shots", shots_df)
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": [0, 10, 20, 30],
        "x": [200, 400, 300, 500],
        "y": [300, 200, 250, 350],
        "confidence": [0.95, 0.92, 0.88, 0.9],
    }))
    store.set("players", {"players": [
        {"id": "player_1", "side": "near"},
        {"id": "player_2", "side": "far"},
    ]})

    stage = PlayerAttributionStage()
    result = stage.run(store, config)
    assert result.status == "success"

    shots_df = store.get_parquet("shots")
    assert "player_id" in shots_df.columns
    assert shots_df["player_id"].notna().all()
    assert "ownership_bst_near" in shots_df.columns
    assert "ownership_bst_far" in shots_df.columns

    # Shot 0: alpha=0.85 (>0.5+0.15) → far; class_id=3 (Top_smash) → overrides to (0.2, 0.8)
    assert shots_df.loc[0, "ownership_bst_near"] == pytest.approx(0.2, abs=0.01)
    assert shots_df.loc[0, "ownership_bst_far"] == pytest.approx(0.8, abs=0.01)
    # Shot 1: alpha=0.55 (within 0.15 threshold) → neutral alpha; class_id=5 (Top_clear, conf=0.4≥0.3) → (0.2, 0.8)
    assert shots_df.loc[1, "ownership_bst_near"] == pytest.approx(0.2, abs=0.01)
    assert shots_df.loc[1, "ownership_bst_far"] == pytest.approx(0.8, abs=0.01)
    # Shot 2: alpha=0.12 (<0.5-0.15) → strong near (0.75); class_id=0 → no override
    assert shots_df.loc[2, "ownership_bst_near"] == pytest.approx(0.75, abs=0.01)
    assert shots_df.loc[2, "ownership_bst_far"] == pytest.approx(0.25, abs=0.01)
    # Shot 3: alpha=0.72 (>0.5+0.15) → moderate far (0.54); class_id=15 (Bottom_lift, conf=0.7≥0.3) → overrides to (0.8, 0.2)
    assert shots_df.loc[3, "ownership_bst_near"] == pytest.approx(0.8, abs=0.01)
    assert shots_df.loc[3, "ownership_bst_far"] == pytest.approx(0.2, abs=0.01)


def test_attention_owner_match_alpha_far(tmp_job_dir):
    """Alpha > 0.5, Viterbi assigns far → attention_owner_match=True."""
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
    # BST sub-score: alpha=0.85 (>0.5+0.15) → far, class_id=0 → no override → (0.3, 0.7)
    assert "ownership_bst_near" in shots_df.columns
    assert shots_df.loc[0, "ownership_bst_near"] == pytest.approx(0.3, abs=0.02)
    assert shots_df.loc[0, "ownership_bst_far"] == pytest.approx(0.7, abs=0.02)
    # Post-attribution diagnostic still works
    assert shots_df.loc[0, "attention_alpha_owner"] == "far"


def test_attention_owner_match_alpha_near(tmp_job_dir):
    """Alpha < 0.5 → BST sub-score favors near; diagnostic check."""
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
    # BST sub-score: alpha=0.12 (<0.5-0.15) → strong near (0.75); class_id=0 → no override
    assert shots_df.loc[0, "ownership_bst_near"] == pytest.approx(0.75, abs=0.02)
    assert shots_df.loc[0, "ownership_bst_far"] == pytest.approx(0.25, abs=0.02)
    # Post-attribution diagnostic: alpha < 0.5 → alpha_owner = "near"
    assert shots_df.loc[0, "attention_alpha_owner"] == "near"


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


def test_bst_alpha_is_diagnostic_only_not_emission():
    scorer = OwnershipScorer(
        trajectory_weight=1.0,
        court_side_weight=0.0,
        proximity_weight=0.0,
        motion_weight=0.0,
        pose_feasibility_weight=0.0,
        turn_prior_weight=0.0,
        bst_weight=0.0,
        calib_near_mean=0.5,
        calib_near_std=1.0,
        calib_far_mean=0.5,
        calib_far_std=1.0,
    )
    shuttle_df = pd.DataFrame(
        {
            "frame": [7, 10, 13],
            "x": [640.0, 660.0, 700.0],
            "y": [300.0, 310.0, 320.0],
            "confidence": [0.9, 0.9, 0.9],
        }
    )
    players = {
        "players": [
            {"id": "p1", "side": "near", "detections": []},
            {"id": "p2", "side": "far", "detections": []},
        ]
    }
    court = {"homography": np.eye(3).tolist()}

    low_alpha = scorer.score(shuttle_df, None, players, court, frame=10, shot={"aimplayer_alpha": 0.10})
    high_alpha = scorer.score(shuttle_df, None, players, court, frame=10, shot={"aimplayer_alpha": 0.90})

    assert low_alpha["near_score"] == pytest.approx(high_alpha["near_score"])
    assert low_alpha["far_score"] == pytest.approx(high_alpha["far_score"])
    assert low_alpha["bst_diag_near"] != pytest.approx(high_alpha["bst_diag_near"])


def test_turn_prior_is_reported_but_not_used_in_local_score():
    scorer = OwnershipScorer(
        trajectory_weight=0.0,
        court_side_weight=1.0,
        proximity_weight=0.0,
        motion_weight=0.0,
        pose_feasibility_weight=0.0,
        turn_prior_weight=0.0,
        bst_weight=0.0,
        calib_near_mean=0.5,
        calib_near_std=1.0,
        calib_far_mean=0.5,
        calib_far_std=1.0,
    )
    shuttle_df = pd.DataFrame({"frame": [10], "x": [500.0], "y": [200.0], "confidence": [0.9]})
    players = {
        "players": [
            {"id": "p1", "side": "near", "detections": []},
            {"id": "p2", "side": "far", "detections": []},
        ]
    }
    court = {"homography": np.eye(3).tolist()}

    first = scorer.score(shuttle_df, None, players, court, frame=10, prev_owner=None, shot={})
    after_near = scorer.score(shuttle_df, None, players, court, frame=10, prev_owner="p1", shot={})

    assert first["near_score"] == pytest.approx(after_near["near_score"])
    assert first["far_score"] == pytest.approx(after_near["far_score"])
    assert after_near["turn_near"] != pytest.approx(after_near["turn_far"])
