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
    assert "owner_confident" in shots_df.columns
    assert "owner_source" in shots_df.columns
    assert set(shots_df["side"]).issubset({"near", "far", "unknown"})


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
    assert "owner_confident" in shots_df.columns
    assert set(shots_df["side"]).issubset({"near", "far", "unknown"})


def test_bst_alpha_attribution_respects_alpha(tmp_job_dir):
    """Verify BST aimplayer_alpha and class_id feed diagnostic BST fields only."""
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
    assert "ownership_bst_diag_near" in shots_df.columns
    assert "ownership_bst_diag_far" in shots_df.columns

    assert shots_df.loc[0, "ownership_bst_diag_near"] == pytest.approx(0.2, abs=0.01)
    assert shots_df.loc[0, "ownership_bst_diag_far"] == pytest.approx(0.8, abs=0.01)
    assert shots_df.loc[1, "ownership_bst_diag_near"] == pytest.approx(0.2, abs=0.01)
    assert shots_df.loc[1, "ownership_bst_diag_far"] == pytest.approx(0.8, abs=0.01)
    assert shots_df.loc[2, "ownership_bst_diag_near"] == pytest.approx(0.75, abs=0.01)
    assert shots_df.loc[2, "ownership_bst_diag_far"] == pytest.approx(0.25, abs=0.01)
    assert shots_df.loc[3, "ownership_bst_diag_near"] == pytest.approx(0.8, abs=0.01)
    assert shots_df.loc[3, "ownership_bst_diag_far"] == pytest.approx(0.2, abs=0.01)


def test_attention_owner_match_alpha_far(tmp_job_dir):
    """Alpha diagnostics are retained even when the final owner abstains."""
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
    assert "ownership_bst_diag_near" in shots_df.columns
    assert shots_df.loc[0, "ownership_bst_diag_near"] == pytest.approx(0.3, abs=0.02)
    assert shots_df.loc[0, "ownership_bst_diag_far"] == pytest.approx(0.7, abs=0.02)
    assert shots_df.loc[0, "attention_alpha_owner"] is None
    assert shots_df.loc[0, "attention_owner_match"] is None


def test_attention_owner_match_alpha_near(tmp_job_dir):
    """Alpha < 0.5 updates diagnostics but not owner matching without an assignment."""
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
    assert shots_df.loc[0, "ownership_bst_diag_near"] == pytest.approx(0.75, abs=0.02)
    assert shots_df.loc[0, "ownership_bst_diag_far"] == pytest.approx(0.25, abs=0.02)
    assert shots_df.loc[0, "attention_alpha_owner"] is None


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


def test_attention_owner_match_requires_reliable_alpha(tmp_job_dir, monkeypatch):
    from app.pipeline.shared.ownership_quality import OwnerDecision

    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {"valid": True, "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)]})
    store.set_parquet(
        "shots",
        pd.DataFrame(
            {
                "frame": [0],
                "rally_id": [1],
                "stroke_type": ["clear"],
                "stroke_confidence": [0.9],
                "aimplayer_alpha": [0.85],
                "aim_alpha_reliable": [False],
            }
        ),
    )
    store.set_parquet("rallies", pd.DataFrame({"rally_id": [1], "start_frame": [0], "end_frame": [0]}))
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0], "x": [200], "y": [300], "confidence": [0.95]}))
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})

    monkeypatch.setattr(
        "app.pipeline.attribution.assign_rally_owners",
        lambda indices, scores, players_by_side, settings: {
            indices[0]: OwnerDecision(side="far", player_id="player_2", confident=True, source="local_anchor", reason="test")
        },
    )

    result = PlayerAttributionStage().run(store, config)
    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert shots_df.loc[0, "attention_alpha_owner"] is None
    assert shots_df.loc[0, "attention_owner_match"] is None


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


def test_unanchored_rally_stays_unknown(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    store.set("court", {"valid": False})
    store.set_parquet("rallies", pd.DataFrame({"rally_id": [1], "start_frame": [0], "end_frame": [20]}))
    store.set_parquet(
        "shots",
        pd.DataFrame(
            {
                "frame": [0, 10, 20],
                "rally_id": [1, 1, 1],
                "stroke_type": ["clear", "clear", "clear"],
                "stroke_confidence": [0.8, 0.8, 0.8],
            }
        ),
    )
    store.set_parquet(
        "shuttle",
        pd.DataFrame(
            {
                "frame": [0, 10, 20],
                "x": [100.0, 100.0, 100.0],
                "y": [200.0, 200.0, 200.0],
                "confidence": [0.1, 0.1, 0.1],
            }
        ),
    )

    PlayerAttributionStage().run(store, StageConfig())
    shots = store.get_parquet("shots")

    assert shots["player_id"].isna().all()
    assert set(shots["side"]) == {"unknown"}
    assert shots["owner_confident"].eq(False).all()
    assert set(shots["owner_source"]) == {"unknown"}


def test_short_compatible_gap_bridges_between_anchors(tmp_job_dir, monkeypatch):
    store = ArtifactStore(tmp_job_dir)
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    store.set("court", {"valid": False})
    store.set_parquet("rallies", pd.DataFrame({"rally_id": [1], "start_frame": [0], "end_frame": [30]}))
    store.set_parquet(
        "shots",
        pd.DataFrame(
            {
                "frame": [0, 10, 20, 30],
                "rally_id": [1, 1, 1, 1],
                "stroke_type": ["clear", "clear", "clear", "clear"],
                "stroke_confidence": [0.8, 0.8, 0.8, 0.8],
            }
        ),
    )
    store.set_parquet(
        "shuttle",
        pd.DataFrame(
            {
                "frame": [0, 10, 20, 30],
                "x": [0.0, 0.0, 0.0, 0.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "confidence": [0.9, 0.9, 0.9, 0.9],
            }
        ),
    )

    scripted = iter(
        [
            {"near_score": 0.82, "far_score": 0.18, "trajectory_near": 0.9, "trajectory_far": 0.1, "court_side_near": 0.8, "court_side_far": 0.2, "proximity_near": 0.8, "proximity_far": 0.2, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.4, "bst_diag_far": 0.6},
            {"near_score": 0.52, "far_score": 0.48, "trajectory_near": 0.5, "trajectory_far": 0.5, "court_side_near": 0.5, "court_side_far": 0.5, "proximity_near": 0.5, "proximity_far": 0.5, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.5, "bst_diag_far": 0.5},
            {"near_score": 0.49, "far_score": 0.51, "trajectory_near": 0.5, "trajectory_far": 0.5, "court_side_near": 0.5, "court_side_far": 0.5, "proximity_near": 0.5, "proximity_far": 0.5, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.5, "bst_diag_far": 0.5},
            {"near_score": 0.19, "far_score": 0.81, "trajectory_near": 0.2, "trajectory_far": 0.8, "court_side_near": 0.2, "court_side_far": 0.8, "proximity_near": 0.2, "proximity_far": 0.8, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.6, "bst_diag_far": 0.4},
        ]
    )
    monkeypatch.setattr(OwnershipScorer, "score", lambda self, **kwargs: next(scripted))

    PlayerAttributionStage().run(store, StageConfig())
    shots = store.get_parquet("shots")

    assert shots["side"].tolist() == ["near", "far", "near", "far"]
    assert shots["owner_source"].tolist() == ["local_anchor", "viterbi_bridge", "viterbi_bridge", "local_anchor"]
    assert shots["owner_confident"].tolist() == [True, True, True, True]
