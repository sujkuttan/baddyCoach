"""Tests for Task 2.1: turn-prior & BST AimPlayer alpha wired into ownership score."""

import numpy as np
import pandas as pd
import pytest

from app.pipeline.shared.ownership_scorer import OwnershipScorer


def _build_inputs():
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
    return shuttle_df, players, court


def test_turn_prior_far_wins_with_strong_far_turn():
    scorer = OwnershipScorer(
        trajectory_weight=0.0, court_side_weight=0.0, proximity_weight=0.0,
        motion_weight=0.0, pose_feasibility_weight=0.0,
        turn_prior_weight=1.0, bst_weight=0.0,
        calib_near_mean=0.5, calib_near_std=1.0, calib_far_mean=0.5, calib_far_std=1.0,
    )
    shuttle_df, players, court = _build_inputs()
    # previous owner was near -> far player should hit next
    res = scorer.score(shuttle_df, None, players, court, frame=10,
                       prev_owner="p1", shot={})
    assert res["far_score"] > res["near_score"]
    assert res["turn_far"] > res["turn_near"]


def test_turn_prior_near_wins_with_strong_near_turn():
    scorer = OwnershipScorer(
        trajectory_weight=0.0, court_side_weight=0.0, proximity_weight=0.0,
        motion_weight=0.0, pose_feasibility_weight=0.0,
        turn_prior_weight=1.0, bst_weight=0.0,
        calib_near_mean=0.5, calib_near_std=1.0, calib_far_mean=0.5, calib_far_std=1.0,
    )
    shuttle_df, players, court = _build_inputs()
    # previous owner was far -> near player should hit next
    res = scorer.score(shuttle_df, None, players, court, frame=10,
                       prev_owner="p2", shot={})
    assert res["near_score"] > res["far_score"]
    assert res["turn_near"] > res["turn_far"]


def test_bst_far_alpha_favors_far():
    scorer = OwnershipScorer(
        trajectory_weight=0.0, court_side_weight=0.0, proximity_weight=0.0,
        motion_weight=0.0, pose_feasibility_weight=0.0,
        turn_prior_weight=0.0, bst_weight=1.0,
        calib_near_mean=0.5, calib_near_std=1.0, calib_far_mean=0.5, calib_far_std=1.0,
    )
    shuttle_df, players, court = _build_inputs()
    res = scorer.score(shuttle_df, None, players, court, frame=10,
                       prev_owner=None, shot={"aimplayer_alpha": 0.90})
    assert res["far_score"] > res["near_score"]
    assert res["bst_diag_far"] > res["bst_diag_near"]


def test_bst_near_alpha_favors_near():
    scorer = OwnershipScorer(
        trajectory_weight=0.0, court_side_weight=0.0, proximity_weight=0.0,
        motion_weight=0.0, pose_feasibility_weight=0.0,
        turn_prior_weight=0.0, bst_weight=1.0,
        calib_near_mean=0.5, calib_near_std=1.0, calib_far_mean=0.5, calib_far_std=1.0,
    )
    shuttle_df, players, court = _build_inputs()
    res = scorer.score(shuttle_df, None, players, court, frame=10,
                       prev_owner=None, shot={"aimplayer_alpha": 0.10})
    assert res["near_score"] > res["far_score"]


def test_five_signal_formula_unchanged_when_turn_bst_zeroed():
    """With turn/bst weights at 0, the calibrated score must equal the
    deterministic 5-signal weighted combination (i.e. the pre-2.1 math)."""
    w = dict(trajectory_weight=0.35, court_side_weight=0.20, proximity_weight=0.15,
             motion_weight=0.15, pose_feasibility_weight=0.10,
             turn_prior_weight=0.0, bst_weight=0.0,
             calib_near_mean=0.62, calib_near_std=0.14,
             calib_far_mean=0.48, calib_far_std=0.18)

    scorer = OwnershipScorer(**w)
    shuttle_df, players, court = _build_inputs()
    res = scorer.score(shuttle_df, None, players, court, frame=10,
                       prev_owner="p1", shot={"aimplayer_alpha": 0.90})

    raw_weights = {"trajectory": 0.35, "court_side": 0.20, "proximity": 0.15,
                   "motion": 0.15, "pose": 0.10}
    total = sum(raw_weights.values())
    raw_near = sum(v / total * res[f"{k}_near"] for k, v in raw_weights.items())
    raw_far = sum(v / total * res[f"{k}_far"] for k, v in raw_weights.items())

    z_n = (raw_near - w["calib_near_mean"]) / w["calib_near_std"]
    z_f = (raw_far - w["calib_far_mean"]) / w["calib_far_std"]
    p_n = 1.0 / (1.0 + np.exp(-z_n))
    p_f = 1.0 / (1.0 + np.exp(-z_f))
    expected_near = p_n / (p_n + p_f)
    expected_far = p_f / (p_n + p_f)

    assert res["near_score"] == pytest.approx(expected_near, abs=1e-4)
    assert res["far_score"] == pytest.approx(expected_far, abs=1e-4)


def test_defaults_use_raised_weights():
    scorer = OwnershipScorer()
    assert scorer.turn_prior_weight == 0.25
    assert scorer.bst_weight == 0.15
