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


def test_racket_motion_score_uses_racket_when_present():
    from app.pipeline.shared.ownership_scorer import racket_motion_score
    # near has high racket-head speed around hit; far has none
    near_heads = [np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([0.0, 0.0])]
    far_heads = [np.array([0.0, 0.0])] * 3
    racket_seq = {"near": near_heads, "far": far_heads}
    n, f = racket_motion_score(racket_seq, hit_idx=1, motion_weight=0.6, dist_weight=0.4)
    assert n > f  # near moved its racket, far did not


def test_racket_motion_score_falls_back_when_none():
    from app.pipeline.shared.ownership_scorer import racket_motion_score
    n, f = racket_motion_score(None, hit_idx=1)
    # returns neutral (unknown_score=0.5) split, no error
    assert n == 0.5 and f == 0.5


def test_racket_motion_score_pose_fallback_uses_keypoints():
    """When racket data is absent (racket_head_seq=None) but keypoint sequences
    are supplied, racket_motion_score must use the pose-derived signal (the
    pre-feature behavior) instead of returning the neutral (0.5, 0.5) split."""
    from app.pipeline.shared.ownership_scorer import racket_motion_score

    # near player swings its wrist through the hit frame; far player is static.
    kps_static = np.array([[0, 0, 1.0]] * 17)  # placeholder per-joint
    # Build a moving wrist (COCO joint 10) for the near sequence only.
    def _kps(xy):
        k = np.zeros((17, 3))
        k[:, 2] = 0.9
        k[10, :2] = xy  # wrist
        return k

    near_seq = [_kps((0, 0)), _kps((20, 0)), _kps((0, 0))]
    far_seq = [kps_static, kps_static, kps_static]

    n, f = racket_motion_score(
        None, hit_idx=1,
        near_kps_list=near_seq, far_kps_list=far_seq,
        min_confidence=0.5,
    )
    # Must NOT be the neutral split, and must favor the moving near player.
    assert (n, f) != (0.5, 0.5)
    assert n > f


def test_scorer_reads_head_point_and_player_side():
    """The ownership scorer's racket-head window must read `head_point` /
    `player_side` (not the wrong `head`/`side` keys) so racket motion proxies
    through to the sub-scores."""
    scorer = OwnershipScorer(
        trajectory_weight=0.0, court_side_weight=0.0, proximity_weight=0.0,
        motion_weight=1.0, pose_feasibility_weight=0.0,
        turn_prior_weight=0.0, bst_weight=0.0,
        calib_near_mean=0.5, calib_near_std=1.0, calib_far_mean=0.5, calib_far_std=1.0,
    )
    shuttle_df, players, court = _build_inputs()
    # Near player has a racket head near the shuttle at the hit frame.
    racket_detections = [
        {"frame": 9, "player_side": "near", "head_point": (640.0, 300.0)},
        {"frame": 10, "player_side": "near", "head_point": (660.0, 310.0)},
        {"frame": 11, "player_side": "near", "head_point": (700.0, 320.0)},
    ]
    res = scorer.score(shuttle_df, None, players, court, frame=10,
                      prev_owner=None, shot={},
                      racket_detections=racket_detections)
    # Racket motion score is non-neutral (racket data was consumed).
    assert res["motion_far"] != res["motion_near"]

