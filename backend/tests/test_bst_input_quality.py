import pytest

from app.pipeline.shared.bst_input_quality import evaluate_aim_alpha_quality, evaluate_bst_clip_quality


def _provenance(**overrides):
    value = {
        "video_len": 20,
        "contact_frame_index": 10,
        "shuttle_observed": [True] * 12 + [False] * 7 + [True],
        "shuttle_repaired": [False] * 20,
        "shuttle_interpolated": [False] * 20,
        "shuttle_court_rejected": [False] * 20,
        "pose_present_far": [True] * 20,
        "pose_present_near": [True] * 20,
        "pose_keypoint_confidence_far": [0.9] * 20,
        "pose_keypoint_confidence_near": [0.9] * 20,
        "bbox_gap_far": [0] * 20,
        "bbox_gap_near": [0] * 20,
        "resolved_far_pid": ["player_2"] * 20,
        "resolved_near_pid": ["player_1"] * 20,
        "wrist_shuttle_distance_far": [0.6] * 20,
        "wrist_shuttle_distance_near": [0.2] * 20,
    }
    value.update(overrides)
    return value


def test_quality_accepts_clip_with_sufficient_observed_shuttle_and_pose():
    result = evaluate_bst_clip_quality(_provenance())

    assert result["eligible"] is True
    assert result["score"] == 1.0
    assert result["reasons"] == []
    assert result["observed_shuttle_frames"] == 13
    assert result["max_shuttle_gap_frames"] == 7


def test_quality_penalizes_single_court_rejected_point_without_hard_rejecting_clip():
    rejected = [False] * 20
    rejected[3] = True

    result = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))

    assert result["eligible"] is True
    assert result["score"] == 0.8
    assert result["reasons"] == []


def test_quality_rejects_clip_with_too_many_court_rejected_points():
    rejected = [True] * 6 + [False] * 14

    result = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))

    assert result["eligible"] is False
    assert result["score"] == 0.8
    assert result["reasons"] == ["court_rejected_shuttle"]


def test_quality_accumulates_all_failed_hard_checks_and_clamps_score():
    result = evaluate_bst_clip_quality(_provenance(
        video_len=10,
        shuttle_observed=[False] * 10,
        shuttle_repaired=[False] * 10,
        shuttle_interpolated=[True] * 10,
        shuttle_court_rejected=[True] * 3 + [False] * 7,
        pose_present_far=[False] * 10,
        pose_present_near=[False] * 10,
        pose_keypoint_confidence_far=[0.1] * 10,
        pose_keypoint_confidence_near=[0.1] * 10,
        bbox_gap_far=[11] * 10,
        bbox_gap_near=[11] * 10,
    ))

    assert result["eligible"] is False
    assert result["score"] == 0.0
    assert result["reasons"] == [
        "clip_too_short",
        "low_observed_shuttle",
        "long_shuttle_gap",
        "court_rejected_shuttle",
        "low_pose_coverage",
        "low_keypoint_confidence",
        "long_bbox_gap",
        "too_many_interpolated_shuttle",
        "low_quality_score",
    ]


def test_quality_scores_and_rejects_repaired_or_interpolated_shuttle_heavily():
    observed = [True] * 7 + [False] * 6 + [True] * 7
    result = evaluate_bst_clip_quality(_provenance(
        shuttle_observed=observed,
        shuttle_repaired=[False] * 7 + [True] * 6 + [False] * 7,
        shuttle_interpolated=[False] * 7 + [True] * 6 + [False] * 7,
    ))

    assert result["repaired_shuttle_fraction"] == 0.3
    assert result["interpolated_shuttle_fraction"] == 0.3
    assert result["eligible"] is False
    assert "too_many_interpolated_shuttle" in result["reasons"]
    assert "low_quality_score" in result["reasons"]


def test_aim_alpha_quality_accepts_balanced_contact_window():
    result = evaluate_aim_alpha_quality(_provenance())

    assert result["reliable"] is True
    assert result["contact_window_valid"] is True
    assert result["identity_stable"] is True
    assert result["reasons"] == []
    assert result["pose_balance_score"] == 1.0
    assert result["contact_separation"] == pytest.approx(0.4)


def test_aim_alpha_quality_rejects_asymmetric_pose_and_identity_instability():
    result = evaluate_aim_alpha_quality(
        _provenance(
            pose_present_far=[True] * 8 + [False] * 5 + [True] * 7,
            pose_keypoint_confidence_far=[0.9] * 8 + [0.0] * 5 + [0.9] * 7,
            resolved_far_pid=["player_2"] * 10 + ["player_3"] * 10,
            wrist_shuttle_distance_far=[0.25] * 20,
            wrist_shuttle_distance_near=[0.2] * 20,
        )
    )

    assert result["reliable"] is False
    assert "contact_pose_imbalance" in result["reasons"]
    assert "identity_unstable" in result["reasons"]
    assert "contact_separation_too_small" in result["reasons"]
