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


def test_quality_hard_rejects_degenerate_joints_that_slip_past_other_gates():
    # A collapsed/NaN skeleton: pose_present is True every frame, keypoint
    # confidence is high, and joint_abs_mean is ~0 (not extreme), so the
    # existing coverage / confidence / extreme_joint_mean gates would ALL pass.
    # Only the degenerate-joints fraction catches it and routes to abstention.
    good = _provenance(
        joint_abs_mean=0.0,
        joint_degenerate_fraction=0.9,
        pose_present_far=[True] * 20,
        pose_present_near=[True] * 20,
        pose_keypoint_confidence_far=[0.9] * 20,
        pose_keypoint_confidence_near=[0.9] * 20,
    )
    result = evaluate_bst_clip_quality(good)

    assert result["eligible"] is False
    assert "degenerate_joints" in result["reasons"]
    assert "low_pose_coverage" not in result["reasons"]
    assert "low_keypoint_confidence" not in result["reasons"]
    assert "extreme_joint_mean" not in result["reasons"]
    assert result["joint_degenerate_fraction"] == pytest.approx(0.9)


def test_quality_accepts_clip_with_clean_joints():
    result = evaluate_bst_clip_quality(_provenance(joint_degenerate_fraction=0.0))

    assert result["eligible"] is True
    assert "degenerate_joints" not in result["reasons"]


def test_quality_soft_penalizes_extreme_joint_mean():
    base = evaluate_bst_clip_quality(_provenance(joint_abs_mean=0.0))
    extreme = evaluate_bst_clip_quality(_provenance(joint_abs_mean=1.5))
    assert base["score"] == 1.0
    assert extreme["score"] == pytest.approx(0.9)
    assert "extreme_joint_mean" in extreme["reasons"]
    assert "extreme_joint_mean" not in base["reasons"]


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


def test_quality_rejects_clip_with_too_many_court_rejected_points(monkeypatch):
    from app.config import settings as settings_mod

    rejected = [True] * 6 + [False] * 14

    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "court")
    result = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))

    assert result["eligible"] is False
    assert result["score"] == 0.8
    assert result["reasons"] == ["court_rejected_shuttle"]


def test_quality_hard_rejects_court_rejected_only_in_court_norm_mode(monkeypatch):
    from app.config import settings as settings_mod

    rejected = [True] * 6 + [False] * 14  # 0.30 > 0.25

    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "resolution")
    result_res = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))
    assert result_res["eligible"] is True or "court_rejected_shuttle" not in result_res["reasons"]
    assert result_res["score"] == pytest.approx(0.8)  # soft −0.20 only
    assert "court_rejected_shuttle" not in result_res["reasons"]

    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "court")
    result_court = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))
    assert result_court["eligible"] is False
    assert "court_rejected_shuttle" in result_court["reasons"]


def test_quality_does_not_hard_reject_repaired_court_rejected_points():
    observed = [True] * 8 + [False] * 4 + [True] * 8
    repaired = [False] * 8 + [True] * 4 + [False] * 8
    rejected = [False] * 8 + [True] * 4 + [False] * 8

    result = evaluate_bst_clip_quality(
        _provenance(
            shuttle_observed=observed,
            shuttle_repaired=repaired,
            shuttle_court_rejected=rejected,
        )
    )

    assert result["eligible"] is True
    assert result["score"] == pytest.approx(0.9)
    assert result["reasons"] == []


def test_quality_accumulates_all_failed_hard_checks_and_clamps_score(monkeypatch):
    from app.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "court")
    result = evaluate_bst_clip_quality(_provenance(
        video_len=20,
        shuttle_observed=[True] * 6 + [False] * 14,
        shuttle_repaired=[False] * 20,
        shuttle_interpolated=[True] * 20,
        shuttle_court_rejected=[True] * 6 + [False] * 14,
        pose_present_far=[False] * 20,
        pose_present_near=[False] * 20,
        pose_keypoint_confidence_far=[0.1] * 20,
        pose_keypoint_confidence_near=[0.1] * 20,
        bbox_gap_far=[11] * 20,
        bbox_gap_near=[11] * 20,
    ))

    assert result["eligible"] is False
    assert result["score"] == 0.0
    assert result["reasons"] == [
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
    # Middle 6 frames are repaired (InpaintNet) AND interpolated (linear fill).
    repaired = [False] * 7 + [True] * 6 + [False] * 7
    interpolated = [False] * 7 + [True] * 6 + [False] * 7
    result = evaluate_bst_clip_quality(_provenance(
        shuttle_observed=observed,
        shuttle_repaired=repaired,
        shuttle_interpolated=interpolated,
    ))

    # Repaired coords now count as present, so the clip is fully present and
    # has no shuttle gap — it is rejected only by the soft quality-score floor.
    assert result["repaired_shuttle_fraction"] == 0.3
    assert result["interpolated_shuttle_fraction"] == 0.3
    assert result["present_shuttle_fraction"] == 1.0
    assert result["max_shuttle_gap_frames"] == 0
    assert result["eligible"] is False
    assert "too_many_interpolated_shuttle" not in result["reasons"]
    assert "long_shuttle_gap" not in result["reasons"]
    assert "low_quality_score" in result["reasons"]


def test_quality_counts_repaired_as_present_and_admits_clip():
    # A clip that was 100% interpolated before, but fully repaired by InpaintNet,
    # must now be admitted (no gap, present == 1.0) and only mildly penalized.
    result = evaluate_bst_clip_quality(_provenance(
        shuttle_observed=[False] * 20,
        shuttle_repaired=[True] * 20,
        shuttle_interpolated=[False] * 20,
    ))

    assert result["present_shuttle_fraction"] == 1.0
    assert result["max_shuttle_gap_frames"] == 0
    assert "low_observed_shuttle" not in result["reasons"]
    assert "long_shuttle_gap" not in result["reasons"]
    # Mild repaired penalty (0.50 * 1.0) leaves score 0.50 < 0.70 -> rejected by floor.
    assert result["score"] == pytest.approx(0.5)
    assert result["eligible"] is False
    assert "low_quality_score" in result["reasons"]


def test_quality_contact_window_limits_shuttle_gap_measurement():
    # A long gap far from contact (pre-serve tail, outside the contact window)
    # must not disqualify a clip whose contact window is contiguous. Use a
    # 40-frame clip so the window (contact +- 15) does not cover the whole clip.
    observed = [False] * 15 + [True] * 25  # 15-frame gap at frames 0-14, before contact
    result = evaluate_bst_clip_quality(_provenance(
        video_len=40,
        contact_frame_index=35,  # contact near the end -> gap is far away
        shuttle_observed=observed,
        shuttle_repaired=[False] * 40,
        shuttle_interpolated=[False] * 40,
        shuttle_court_rejected=[False] * 40,
    ))

    # Unbounded gap is 15 (> 12), but within the contact window (20-40) the
    # shuttle is fully present, so the gap is NOT fatal.
    assert "long_shuttle_gap" not in result["reasons"]
    assert result["max_shuttle_gap_frames"] == 0
    assert result["full_shuttle_gap_frames"] == 15
    assert result["present_shuttle_fraction"] == pytest.approx(25 / 40)


def test_quality_long_gap_inside_contact_window_still_rejects():
    # A gap that straddles the contact frame (inside the window) is still fatal.
    # 3 present + 13-gap + 4 present -> present 0.35 (no low_observed), but the
    # 13-frame gap (> 12) around contact idx 10 triggers long_shuttle_gap.
    observed = [True] * 3 + [False] * 13 + [True] * 4
    result = evaluate_bst_clip_quality(_provenance(
        shuttle_observed=observed,
        shuttle_repaired=[False] * 20,
        shuttle_interpolated=[False] * 20,
    ))

    assert "long_shuttle_gap" in result["reasons"]
    assert "low_observed_shuttle" not in result["reasons"]
    assert result["eligible"] is False


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


def test_aim_alpha_quality_does_not_flag_repaired_contact_as_court_instability():
    observed = [True] * 10 + [False] + [True] * 9
    repaired = [False] * 10 + [True] + [False] * 9
    rejected = [False] * 10 + [True] + [False] * 9

    result = evaluate_aim_alpha_quality(
        _provenance(
            shuttle_observed=observed,
            shuttle_repaired=repaired,
            shuttle_court_rejected=rejected,
        )
    )

    assert result["reliable"] is True
    assert "contact_shuttle_unstable" not in result["reasons"]
