from app.pipeline.shared.bst_input_quality import evaluate_bst_clip_quality


def _provenance(**overrides):
    value = {
        "video_len": 20,
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


def test_quality_rejects_court_rejected_point_even_when_other_coverage_is_good():
    rejected = [False] * 20
    rejected[3] = True

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
        shuttle_court_rejected=[True] + [False] * 9,
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
        "low_quality_score",
    ]
