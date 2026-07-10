"""Quality scoring for evidence supplied to the fixed-input BST model."""

import numpy as np

from app.config.settings import settings


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _coverage(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _median_confidence(values: np.ndarray, present: np.ndarray) -> float:
    usable = values[present]
    return float(np.median(usable)) if len(usable) else 0.0


def evaluate_bst_clip_quality(provenance: dict) -> dict:
    """Return deterministic admission evidence for one unpadded BST clip."""
    video_len = int(provenance["video_len"])

    def values(name: str, dtype) -> np.ndarray:
        return np.asarray(provenance[name][:video_len], dtype=dtype)

    observed = values("shuttle_observed", bool)
    repaired = values("shuttle_repaired", bool)
    interpolated = values("shuttle_interpolated", bool)
    rejected = values("shuttle_court_rejected", bool)
    far_present = values("pose_present_far", bool)
    near_present = values("pose_present_near", bool)
    far_conf = values("pose_keypoint_confidence_far", float)
    near_conf = values("pose_keypoint_confidence_near", float)
    far_gaps = values("bbox_gap_far", float)
    near_gaps = values("bbox_gap_near", float)

    observed_fraction = _coverage(observed)
    max_shuttle_gap = _longest_false_run(observed)
    far_coverage = _coverage(far_present)
    near_coverage = _coverage(near_present)
    far_median_conf = _median_confidence(far_conf, far_present)
    near_median_conf = _median_confidence(near_conf, near_present)
    max_bbox_gap = int(max(np.max(far_gaps, initial=0), np.max(near_gaps, initial=0)))

    reasons = []
    score = 1.0
    if video_len < settings.bst_min_clip_video_frames:
        reasons.append("clip_too_short")
    if observed_fraction < settings.bst_min_observed_shuttle_fraction:
        reasons.append("low_observed_shuttle")
        score -= 0.35
    if max_shuttle_gap > settings.bst_max_raw_shuttle_gap_frames:
        reasons.append("long_shuttle_gap")
        score -= 0.25
    if rejected.any():
        reasons.append("court_rejected_shuttle")
        score -= 0.20
    if min(far_coverage, near_coverage) < settings.bst_min_pose_coverage:
        reasons.append("low_pose_coverage")
        score -= 0.20
    if min(far_median_conf, near_median_conf) < settings.bst_min_keypoint_confidence:
        reasons.append("low_keypoint_confidence")
        score -= 0.15
    if max_bbox_gap > settings.bst_max_bbox_interp_gap:
        reasons.append("long_bbox_gap")
        score -= 0.15

    score = float(np.clip(score, 0.0, 1.0))
    hard_failures = bool(reasons)
    if score < settings.bst_quality_score_min:
        reasons.append("low_quality_score")

    return {
        "eligible": not hard_failures and score >= settings.bst_quality_score_min,
        "score": score,
        "reasons": reasons,
        "observed_shuttle_frames": int(observed.sum()),
        "repaired_shuttle_frames": int(repaired.sum()),
        "interpolated_shuttle_frames": int(interpolated.sum()),
        "court_rejected_shuttle_frames": int(rejected.sum()),
        "observed_shuttle_fraction": observed_fraction,
        "max_shuttle_gap_frames": max_shuttle_gap,
        "far_pose_coverage": far_coverage,
        "near_pose_coverage": near_coverage,
        "far_pose_median_confidence": far_median_conf,
        "near_pose_median_confidence": near_median_conf,
        "max_bbox_gap_frames": max_bbox_gap,
    }
