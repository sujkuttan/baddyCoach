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


def _window_slice(video_len: int, center: int, radius: int) -> slice:
    start = max(0, center - radius)
    stop = min(video_len, center + radius + 1)
    return slice(start, stop)


def evaluate_bst_clip_quality(provenance: dict) -> dict:
    """Return deterministic admission evidence for one unpadded BST clip."""
    video_len = int(provenance["video_len"])

    def values(name: str, dtype) -> np.ndarray:
        return np.asarray(provenance[name][:video_len], dtype=dtype)

    observed = values("shuttle_observed", bool)
    repaired = values("shuttle_repaired", bool)
    interpolated = values("shuttle_interpolated", bool)
    rejected = values("shuttle_court_rejected", bool)
    observed_rejected = rejected & observed
    far_present = values("pose_present_far", bool)
    near_present = values("pose_present_near", bool)
    far_conf = values("pose_keypoint_confidence_far", float)
    near_conf = values("pose_keypoint_confidence_near", float)
    far_gaps = values("bbox_gap_far", float)
    near_gaps = values("bbox_gap_near", float)

    observed_fraction = _coverage(observed)
    repaired_fraction = _coverage(repaired)
    interpolated_fraction = _coverage(interpolated)
    rejected_fraction = _coverage(observed_rejected)
    max_shuttle_gap = _longest_false_run(observed)
    far_coverage = _coverage(far_present)
    near_coverage = _coverage(near_present)
    far_median_conf = _median_confidence(far_conf, far_present)
    near_median_conf = _median_confidence(near_conf, near_present)
    max_bbox_gap = int(max(np.max(far_gaps, initial=0), np.max(near_gaps, initial=0)))

    reasons = []
    hard_reasons = []
    score = 1.0
    if video_len < settings.bst_min_clip_video_frames:
        hard_reasons.append("clip_too_short")
    if observed_fraction < settings.bst_min_observed_shuttle_fraction:
        hard_reasons.append("low_observed_shuttle")
        score -= 0.35
    if max_shuttle_gap > settings.bst_max_raw_shuttle_gap_frames:
        hard_reasons.append("long_shuttle_gap")
        score -= 0.25
    if observed_rejected.any():
        score -= 0.20
    if (
        settings.bst_shuttle_norm == "court"
        and rejected_fraction > settings.bst_max_court_rejected_shuttle_fraction
    ):
        hard_reasons.append("court_rejected_shuttle")
    if min(far_coverage, near_coverage) < settings.bst_min_pose_coverage:
        hard_reasons.append("low_pose_coverage")
        score -= 0.20
    if min(far_median_conf, near_median_conf) < settings.bst_min_keypoint_confidence:
        hard_reasons.append("low_keypoint_confidence")
        score -= 0.15
    if max_bbox_gap > settings.bst_max_bbox_interp_gap:
        hard_reasons.append("long_bbox_gap")
        score -= 0.15

    score -= 0.50 * repaired_fraction
    score -= 0.80 * interpolated_fraction
    if repaired_fraction > settings.bst_max_repaired_shuttle_fraction:
        hard_reasons.append("too_many_repaired_shuttle")
    if interpolated_fraction > settings.bst_max_interpolated_shuttle_fraction:
        hard_reasons.append("too_many_interpolated_shuttle")

    score = float(np.clip(score, 0.0, 1.0))
    reasons.extend(hard_reasons)
    if score < settings.bst_quality_score_min:
        reasons.append("low_quality_score")

    return {
        "eligible": not hard_reasons and score >= settings.bst_quality_score_min,
        "score": score,
        "reasons": reasons,
        "observed_shuttle_frames": int(observed.sum()),
        "repaired_shuttle_frames": int(repaired.sum()),
        "interpolated_shuttle_frames": int(interpolated.sum()),
        "court_rejected_shuttle_frames": int(observed_rejected.sum()),
        "observed_shuttle_fraction": observed_fraction,
        "repaired_shuttle_fraction": repaired_fraction,
        "interpolated_shuttle_fraction": interpolated_fraction,
        "court_rejected_shuttle_fraction": rejected_fraction,
        "max_shuttle_gap_frames": max_shuttle_gap,
        "far_pose_coverage": far_coverage,
        "near_pose_coverage": near_coverage,
        "far_pose_median_confidence": far_median_conf,
        "near_pose_median_confidence": near_median_conf,
        "max_bbox_gap_frames": max_bbox_gap,
    }


def evaluate_aim_alpha_quality(provenance: dict) -> dict:
    """Return stricter quality evidence for using AimPlayer alpha."""
    video_len = int(provenance["video_len"])

    def values(name: str, dtype) -> np.ndarray:
        return np.asarray(provenance[name][:video_len], dtype=dtype)

    contact_center = int(provenance.get("contact_frame_index", 0))
    contact_window = _window_slice(video_len, contact_center, settings.aim_alpha_contact_window)

    observed = values("shuttle_observed", bool)[contact_window]
    repaired = values("shuttle_repaired", bool)[contact_window]
    interpolated = values("shuttle_interpolated", bool)[contact_window]
    rejected = values("shuttle_court_rejected", bool)[contact_window]
    observed_rejected = rejected & observed
    far_present = values("pose_present_far", bool)[contact_window]
    near_present = values("pose_present_near", bool)[contact_window]
    far_conf = values("pose_keypoint_confidence_far", float)[contact_window]
    near_conf = values("pose_keypoint_confidence_near", float)[contact_window]
    far_ids = values("resolved_far_pid", object)[contact_window]
    near_ids = values("resolved_near_pid", object)[contact_window]
    far_dist = values("wrist_shuttle_distance_far", float)[contact_window]
    near_dist = values("wrist_shuttle_distance_near", float)[contact_window]

    contact_window_valid = len(observed) > 0
    far_cov = _coverage(far_present)
    near_cov = _coverage(near_present)
    far_med = _median_confidence(far_conf, far_present)
    near_med = _median_confidence(near_conf, near_present)
    pose_coverage_gap = abs(far_cov - near_cov)
    pose_conf_gap = abs(far_med - near_med)

    coverage_ratio = pose_coverage_gap / max(settings.aim_alpha_max_pose_coverage_gap, 1e-6)
    conf_ratio = pose_conf_gap / max(settings.aim_alpha_max_pose_conf_gap, 1e-6)
    pose_balance_score = float(np.clip(1.0 - max(coverage_ratio, conf_ratio), 0.0, 1.0))

    unique_far = {pid for pid in far_ids.tolist() if pid}
    unique_near = {pid for pid in near_ids.tolist() if pid}
    identity_stable = len(unique_far) == 1 and len(unique_near) == 1 and unique_far != unique_near

    valid_distance = np.isfinite(far_dist) & np.isfinite(near_dist)
    if valid_distance.any():
        contact_separation = float(np.max(np.abs(far_dist[valid_distance] - near_dist[valid_distance])))
    else:
        contact_separation = 0.0

    reasons = []
    score = 1.0
    if not contact_window_valid:
        reasons.append("contact_window_missing")
        score -= 1.0
    if not observed.any():
        reasons.append("contact_shuttle_missing")
        score -= 0.35
    if interpolated.any() or observed_rejected.any():
        reasons.append("contact_shuttle_unstable")
        score -= 0.30
    if pose_coverage_gap > settings.aim_alpha_max_pose_coverage_gap or pose_conf_gap > settings.aim_alpha_max_pose_conf_gap:
        reasons.append("contact_pose_imbalance")
        score -= 0.25
    if not identity_stable:
        reasons.append("identity_unstable")
        score -= 0.20
    if contact_separation < settings.aim_alpha_min_contact_separation:
        reasons.append("contact_separation_too_small")
        score -= 0.20

    score = float(np.clip(score, 0.0, 1.0))
    reliable = not reasons and score >= settings.aim_alpha_min_quality_score
    return {
        "reliable": reliable,
        "score": score,
        "reasons": reasons,
        "contact_window_valid": contact_window_valid,
        "pose_balance_score": pose_balance_score,
        "identity_stable": identity_stable,
        "contact_separation": contact_separation,
        "pose_coverage_gap": pose_coverage_gap,
        "pose_conf_gap": pose_conf_gap,
    }
