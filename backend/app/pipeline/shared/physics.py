"""Physics-consistency gate + BST × physics ensemble (Spec 6).

Physics has veto power over BST: rejects physically-impossible strokes
and fills in when BST abstains, but defers to BST when BST is plausible.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from app.config.settings import settings
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, image_to_court
from app.pipeline.shared.logging import logger


# ── Physical families ───────────────────────────────────────────────
FAMILIES = {
    "DOWN_FAST_OVERHEAD": "smash",
    "DOWN_SLOW_OVERHEAD": "drop",
    "OVERHEAD_DEEP": "clear",
    "FLAT_FAST": "drive",
    "UP_DEEP_UNDERARM": "lift",
    "SHORT_SLOW_FRONT": "net_shot",
}

# Veto table: for each coach class, the set of REQUIRED conditions.
# Each condition is a key in the Features dataclass; enforced only if
# the underlying cue is usable (not None).
# Conditions: "descend", "ascend", "flat", "fast", "med", "slow",
#   "overhead", "side", "underarm", "low",
#   "short", "mid", "deep",
#   "rise_fall", "front", "back"
CLASS_VETO = {
    "smash":     {"descend", "fast", "overhead"},
    "rush":      {"descend", "med", "front", "overhead"},
    "drop":      {"descend", "slow", "overhead", "short"},
    "clear":     {"overhead", "rise_fall", "med", "deep"},
    "drive":     {"flat", "fast", "side"},
    "net_shot":  {"slow", "short", "front", "underarm"},
    "block":     {"slow", "short", "descend"},
    "push":      {"slow", "short", "low"},
    "lift":      {"ascend", "deep", "underarm"},
    "short_serve": {"ascend", "slow", "short", "underarm"},
    "long_serve":  {"ascend", "deep", "underarm"},
    "cross_court": {"cross"},  # orthogonal — combined with other classes
}


@dataclass
class Features:
    quality: float = 0.0
    usable: bool = False
    v_down: Optional[float] = None       # >0 descending, <0 ascending
    speed_mps: Optional[float] = None    # m/s (homography) or None
    speed_norm: Optional[float] = None   # normalized-frame/s (no homography)
    arc_rise_fall: Optional[bool] = None # True if arc rises then falls
    dx_total: Optional[float] = None     # lateral travel (normalized)
    contact: Optional[str] = None        # "overhead", "side", "underarm", "low"
    zone: Optional[str] = None           # "front", "mid", "back"
    depth: Optional[str] = None          # "short", "mid", "deep"


def court_speed_mps(seg_x, seg_y, court, fps) -> Optional[float]:
    """Compute shuttle speed in m/s via homography projection."""
    if not court or not court.get("homography") or not court.get("valid", False):
        return None
    H = np.array(court["homography"], dtype=np.float64)
    if len(seg_x) < 2:
        return None
    p0 = image_to_court(H, (float(seg_x[0]), float(seg_y[0])))
    p1 = image_to_court(H, (float(seg_x[-1]), float(seg_y[-1])))
    if p0 is None or p1 is None:
        return None
    dist_m = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    dt_s = (len(seg_x) - 1) / fps
    if dt_s <= 0:
        return None
    return float(dist_m / dt_s)


def px_speed_per_s(seg_x, seg_y, fps, vid_w, vid_h) -> float:
    """Compute shuttle speed in normalized-frame-units per second."""
    diag = np.sqrt(vid_w**2 + vid_h**2) if vid_w and vid_h else 1920.0
    if len(seg_x) < 2:
        return 0.0
    dx = seg_x[-1] - seg_x[0]
    dy = seg_y[-1] - seg_y[0]
    dist_px = np.sqrt(dx**2 + dy**2)
    dt_s = (len(seg_x) - 1) / fps
    if dt_s <= 0:
        return 0.0
    return float((dist_px / diag) / dt_s)


def classify_arc(y_vals) -> str:
    """Classify shuttle vertical trajectory.

    Returns "down_monotonic", "rise_fall", or "flat".
    """
    if len(y_vals) < 3:
        return "flat"
    valid = ~np.isnan(y_vals)
    if valid.sum() < 3:
        return "flat"
    y = y_vals[valid]
    dy = np.diff(y)
    # If all dy have same sign (all down or all up) → monotonic
    if np.all(dy >= -1e-6) or np.all(dy <= 1e-6):
        return "down_monotonic" if np.median(dy) > 0 else "up_monotonic"
    # If it rises then falls (or vice versa)
    mid = len(y) // 2
    first_half = y[:mid]
    second_half = y[mid:]
    if np.median(np.diff(first_half)) < 0 and np.median(np.diff(second_half)) > 0:
        return "rise_fall"  # up then down
    return "flat"


def contact_height(pose_df, frame, hitter_id, net_y) -> Optional[str]:
    """Determine contact height from wrist vs shoulder position.

    Returns "overhead", "side", "underarm", "low", or None if pose missing.
    """
    if pose_df is None or len(pose_df) == 0:
        return None
    row = pose_df[(pose_df["frame"] == frame) & (pose_df["player_id"] == hitter_id)]
    if len(row) == 0:
        return None
    raw = row.iloc[0]["keypoints"]
    kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
    if kps.ndim != 2 or kps.shape[0] < 13 or kps.shape[1] < 2:
        return None
    # Keypoint indices: 5=left shoulder, 6=right shoulder, 9=left wrist, 10=right wrist
    l_wrist = kps[9, 1] if len(kps) > 9 else None
    r_wrist = kps[10, 1] if len(kps) > 10 else None
    l_shoulder = kps[5, 1] if len(kps) > 5 else None
    r_shoulder = kps[6, 1] if len(kps) > 6 else None
    if l_wrist is None or r_wrist is None or l_shoulder is None or r_shoulder is None:
        return None
    # Image y: smaller = higher. Wrist above shoulder → overhead
    wrist_y = min(l_wrist, r_wrist)
    shoulder_y = min(l_shoulder, r_shoulder)
    if wrist_y < shoulder_y - 20:  # wrist significantly higher than shoulder
        return "overhead"
    # Wrist near shoulder level
    if abs(wrist_y - shoulder_y) < 40:
        return "side"
    # Wrist below shoulder but above hip (hip is kps[11] or kps[12])
    hip_y = min(kps[11, 1] if len(kps) > 11 else shoulder_y + 100,
                kps[12, 1] if len(kps) > 12 else shoulder_y + 100)
    if wrist_y < hip_y:
        return "underarm"
    return "low"


def court_zone(court_x) -> Optional[str]:
    """Determine hitter zone from court_x position (normalized 0..1 or metres)."""
    if court_x is None or np.isnan(court_x):
        return None
    if court_x < settings.physics_zone_front:
        return "front"
    if court_x > settings.physics_zone_back:
        return "back"
    return "mid"


def landing_depth(dx_total, zone) -> Optional[str]:
    """Determine landing depth from lateral travel and hitter zone.

    Uses dx_total (normalized 0..1) as a rough heuristic.
    """
    if dx_total is None:
        return None
    if dx_total < 0.15:
        return "short"
    if dx_total < 0.40:
        return "mid"
    return "deep"


def extract_physics_features(
    f: int,
    shuttle_raw: "pd.DataFrame",
    pose_df: "pd.DataFrame",
    hitter_id: str,
    court: dict,
    fps: float,
    vid_w: float,
    vid_h: float,
) -> Features:
    """Extract physics features for a hit frame from shuttle_raw and pose.

    Features degrade independently; missing cue → None (never veto on missing cue).
    """
    K = settings.physics_window_frames
    min_valid = settings.physics_min_valid
    min_conf = settings.shuttle_min_conf

    # Post-contact window
    seg = shuttle_raw[
        (shuttle_raw["frame"] >= f) &
        (shuttle_raw["frame"] <= f + K) &
        (shuttle_raw["confidence"] >= min_conf)
    ].copy().sort_values("frame")

    if len(seg) < min_valid:
        return Features(quality=len(seg) / K if K > 0 else 0, usable=False)

    quality = min(1.0, len(seg) / K)

    seg_x = seg["x"].values.astype(np.float64)
    seg_y = seg["y"].values.astype(np.float64)

    # Vertical direction
    dy = np.diff(seg_y)
    v_down = float(np.median(dy)) if len(dy) > 0 else 0.0

    # Speed (prefer m/s via homography)
    speed_mps = court_speed_mps(seg_x, seg_y, court, fps)
    speed_norm = None
    if speed_mps is None:
        speed_norm = px_speed_per_s(seg_x, seg_y, fps, vid_w, vid_h)

    # Arc
    arc = classify_arc(seg_y)

    # Lateral travel
    dx_total = float(abs(seg_x[-1] - seg_x[0]) / (vid_w if vid_w > 0 else 1920.0))

    # Contact height from pose
    net_y = None
    if court and court.get("corners_pixel"):
        corners = court["corners_pixel"]
        bl_y = corners[0][1]
        tl_y = corners[2][1]
        net_y = (tl_y + bl_y) / 2.0
    contact = contact_height(pose_df, f, hitter_id, net_y)

    # Hitter zone from court position (via homography)
    zone = None
    if court and court.get("homography") and court.get("valid", False):
        H = np.array(court["homography"], dtype=np.float64)
        pos = image_to_court(H, (float(seg_x[0]), float(seg_y[0])))
        if pos is not None:
            cx = pos[0] / settings.court_length
            zone = court_zone(cx)

    # Landing depth
    depth = landing_depth(dx_total, zone)

    return Features(
        quality=quality,
        usable=True,
        v_down=v_down,
        speed_mps=speed_mps,
        speed_norm=speed_norm,
        arc_rise_fall=(arc == "rise_fall"),
        dx_total=dx_total,
        contact=contact,
        zone=zone,
        depth=depth,
    )


def _check_condition(cond: str, feats: Features) -> Optional[bool]:
    """Check a single physical condition against features.

    Returns True if satisfied, False if violated, None if cue missing (skip).
    """
    if cond == "descend":
        if feats.v_down is None:
            return None
        return feats.v_down > 0
    if cond == "ascend":
        if feats.v_down is None:
            return None
        return feats.v_down < 0
    if cond == "flat":
        if feats.v_down is None:
            return None
        return abs(feats.v_down) < 1.0  # near-zero vertical motion
    if cond in ("fast", "med", "slow"):
        if feats.speed_mps is not None:
            thr_fast = settings.physics_speed_fast_mps
            thr_slow = settings.physics_speed_slow_mps
        elif feats.speed_norm is not None:
            thr_fast = settings.physics_speed_fast_norm
            thr_slow = settings.physics_speed_slow_norm
        else:
            return None
        speed = feats.speed_mps if feats.speed_mps is not None else feats.speed_norm
        if speed is None:
            return None
        if cond == "fast":
            return speed >= thr_fast
        if cond == "slow":
            return speed < thr_slow
        if cond == "med":
            return thr_slow <= speed < thr_fast
    if cond in ("overhead", "side", "underarm", "low"):
        if feats.contact is None:
            return None
        if cond == "overhead":
            return feats.contact == "overhead"
        if cond == "side":
            return feats.contact == "side"
        if cond == "underarm":
            return feats.contact == "underarm"
        if cond == "low":
            return feats.contact == "low"
    if cond in ("short", "mid", "deep"):
        if feats.depth is None:
            return None
        return feats.depth == cond
    if cond in ("front", "back"):
        if feats.zone is None:
            return None
        if cond == "front":
            return feats.zone == "front"
        return feats.zone == "back"
    if cond == "rise_fall":
        if feats.arc_rise_fall is None:
            return None
        return feats.arc_rise_fall
    if cond == "cross":
        if feats.dx_total is None:
            return None
        return feats.dx_total >= settings.physics_cross_court_dx
    return None


def consistent(bst_stroke: str, feats: Features) -> bool:
    """Check if a BST-predicted stroke is physically consistent.

    A stroke is consistent if ALL its REQUIRED conditions pass.
    Conditions with None cues (missing data) are skipped — never veto
    on a missing cue.
    """
    required = CLASS_VETO.get(bst_stroke, set())
    if not required:
        return True
    for cond in required:
        ok = _check_condition(cond, feats)
        if ok is False:  # explicitly violated (not None = missing)
            return False
    return True


def best_consistent_class(
    probs: np.ndarray, classes: list, feats: Features
) -> Optional[str]:
    """Walk BST probability vector high→low, return first physically-consistent class."""
    if probs is None or len(probs) == 0:
        return None
    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        st = classes[idx] if idx < len(classes) else "unknown"
        if st == "unknown":
            continue
        if consistent(st, feats):
            return st
    return None


def classify_physics(feats: Features) -> tuple:
    """Map features to a physical family + representative stroke + confidence.

    Returns (family_name, stroke_type, confidence).
    Confidence = quality * mean(margin) where margin ∈ [0,1] per defining cue.
    """
    if not feats.usable:
        return (None, "unknown", 0.0)

    for fam_name, rep_stroke in FAMILIES.items():
        required = CLASS_VETO.get(rep_stroke, set())
        if not required:
            continue
        margins = []
        all_pass = True
        for cond in required:
            ok = _check_condition(cond, feats)
            if ok is False:
                all_pass = False
                break
            if ok is True:
                margins.append(1.0)
            # ok is None → missing cue, skip margin
        if all_pass and margins:
            conf = feats.quality * (sum(margins) / len(margins))
            conf = min(conf, 0.99)
            return (fam_name, rep_stroke, conf)

    return (None, "unknown", 0.0)


def combine_agree(c_bst: float, c_phy: float) -> float:
    """Confidence boost when BST and physics agree.

    c_bst + (1 - c_bst) * c_phy * agree_boost, bounded ≤ 0.99.
    """
    boost = c_bst + (1.0 - c_bst) * c_phy * settings.physics_agree_boost
    return min(boost, 0.99)


def apply_physics_ensemble(
    shot_records: list,
    probs_matrix: np.ndarray,
    classes: list,
    shuttle_raw: "pd.DataFrame",
    pose_df: "pd.DataFrame",
    court: dict,
    fps: float,
    vid_w: float,
    vid_h: float,
) -> list:
    """Apply the physics-consistency gate to all shots.

    Modifies shot_records in-place and returns them. Adds 'stroke_source'
    field to each shot.
    """
    if not settings.physics_gate_enabled:
        for s in shot_records:
            s["stroke_source"] = "bst"
        return shot_records

    if shuttle_raw is None or len(shuttle_raw) == 0:
        for s in shot_records:
            s["stroke_source"] = "bst_no_physics"
        return shot_records

    override_count = 0
    fallback_count = 0
    agree_count = 0
    bst_count = 0
    no_physics_count = 0

    for i, shot in enumerate(shot_records):
        f = int(shot["frame"])
        hitter_id = shot.get("player_id", "player_1")
        is_fallback = shot.get("is_bst_fallback", False)

        feats = extract_physics_features(f, shuttle_raw, pose_df, hitter_id, court, fps, vid_w, vid_h)

        if not feats.usable or feats.quality < settings.physics_quality_min:
            shot["stroke_source"] = "bst_no_physics"
            no_physics_count += 1
            continue

        fam, phys_stroke, c_p = classify_physics(feats)

        if is_fallback:
            if phys_stroke != "unknown" and c_p > 0:
                shot["stroke_type"] = phys_stroke
                shot["stroke_confidence"] = c_p
                shot["shuttleset_class_id"] = 0
                shot["is_rule_based"] = True
                shot["is_bst_fallback"] = True
                shot["stroke_source"] = "physics_fallback"
                fallback_count += 1
            else:
                shot["stroke_source"] = "bst_no_physics"
                no_physics_count += 1
            continue

        bst_stroke = shot.get("stroke_type", "unknown")
        c_bst = shot.get("stroke_confidence", 0.5)

        # Get probs for this shot (index i in the probs matrix)
        shot_probs = probs_matrix[i] if probs_matrix is not None and i < len(probs_matrix) else None

        if consistent(bst_stroke, feats):
            if bst_stroke == phys_stroke and phys_stroke != "unknown":
                shot["stroke_confidence"] = combine_agree(c_bst, c_p)
                shot["stroke_source"] = "agree"
                agree_count += 1
            else:
                shot["stroke_source"] = "bst"
                bst_count += 1
        else:
            # BST impossible → physics VETO
            alt = None
            if shot_probs is not None:
                alt = best_consistent_class(shot_probs, classes, feats)
            stroke_before = bst_stroke
            if alt is not None:
                shot["stroke_type"] = alt
            else:
                shot["stroke_type"] = phys_stroke if phys_stroke != "unknown" else bst_stroke
            shot["stroke_confidence"] = min(c_bst, c_p) if c_p > 0 else c_bst
            shot["stroke_source"] = "physics_override"
            override_count += 1
            logger.info(
                "physics veto", frame=f, before=stroke_before, after=shot["stroke_type"], family=fam or "?",
            )

        total = len(shot_records)
        logger.info(
            "Physics gate",
            total=total, agree=agree_count, bst=bst_count,
            veto=override_count, fallback=fallback_count, no_physics=no_physics_count,
        )

    return shot_records
