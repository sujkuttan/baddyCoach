"""Technique reference ranges for angle-based quality scoring.

Adapted from Haimantika/badminton-coach: per-stroke reference ranges for
elbow extension, shoulder angle, knee flexion, and trunk lean.  Used to
score technique quality and validate stroke classifications.

All angles in degrees.
"""

import numpy as np

# ── Reference ranges per stroke type ────────────────────────────────
# Each entry: { "angle_name": (min_ok, max_ok, min_good, max_good) }
# ok range = acceptable, good range = ideal (subset of ok)

REFERENCE_RANGES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "smash": {
        "elbow":      (120, 180,  150, 175),   # near-full extension
        "shoulder":   (30,  100,  50,  90),     # high arm raise
        "knee":       (100, 170,  120, 155),    # slight bend for power
        "trunk_lean": (0,   30,   5,   20),     # forward lean
    },
    "clear": {
        "elbow":      (110, 175, 130, 165),
        "shoulder":   (40,  110, 60,  95),
        "knee":       (100, 165, 115, 150),
        "trunk_lean": (0,   25,   5,   18),
    },
    "drop": {
        "elbow":      (100, 165, 115, 150),
        "shoulder":   (35,  105, 50,  90),
        "knee":       (105, 170, 120, 155),
        "trunk_lean": (0,   20,   3,   15),
    },
    "lift": {
        "elbow":      (100, 170, 120, 160),
        "shoulder":   (20,  85,  30,  70),
        "knee":       (90,  160, 105, 145),
        "trunk_lean": (0,   15,   2,   12),
    },
    "drive": {
        "elbow":      (100, 170, 120, 160),
        "shoulder":   (20,  80,  30,  65),
        "knee":       (100, 165, 115, 150),
        "trunk_lean": (0,   20,   2,   14),
    },
    "net_shot": {
        "elbow":      (60,  150, 80,  130),
        "shoulder":   (10,  60,  15,  45),
        "knee":       (90,  160, 105, 145),
        "trunk_lean": (0,   25,   2,   18),
    },
    "block": {
        "elbow":      (80,  155, 100, 140),
        "shoulder":   (15,  70,  25,  55),
        "knee":       (100, 165, 115, 150),
        "trunk_lean": (0,   20,   2,   14),
    },
    "push": {
        "elbow":      (80,  160, 100, 145),
        "shoulder":   (10,  65,  20,  50),
        "knee":       (95,  165, 110, 150),
        "trunk_lean": (0,   18,   2,   12),
    },
    "defensive_lift": {
        "elbow":      (90,  160, 110, 150),
        "shoulder":   (15,  75,  25,  60),
        "knee":       (80,  155, 95,  140),
        "trunk_lean": (0,   20,   3,   15),
    },
    "long_serve": {
        "elbow":      (110, 175, 130, 165),
        "shoulder":   (30,  90,  40,  75),
        "knee":       (100, 165, 115, 150),
        "trunk_lean": (0,   20,   3,   14),
    },
    "short_serve": {
        "elbow":      (60,  135, 80,  120),
        "shoulder":   (10,  50,  15,  40),
        "knee":       (95,  160, 110, 148),
        "trunk_lean": (0,   15,   1,   10),
    },
    "cross_court": {
        "elbow":      (100, 170, 120, 158),
        "shoulder":   (25,  85,  35,  70),
        "knee":       (100, 165, 115, 150),
        "trunk_lean": (0,   25,   3,   18),
    },
}

_ANGLES = ["elbow", "shoulder", "knee", "trunk_lean"]


def score_technique(stroke_type: str, angles: dict[str, float]) -> dict:
    """Score technique quality for a stroke.

    Parameters
    ----------
    stroke_type : str
        Coach-class stroke name (e.g. 'smash', 'lift').
    angles : dict[str, float]
        Measured angles in degrees.  Expected keys: elbow, shoulder,
        knee, trunk_lean.

    Returns
    -------
    dict with keys: score (0-100 per-angle and composite), feedback list.
    """
    ranges = REFERENCE_RANGES.get(stroke_type, {})
    if not ranges:
        return {"composite": 50.0, "per_angle": {}, "feedback": ["No reference data for this stroke"]}

    per_angle = {}
    feedback = []
    total_score = 0.0
    n_scored = 0

    for angle in _ANGLES:
        val = angles.get(angle)
        if val is None:
            per_angle[angle] = None
            continue

        r = ranges.get(angle)
        if r is None:
            per_angle[angle] = None
            continue

        min_ok, max_ok, min_good, max_good = r
        n_scored += 1

        if min_good <= val <= max_good:
            score = 95.0
            per_angle[angle] = score
            total_score += score
        elif min_ok <= val <= max_ok:
            # Score proportional to distance from good range
            if val < min_good:
                frac = (val - min_ok) / max(min_good - min_ok, 1)
            else:
                frac = (max_ok - val) / max(max_ok - max_good, 1)
            score = 50.0 + 45.0 * max(0, min(1, frac))
            per_angle[angle] = score
            total_score += score
            feedback.append(f"{angle}: {val:.0f}° — outside ideal range ({min_good}-{max_good}°)")
        else:
            score = max(10.0, 50.0 - 40.0 * min(1, abs(val - min_ok) / max(min_ok, 1)))
            per_angle[angle] = score
            total_score += score
            feedback.append(f"{angle}: {val:.0f}° — outside acceptable range ({min_ok}-{max_ok}°)")

    composite = round(total_score / max(n_scored, 1), 1)
    return {"composite": composite, "per_angle": per_angle, "feedback": feedback}
