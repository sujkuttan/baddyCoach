"""Confusion-pair correction layer.

Resolves common pairwise BST ambiguities by checking physics features.
Runs after the hierarchical classifier, before the physics gate.
"""

import numpy as np
from app.pipeline.shared.physics import extract_physics_features, Features
from app.models.bst import COACH_STROKE_CLASSES

LOG = __import__("logging").getLogger("confusion_pairs")

_CLS_NAMES = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES


def _speed(feats: Features) -> float | None:
    """Get speed in m/s, falling back to speed_norm heuristic."""
    if feats.speed_mps is not None:
        return feats.speed_mps
    if feats.speed_norm is not None:
        return feats.speed_norm * 20.0
    return None


def _v_down(feats: Features) -> float | None:
    return feats.v_down


def _depth(feats: Features) -> str | None:
    return feats.depth


def _zone(feats: Features) -> str | None:
    return feats.zone


def _contact(feats: Features) -> str | None:
    return feats.contact


def _arc(feats: Features) -> bool | None:
    return feats.arc_rise_fall


# ── Pair rule functions ─────────────────────────────────────────
# Each returns 0 (boost first class in pair) or 1 (boost second class)
# when features disambiguate, or None when they don't.

def rule_clear_drop(feats: Features) -> int | None:
    """clear(0) vs drop(1)."""
    s = _speed(feats)
    d = _depth(feats)
    a = _arc(feats)
    if a is True and d == "deep":
        return 0  # boost clear
    if s is not None and s < 6.0 and d == "short":
        return 1  # boost drop
    return None


def rule_drop_smash(feats: Features) -> int | None:
    """drop(0) vs smash(1)."""
    s = _speed(feats)
    v = _v_down(feats)
    d = _depth(feats)
    if s is not None and s > 9.0 and v is not None and v > 2.0:
        return 1  # boost smash
    if s is not None and s < 6.0 and d == "short":
        return 0  # boost drop
    return None


def rule_lift_clear(feats: Features) -> int | None:
    """lift(0) vs clear(1)."""
    c = _contact(feats)
    d = _depth(feats)
    a = _arc(feats)
    if c in ("underarm", "low") and d == "deep":
        return 0  # boost lift
    if c == "overhead" and a is True:
        return 1  # boost clear
    return None


def rule_drive_block(feats: Features) -> int | None:
    """drive(0) vs block(1)."""
    s = _speed(feats)
    v = _v_down(feats)
    if s is not None and s > 7.0 and v is not None and abs(v) < 1.0:
        return 0  # boost drive (flat & fast)
    if s is not None and s < 5.0 and v is not None and v > 0.5:
        return 1  # boost block (slow & descending)
    return None


def rule_net_shot_push(feats: Features) -> int | None:
    """net_shot(0) vs push(1)."""
    s = _speed(feats)
    z = _zone(feats)
    c = _contact(feats)
    if z == "front" and c in ("underarm", "low") and s is not None and s < 3.0:
        return 0  # boost net_shot
    if z == "mid" and c in ("side", "underarm") and s is not None and 3.0 <= s <= 6.0:
        return 1  # boost push
    return None


def rule_short_serve_lift(feats: Features) -> int | None:
    """short_serve(0) vs lift(1)."""
    s = _speed(feats)
    z = _zone(feats)
    c = _contact(feats)
    if z == "front" and c == "low" and s is not None and s < 4.0:
        return 0  # boost short_serve
    if z == "back" and c == "underarm":
        return 1  # boost lift
    return None


# ── Confusion pair table ────────────────────────────────────────
# Each entry: (name_a, name_b, rule_fn)
# name_a is the first class in the pair, name_b is the second.
# rule_fn returns 0 to boost name_a, 1 to boost name_b, None to skip.

PAIR_RULES: list[tuple[str, str, callable]] = [
    ("clear",       "drop",        rule_clear_drop),
    ("drop",        "smash",       rule_drop_smash),
    ("lift",        "clear",       rule_lift_clear),
    ("drive",       "block",       rule_drive_block),
    ("net_shot",    "push",        rule_net_shot_push),
    ("short_serve", "lift",        rule_short_serve_lift),
]

# ── Resolution function ─────────────────────────────────────────

def _logits_from_probs(probs: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-10, 1.0 - 1e-10))
    logits -= logits.max(axis=1, keepdims=True)
    return logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _name(idx: int) -> str:
    """Coach-class name for column index (1-24). Returns '' for unknown."""
    if idx < 1 or idx > 24:
        return ""
    if idx <= 12:
        return COACH_STROKE_CLASSES[idx - 1]
    return COACH_STROKE_CLASSES[idx - 13]


def resolve_confusion_pairs(
    probs_matrix: np.ndarray,
    shots: list,
    shuttle_cleaned,
    shuttle_raw,
    pose_df,
    court: dict,
    fps: float,
    vid_w: float,
    vid_h: float,
    boost: float = 0.3,
) -> np.ndarray:
    """Apply confusion-pair corrections to BST probabilities.

    For each shot: extract top-2 classes, check if they form a known
    confusion pair, run the pair's rule function, and apply a logit
    boost to the resolved class.

    Args:
        probs_matrix: (N, 25) softmax probabilities.
        shots: List of shot dicts (must have 'frame').
        shuttle_cleaned, shuttle_raw, pose_df, court, fps, vid_w, vid_h: physics features.
        boost: Logit boost for the resolved class.

    Returns:
        Adjusted (N, 25) softmax probabilities.
    """
    n_shots = probs_matrix.shape[0]
    if n_shots == 0 or shuttle_cleaned is None or len(shuttle_cleaned) == 0:
        return probs_matrix

    logits = _logits_from_probs(probs_matrix)
    n_corrected = 0

    for i, shot in enumerate(shots):
        frame = shot["frame"]
        feats = extract_physics_features(
            frame, shuttle_cleaned, pose_df, "player_1",
            court, fps, vid_w, vid_h, shuttle_raw,
        )
        if not feats.usable:
            continue

        # Get top-2 class indices
        sort_idx = np.argsort(logits[i])[::-1]
        top0, top1 = int(sort_idx[0]), int(sort_idx[1])
        name0, name1 = _name(top0), _name(top1)

        # Check all pairs
        for name_a, name_b, rule_fn in PAIR_RULES:
            if {name0, name1} != {name_a, name_b}:
                continue
            result = rule_fn(feats)
            if result is None:
                continue
            # result=0 → boost name_a, result=1 → boost name_b
            target_name = name_a if result == 0 else name_b
            # Find the column index for this class (both players)
            for c in range(1, 25):
                if _name(c) == target_name:
                    logits[i, c] += boost
            n_corrected += 1
            break

    adjusted = _softmax(logits)

    # Update shots
    for i, shot in enumerate(shots):
        top_idx = int(np.argmax(adjusted[i]))
        if top_idx > 0:
            shot["stroke_type"] = _CLS_NAMES[top_idx]
            shot["stroke_confidence"] = float(adjusted[i, top_idx])

    if n_corrected > 0:
        LOG.info("Confusion-pair corrections applied", n=n_corrected)

    return adjusted
