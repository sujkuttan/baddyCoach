"""Hierarchical family classifier on top of BST output.

Refines the 25-class BST probability matrix by applying a family-level
structural prior: aggregate per-shot probs by stroke family, select the
strongest family, then soft-penalize classes outside that family.

Runs after context fusion, before the physics gate.
"""

import numpy as np
from app.models.bst import COACH_STROKE_CLASSES

# ── Family definitions ──────────────────────────────────────────
# Each family groups coach stroke classes that share a biomechanical
# pattern. cross_court is ungrouped (exempt from penalty).

FAMILY_MAP = {
    "serve":       {"short_serve", "long_serve"},
    "overhead":    {"clear", "drop", "smash"},  # rush excluded — not a stroke
    "underhand":   {"lift"},
    "net":         {"net_shot", "push"},
    "drive_block": {"drive", "block"},
}

# Classes that appear in multiple families (union membership).
# push is valid in both "net" and "drive_block".
_MULTI_FAMILY = {"push": {"net", "drive_block"}}

# Build reverse lookup: coach_class → set of families
_CLASS_TO_FAMILIES: dict[str, set[str]] = {}
for family, classes in FAMILY_MAP.items():
    for c in classes:
        _CLASS_TO_FAMILIES.setdefault(c, set()).add(family)
for cls, families in _MULTI_FAMILY.items():
    _CLASS_TO_FAMILIES.setdefault(cls, set()).update(families)

# ── Per-shot scoring helpers ─────────────────────────────────────

def _stroke_name(idx: int) -> str:
    """Coach-class name for a given column index (1-24). Returns '' for unknown."""
    if idx < 1 or idx > 24:
        return ""
    if idx <= 12:
        return COACH_STROKE_CLASSES[idx - 1]
    return COACH_STROKE_CLASSES[idx - 13]


def _family_of(idx: int) -> set[str]:
    """Families for a given column index. Returns empty set for unknown."""
    name = _stroke_name(idx)
    if not name:
        return set()
    return _CLASS_TO_FAMILIES.get(name, set())


def _families_for_class(name: str) -> set[str]:
    return _CLASS_TO_FAMILIES.get(name, set())


def aggregate_probs_by_family(shot_probs: np.ndarray, cls_names: list[str]) -> dict[str, float]:
    """Sum per-class probabilities by family across all 25 classes.

    Args:
        shot_probs: (25,) probability vector.
        cls_names: 25-element list of class names (index 0 = "unknown").

    Returns:
        dict mapping family_name → total probability.
    """
    family_scores: dict[str, float] = {}
    for c in range(1, len(cls_names)):  # skip unknown
        name = cls_names[c]
        families = _families_for_class(name)
        for family in families:
            family_scores[family] = family_scores.get(family, 0.0) + float(shot_probs[c])
    # Fill missing families with 0
    for family in FAMILY_MAP:
        family_scores.setdefault(family, 0.0)
    return family_scores


def _soft_mask(
    shot_logits: np.ndarray,
    selected_family: str,
    cls_names: list[str],
    penalty: float,
) -> np.ndarray:
    """Apply soft penalty to classes outside the selected family.

    Unknown (index 0) and cross_court classes are never penalized.
    """
    for c in range(1, len(cls_names)):
        name = cls_names[c]
        families = _families_for_class(name)
        if selected_family in families:
            continue  # in-family → keep as-is
        if not families:
            continue  # cross_court / ungrouped → exempt
        shot_logits[c] -= penalty
    return shot_logits


def hierarchical_refine(
    probs_matrix: np.ndarray,
    penalty: float = 1.5,
) -> np.ndarray:
    """Apply family-level hierarchical refinement to a batch of BST probs.

    For each shot:
      1. Recover logits from softmax probabilities.
      2. Aggregate per-family probability.
      3. Select the family with the highest total probability.
      4. Soft-penalize (logit -= penalty) classes outside that family.
      5. Re-softmax.

    Args:
        probs_matrix: (N, 25) softmax probabilities.
        penalty: Logit penalty for out-of-family classes.

    Returns:
        (N, 25) adjusted softmax probabilities.
    """
    n_shots = probs_matrix.shape[0]
    cls_names = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES

    logits = np.log(np.clip(probs_matrix, 1e-10, 1.0 - 1e-10))
    logits -= logits.max(axis=1, keepdims=True)

    for i in range(n_shots):
        family_scores = aggregate_probs_by_family(probs_matrix[i], cls_names)
        selected = max(family_scores, key=family_scores.__getitem__)
        logits[i] = _soft_mask(logits[i], selected, cls_names, penalty)

    # Re-softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    adjusted = exp / exp.sum(axis=1, keepdims=True)

    return adjusted
