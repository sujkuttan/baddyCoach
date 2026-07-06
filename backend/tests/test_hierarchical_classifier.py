"""Tests for hierarchical family classifier."""

import numpy as np
from app.models.bst import COACH_STROKE_CLASSES


# ── Helpers ─────────────────────────────────────────────────────

_CLASS_NAMES = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES


def _idx(name: str, player: int = 1) -> int:
    """Column index for a coach class name. player=1 (far), player=2 (near)."""
    if name == "unknown":
        return 0
    base = COACH_STROKE_CLASSES.index(name)
    if player == 1:
        return 1 + base
    return 1 + len(COACH_STROKE_CLASSES) + base


# ── Family aggregation tests ────────────────────────────────────

def test_aggregate_overhead_summed_correctly():
    """Overhead family sums probs across all overhead classes."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[_idx("clear")] = 0.3
    probs[_idx("drop")] = 0.2
    probs[_idx("smash")] = 0.1
    # rush excluded — not a stroke type (movement to net)
    probs[_idx("net_shot")] = 0.15
    probs[_idx("drive")] = 0.05
    probs[0] = 0.20
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    assert abs(scores["overhead"] - 0.60) < 1e-6


def test_aggregate_net_summed_correctly():
    """Net family sums probs for net_shot and push."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[_idx("net_shot")] = 0.4
    probs[_idx("push")] = 0.15
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    assert abs(scores["net"] - 0.55) < 1e-6


def test_aggregate_unknown_ignored():
    """Unknown (index 0) contributes to no family."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[0] = 0.8
    probs[_idx("smash")] = 0.2
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    assert abs(scores["overhead"] - 0.2) < 1e-6
    assert scores.get("overhead", 0) < 0.8


def test_aggregate_cross_court_ungrouped():
    """Cross_court is ungrouped — never counted in any family."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[_idx("cross_court")] = 0.5
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    total_family = sum(scores.values())
    assert total_family < 1e-9  # cross_court is in no family


def test_aggregate_both_players():
    """Far and near player classes both contribute to the same family."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[_idx("smash", player=1)] = 0.3
    probs[_idx("smash", player=2)] = 0.2
    probs[_idx("clear", player=1)] = 0.1
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    assert abs(scores["overhead"] - 0.6) < 1e-6


# ── Family selection tests ──────────────────────────────────────

def test_select_overhead_when_overhead_dominant():
    """Overhead is selected when overhead family has highest total."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[_idx("smash")] = 0.5
    probs[_idx("net_shot")] = 0.2
    probs[0] = 0.3
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    selected = max(scores, key=scores.__getitem__)
    assert selected == "overhead"


def test_select_drive_block_when_drive_dominant():
    """Drive_block is selected when drive+block has highest total."""
    from app.pipeline.shared.hierarchical_classifier import aggregate_probs_by_family
    probs = np.zeros(25, dtype=np.float64)
    probs[_idx("drive")] = 0.35
    probs[_idx("block")] = 0.20
    probs[_idx("clear")] = 0.15
    probs[0] = 0.30
    scores = aggregate_probs_by_family(probs, _CLASS_NAMES)
    selected = max(scores, key=scores.__getitem__)
    assert selected == "drive_block"


# ── Soft mask tests ─────────────────────────────────────────────

def test_soft_mask_penalizes_out_of_family():
    """Out-of-family classes get logit -= penalty."""
    from app.pipeline.shared.hierarchical_classifier import _soft_mask
    logits = np.zeros(25, dtype=np.float64)
    masked = _soft_mask(logits.copy(), "overhead", _CLASS_NAMES, penalty=1.5)
    # net_shot is outside overhead
    assert masked[_idx("net_shot")] == -1.5
    # smash is inside overhead
    assert masked[_idx("smash")] == 0.0


def test_soft_mask_unknown_exempt():
    """Unknown (index 0) is never penalized."""
    from app.pipeline.shared.hierarchical_classifier import _soft_mask
    logits = np.zeros(25, dtype=np.float64)
    logits[0] = 5.0  # high unknown logit
    masked = _soft_mask(logits.copy(), "overhead", _CLASS_NAMES, penalty=5.0)
    assert masked[0] == 5.0  # unchanged


def test_soft_mask_cross_court_exempt():
    """Cross_court is ungrouped — never penalized."""
    from app.pipeline.shared.hierarchical_classifier import _soft_mask
    logits = np.zeros(25, dtype=np.float64)
    logits[_idx("cross_court")] = 3.0
    masked = _soft_mask(logits.copy(), "overhead", _CLASS_NAMES, penalty=5.0)
    assert masked[_idx("cross_court")] == 3.0  # unchanged


def test_soft_mask_multi_family_push():
    """Push belongs to net AND drive_block — passes through in either."""
    from app.pipeline.shared.hierarchical_classifier import _soft_mask
    logits = np.zeros(25, dtype=np.float64)
    logits[_idx("push")] = 2.0
    # Push is in net → should not be penalized when net is selected
    masked_net = _soft_mask(logits.copy(), "net", _CLASS_NAMES, penalty=5.0)
    assert masked_net[_idx("push")] == 2.0
    # Push is in drive_block → should not be penalized when drive_block is selected
    masked_db = _soft_mask(logits.copy(), "drive_block", _CLASS_NAMES, penalty=5.0)
    assert masked_db[_idx("push")] == 2.0
    # Push is NOT in overhead → should be penalized when overhead is selected
    masked_oh = _soft_mask(logits.copy(), "overhead", _CLASS_NAMES, penalty=5.0)
    assert masked_oh[_idx("push")] == -3.0


def test_soft_mask_both_players():
    """Both player-1 and player-2 classes are penalized."""
    from app.pipeline.shared.hierarchical_classifier import _soft_mask
    logits = np.zeros(25, dtype=np.float64)
    logits[_idx("block", player=1)] = 1.0
    logits[_idx("block", player=2)] = 2.0
    masked = _soft_mask(logits.copy(), "overhead", _CLASS_NAMES, penalty=1.5)
    assert masked[_idx("block", player=1)] == -0.5
    assert masked[_idx("block", player=2)] == 0.5


# ── Integration test ────────────────────────────────────────────

def test_hierarchical_refine_keeps_argmax_in_family():
    """After hierarchical refinement, the argmax should be in the selected family."""
    from app.pipeline.shared.hierarchical_classifier import hierarchical_refine
    # Smash (overhead) has 0.5, next best is net_shot (net) with 0.2
    probs = np.zeros((1, 25), dtype=np.float64)
    probs[0, _idx("smash")] = 0.5
    probs[0, _idx("net_shot")] = 0.2
    probs[0, _idx("block")] = 0.1
    probs[0, _idx("drive")] = 0.1
    probs[0, 0] = 0.1
    adjusted = hierarchical_refine(probs, penalty=2.0)
    top = int(np.argmax(adjusted[0]))
    # Top should be in overhead
    from app.pipeline.shared.hierarchical_classifier import _family_of
    assert "overhead" in _family_of(top), f"Top class {_CLASS_NAMES[top]} not in overhead"


def test_hierarchical_refine_preserves_unknown():
    """Unknown probability should not collapse to 0 after refinement."""
    from app.pipeline.shared.hierarchical_classifier import hierarchical_refine
    probs = np.ones((1, 25), dtype=np.float64) / 25.0
    adjusted = hierarchical_refine(probs, penalty=1.5)
    assert adjusted[0, 0] > 0.01  # unknown still has non-negligible prob


def test_hierarchical_refine_changes_distribution():
    """Penalized out-of-family probs should decrease."""
    from app.pipeline.shared.hierarchical_classifier import hierarchical_refine
    probs = np.zeros((1, 25), dtype=np.float64)
    probs[0, _idx("smash")] = 0.50
    probs[0, _idx("clear")] = 0.15  # overhead → safe
    probs[0, _idx("net_shot")] = 0.20  # net → will be penalized
    probs[0, _idx("block")] = 0.10  # drive_block → will be penalized
    probs[0, 0] = 0.05
    adjusted = hierarchical_refine(probs, penalty=3.0)
    # Net_shot should have lower prob than before
    assert adjusted[0, _idx("net_shot")] < 0.20
    # Overhead should be boosted
    assert adjusted[0, _idx("smash")] > 0.50


def test_refine_identity_at_uniform():
    """At perfectly uniform probs, refinement should still produce valid probs."""
    from app.pipeline.shared.hierarchical_classifier import hierarchical_refine
    probs = np.full((1, 25), 1.0 / 25.0)
    adjusted = hierarchical_refine(probs, penalty=1.5)
    assert adjusted.shape == (1, 25)
    assert abs(adjusted[0].sum() - 1.0) < 1e-6
    assert np.all(adjusted >= 0)
