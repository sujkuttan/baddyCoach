"""Tests for Task 2.2: full-rally Viterbi decode in assign_rally_owners."""

from types import SimpleNamespace

from app.config.settings import settings as global_settings
from app.pipeline.shared.ownership_quality import (
    assign_rally_owners,
    OwnerDecision,
)


def _settings(**overrides):
    base = dict(
        ownership_min_anchor_confidence=0.68,
        ownership_min_anchor_margin=0.18,
        ownership_min_anchor_signals=2,
        ownership_signal_neutral_epsilon=0.08,
        ownership_viterbi_bridge_enabled=True,
        ownership_viterbi_max_bridge_shots=2,
        ownership_viterbi_rally_enabled=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _score(near, far, conf=0.9, margin=None, signals=3):
    if margin is None:
        margin = abs(near - far)
    s = {
        "near_score": near,
        "far_score": far,
    }
    # independent-signal pairs consumed by count_independent_signals / is_anchor.
    # Always emit 5 strong independent signals so anchor-ness is driven purely
    # by the near/far score confidence + margin.
    sig_pairs = [
        ("trajectory_near", "trajectory_far"),
        ("court_side_near", "court_side_far"),
        ("proximity_near", "proximity_far"),
        ("motion_near", "motion_far"),
        ("pose_near", "pose_far"),
    ]
    for near_key, far_key in sig_pairs:
        s[near_key] = 0.9
        s[far_key] = 0.1
    return s


def _players():
    return {"near": "p1", "far": "p2"}


def test_emissions_favoring_one_side_still_alternate():
    n = 6
    scores = [_score(0.9, 0.1) for _ in range(n)]
    indices = list(range(n))
    decisions = assign_rally_owners(indices, scores, _players(), _settings())
    sides = [decisions[i].side for i in indices]
    for a, b in zip(sides, sides[1:]):
        assert a != b


def test_uniform_emissions_alternate_starting_near():
    n = 5
    scores = [_score(0.5, 0.5) for _ in range(n)]
    indices = list(range(n))
    decisions = assign_rally_owners(indices, scores, _players(), _settings())
    sides = [decisions[i].side for i in indices]
    assert sides[0] == "near"
    for a, b in zip(sides, sides[1:]):
        assert a != b


def test_flag_false_falls_back_to_anchor_logic():
    n = 5
    scores = [_score(0.9, 0.1) for _ in range(n)]
    indices = list(range(n))
    decisions = assign_rally_owners(
        indices, scores, _players(), _settings(ownership_viterbi_rally_enabled=False)
    )
    anchored = [i for i in indices if decisions[i].source == "local_anchor"]
    assert len(anchored) > 0
    for i in anchored:
        assert decisions[i].confident is True


def test_low_confidence_shot_not_anchor():
    near = _score(0.9, 0.1)
    low = _score(0.55, 0.45, conf=0.3, margin=0.1, signals=1)
    scores = [near, low, near, low, near]
    indices = list(range(len(scores)))
    decisions = assign_rally_owners(indices, scores, _players(), _settings())
    for i, s in enumerate(scores):
        if s is low:
            assert decisions[i].confident is False


def test_missing_score_falls_back_to_anchor():
    good = _score(0.9, 0.1)
    incomplete = {"near": 0.9}  # missing far_score
    scores = [good, incomplete, good, good, good]
    indices = list(range(len(scores)))
    decisions = assign_rally_owners(indices, scores, _players(), _settings())
    anchored = [i for i in indices if decisions[i].source == "local_anchor"]
    assert len(anchored) > 0


def test_viterbi_rally_source_set():
    scores = [_score(0.8, 0.2) for _ in range(4)]
    indices = list(range(len(scores)))
    decisions = assign_rally_owners(indices, scores, _players(), _settings())
    for i in indices:
        assert decisions[i].source == "viterbi_rally"
        assert decisions[i].reason == "full_rally_viterbi"
        assert decisions[i].side in ("near", "far")
