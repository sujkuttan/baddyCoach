import pytest

from app.shuttle_coach.feedback.patterns import derive_pattern_findings
from app.shuttle_coach.metrics.base import MetricResult


def _make_result(metric_id, player_id, value, confidence=0.8):
    return MetricResult(
        metric_id=metric_id, player_id=player_id,
        value=value, unit="ratio", sample_size=10,
        confidence=confidence, context={},
    )


def test_pattern_conditional_outcome_finding():
    r = _make_result("patterns.conditional_outcome", "player_1", {
        "stroke": "smash", "zone": "rear_right", "pressed": True,
        "loss_rate": 0.7, "baseline_loss": 0.3,
        "n": 20, "wilson_loss_lb": 0.55,
    })
    findings = derive_pattern_findings([r], quality={"capability_trust": {"patterns": True}})
    assert len(findings) == 1
    assert "smash" in findings[0].detail


def test_pattern_transition_finding():
    r = _make_result("patterns.transition_outcome", "player_1", {
        "prev_stroke": "clear", "stroke": "drop",
        "loss_rate": 0.6, "baseline_loss": 0.3,
        "n": 15, "wilson_loss_lb": 0.55,
    })
    findings = derive_pattern_findings([r], quality={"capability_trust": {"patterns": True}})
    assert len(findings) == 1
    assert "clear" in findings[0].detail or "clear" in findings[0].headline


def test_patterns_suppressed_when_untrusted():
    r = _make_result("patterns.conditional_outcome", "player_1", {
        "stroke": "smash", "zone": "rear_right", "pressed": False,
        "loss_rate": 0.7, "baseline_loss": 0.3,
        "n": 20, "wilson_loss_lb": 0.55,
    })
    findings = derive_pattern_findings([r], quality={"capability_trust": {"patterns": False}})
    assert len(findings) == 0


def test_patterns_empty_when_below_floor():
    r = _make_result("patterns.conditional_outcome", "player_1", {
        "stroke": "drop", "zone": "net", "pressed": False,
        "loss_rate": 0.35, "baseline_loss": 0.30,
        "n": 10, "wilson_loss_lb": 0.35,
    })
    findings = derive_pattern_findings([r], quality={"capability_trust": {"patterns": True}})
    assert len(findings) == 0
