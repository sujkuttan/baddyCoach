from app.shuttle_coach.metrics.base import MetricResult
from app.shuttle_coach.feedback.rules import Finding, derive_findings
from app.shuttle_coach.feedback.prioritize import prioritize_findings


def _make_result(metric_id, player_id, value, unit="score", sample_size=10, confidence=1.0, context=None):
    return MetricResult(
        metric_id=metric_id,
        player_id=player_id,
        value=value,
        unit=unit,
        sample_size=sample_size,
        confidence=confidence,
        context=context or {},
    )


def test_derive_findings_slow_recovery():
    results = {
        "P1": [
            _make_result("movement.recovery_time", "P1", 0.92),
            _make_result("shots.effectiveness", "P1", {"smash": 0.45, "drop": 0.52}),
        ],
        "P2": [
            _make_result("movement.recovery_time", "P2", 0.5),
        ],
    }
    findings = derive_findings(results)
    codes = [f.code for f in findings]
    assert "slow_recovery" in codes
    player_ids = [f.player_id for f in findings if f.code == "slow_recovery"]
    assert "P1" in player_ids
    assert "P2" not in player_ids


def test_derive_findings_weak_shots():
    results = {
        "P1": [
            _make_result("shots.effectiveness", "P1", {"smash": 0.20, "drop": 0.50}),
        ],
    }
    findings = derive_findings(results)
    weak = [f for f in findings if f.code == "weak_shot"]
    assert len(weak) >= 1
    assert weak[0].player_id == "P1"


def test_derive_findings_high_unforced():
    results = {
        "P1": [
            _make_result("errors.location_reason", "P1", {"unforced": 35.0, "forced": 65.0}),
        ],
    }
    findings = derive_findings(results)
    unforced = [f for f in findings if f.code == "high_unforced_errors"]
    assert len(unforced) == 1
    assert unforced[0].player_id == "P1"
    assert unforced[0].severity > 0


def test_derive_findings_empty():
    findings = derive_findings({})
    assert findings == []


def test_prioritize_findings():
    f1 = Finding(code="a", player_id="P1", severity=0.3, headline="Low", detail="", evidence=[])
    f2 = Finding(code="b", player_id="P1", severity=0.9, headline="High", detail="", evidence=[])
    f3 = Finding(code="c", player_id="P1", severity=0.6, headline="Mid", detail="", evidence=[])
    result = prioritize_findings([f1, f2, f3])
    assert [f.code for f in result] == ["b", "c", "a"]


def test_prioritize_findings_empty():
    assert prioritize_findings([]) == []
