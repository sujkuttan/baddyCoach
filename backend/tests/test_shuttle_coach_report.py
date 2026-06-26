from app.shuttle_coach.feedback.rules import Finding
from app.shuttle_coach.feedback.report import render_report, render_report_json


def _finding(code="test", severity=0.8, headline="Test Headline", detail="Some detail", evidence=None):
    return Finding(
        code=code,
        player_id="P1",
        severity=severity,
        headline=headline,
        detail=detail,
        evidence=evidence or ["movement.recovery_time"],
    )


def test_render_report_markdown():
    findings = [
        _finding("slow_recovery", 0.85, "Slow Recovery", "Takes 2.1s to recover.", ["movement.recovery_time"]),
        _finding("weak_shot", 0.42, "Weak Smash", "Smash effectiveness 20%.", ["shots.effectiveness"]),
    ]
    md = render_report(findings)
    assert "# Coaching Report" in md
    assert "**Slow Recovery**" in md
    assert "Takes 2.1s" in md
    assert "movement.recovery_time" in md
    assert "0.42" in md


def test_render_report_markdown_top_k():
    findings = [_finding(severity=s) for s in [0.9, 0.7, 0.5, 0.3]]
    md = render_report(findings, top_k=2)
    assert md.count("**") >= 2


def test_render_report_markdown_empty():
    md = render_report([])
    assert "# Coaching Report" in md
    assert "No findings" in md


def test_render_report_json():
    findings = [_finding()]
    result = render_report_json(findings, player_ids=["P1", "P2"], capabilities={"movement", "shots"})
    assert "findings" in result
    assert result["player_ids"] == ["P1", "P2"]
    assert "movement" in result["capabilities"]
    assert len(result["findings"]) == 1
    assert result["findings"][0]["code"] == "test"


def test_render_report_json_empty():
    result = render_report_json([], player_ids=[], capabilities=set())
    assert result["findings"] == []
    assert result["player_ids"] == []
