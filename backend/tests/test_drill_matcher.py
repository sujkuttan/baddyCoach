import pytest

from app.shuttle_coach.feedback.drill_matcher import (
    _load_drills, _glob_match, _catalog_matching, _pick_level,
    select_drills, format_drill_flat,
)
from app.shuttle_coach.feedback.rules import Finding


def test_glob_match_exact():
    assert _glob_match("recovery_slow", "recovery_*") is True


def test_glob_match_wildcard():
    assert _glob_match("pattern::smash::rear_right::pressed", "pattern::*") is True


def test_glob_match_no_match():
    assert _glob_match("high_unforced", "technique_*") is False


def test_pick_level_high_severity():
    assert _pick_level(0.8, None) == "foundational"


def test_pick_level_low_severity():
    assert _pick_level(0.3, None) == "advanced"


def test_pick_level_declining():
    assert _pick_level(0.5, {"direction": "declining"}) == "foundational"


def test_pick_level_improving():
    assert _pick_level(0.5, {"direction": "improving"}) == "advanced"


def test_catalog_matching():
    drills = [
        {"id": "recovery_drill", "targets": ["recovery_*"]},
        {"id": "smash_drill", "targets": ["technique_smash*", "pattern::smash*"]},
    ]
    matches = _catalog_matching("recovery_slow", drills)
    assert len(matches) == 1
    assert matches[0]["id"] == "recovery_drill"


def test_select_drills_returns_top_n():
    findings = [
        Finding(code="recovery_slow", player_id="p1", severity=0.8,
                headline="Slow", detail="Slow recovery", evidence=[]),
        Finding(code="technique_smash", player_id="p1", severity=0.7,
                headline="Smash", detail="Weak smash", evidence=[]),
    ]
    drills = select_drills(findings, top_n=2)
    assert len(drills) <= 2


def test_format_drill_flat():
    drill = {"drill_id": "test", "name": "Test Drill", "dosage": "3x10",
             "rationale": "Because", "level": "intermediate",
             "success_criteria": "Hit 8/10", "focus": "accuracy",
             "linked_finding": "test_code"}
    flat = format_drill_flat(drill)
    assert "Test Drill" in flat
    assert "3x10" in flat
    assert "Because" in flat


def test_select_drills_dedup():
    findings = [
        Finding(code="recovery_slow", player_id="p1", severity=0.9,
                headline="Slow", detail="Slow", evidence=[]),
        Finding(code="recovery_slow", player_id="p2", severity=0.6,
                headline="Slow2", detail="Slow2", evidence=[]),
    ]
    drills = select_drills(findings, top_n=5)
    ids = [d["drill_id"] for d in drills]
    assert len(ids) == len(set(ids))
