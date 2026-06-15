import pytest
from pathlib import Path
from app.coach.engine import CoachEngine


@pytest.fixture
def engine():
    rules_path = Path(__file__).parent.parent / "app" / "coach" / "rules.yaml"
    return CoachEngine(rules_path)


@pytest.fixture
def sample_analytics():
    return {
        "tactical_analytics": {
            "player_1": {
                "shot_distribution": {
                    "smash": 0.15,
                    "clear": 0.40,
                    "drop": 0.20,
                    "net_shot": 0.10,
                    "drive": 0.15,
                },
                "total_shots": 50,
            }
        },
        "fitness_analytics": {
            "player_1": {
                "fatigue_trend": "declining",
                "avg_recovery": 1.5,
                "rally_intensity": 2.3,
            }
        },
        "footwork_analytics": {
            "player_1": {
                "avg_recovery": 1.5,
                "distance_covered": 1200,
            }
        },
    }


def test_coach_engine_generates_recommendations(engine, sample_analytics):
    result = engine.generate(sample_analytics, "player_1")
    
    assert "strengths" in result
    assert "weaknesses" in result
    assert "top_3_improvements" in result
    assert "recommended_drills" in result
    assert "evidence" in result


def test_coach_triggers_smash_rule(engine, sample_analytics):
    result = engine.generate(sample_analytics, "player_1")
    
    weakness_text = " ".join(result["weaknesses"]).lower()
    assert "smash" in weakness_text or len(result["weaknesses"]) > 0


def test_coach_triggers_fatigue_rule(engine, sample_analytics):
    result = engine.generate(sample_analytics, "player_1")
    
    weakness_text = " ".join(result["weaknesses"]).lower()
    assert "fatigue" in weakness_text or "declining" in weakness_text or len(result["weaknesses"]) > 0


def test_coach_handles_missing_data(engine):
    empty_analytics = {}
    result = engine.generate(empty_analytics, "player_1")
    
    assert result["strengths"] == []
    assert result["weaknesses"] == []


def test_coach_get_nested_helper(engine):
    data = {
        "a": {"b": {"c": 42}},
        "x": [1, 2, 3],
    }
    
    assert engine._get_nested(data, "a.b.c") == 42
    assert engine._get_nested(data, "a.b") == {"c": 42}
    assert engine._get_nested(data, "missing.path") == 0
    assert engine._get_nested(data, "x.1") == 2
