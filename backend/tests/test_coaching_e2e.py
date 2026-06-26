"""End-to-end tests for the coaching engine — validates structure of
analyze_from_pipeline output including all new report sections."""

import pytest

from app.shuttle_coach.engine import analyze_from_pipeline


@pytest.fixture
def minimal_analytics():
    return {
        "tactical_analytics": {
            "player_1": {
                "shot_distribution": {"smash": 0.2, "clear": 0.3,
                                      "drop": 0.2, "net_shot": 0.15,
                                      "drive": 0.15},
                "total_shots": 40,
            }
        },
        "fitness_analytics": {
            "player_1": {
                "fatigue_trend": "stable",
                "avg_recovery": 1.2,
                "rally_intensity": 2.0,
            }
        },
        "footwork_analytics": {
            "player_1": {
                "avg_recovery": 1.2,
                "distance_covered": 1000,
            }
        },
        "technical_analytics": {
            "player_1": {
                "smash": {"avg_score": 0.7, "shot_count": 10},
                "clear": {"avg_score": 0.6, "shot_count": 8},
            }
        },
    }


@pytest.fixture
def sample_shuttle_metrics():
    return {
        "player_1": {
            "technique.reference": {
                "stroke": "smash", "feature": "elbow_extension",
                "current": 25.0, "ref_p50": 22.0, "percentile": 0.65, "n": 10,
            }
        }
    }


def test_coaching_report_structure(minimal_analytics):
    result = analyze_from_pipeline(minimal_analytics, shuttle_metrics={},
                                   player_id="player_1")
    assert "strengths" in result
    assert "weaknesses" in result
    assert "top_3_improvements" in result
    assert "recommended_drills" in result
    assert "recommended_drills_detailed" in result
    assert "evidence" in result
    assert "rally_stats" in result


def test_coaching_report_with_metrics(minimal_analytics, sample_shuttle_metrics):
    result = analyze_from_pipeline(minimal_analytics, sample_shuttle_metrics,
                                   player_id="player_1")
    assert len(result["evidence"]) >= 0
    assert isinstance(result["recommended_drills"], list)
    assert isinstance(result["recommended_drills_detailed"], list)


def test_coaching_report_quality_gating(minimal_analytics):
    quality = {
        "capability_trust": {
            "tactical": False,
            "movement": False,
            "technique": False,
            "patterns": False,
        }
    }
    result = analyze_from_pipeline(minimal_analytics, shuttle_metrics={},
                                   player_id="player_1", data_quality=quality)
    assert result["strengths"] is not None


def test_coaching_with_empty_analytics():
    result = analyze_from_pipeline({}, shuttle_metrics={}, player_id="player_1")
    assert result["strengths"] == []
    assert result["weaknesses"] == []
    assert result["recommended_drills"] == []


def test_coaching_with_progress_trends(minimal_analytics):
    quality = {"tier": "high", "quality_score": 0.9, "capability_trust": {
        "tactical": True, "movement": True, "technique": True, "patterns": True,
        "progress": True,
    }}
    result = analyze_from_pipeline(minimal_analytics, shuttle_metrics={},
                                   player_id="player_1", data_quality=quality)
    assert "recommended_drills_detailed" in result
