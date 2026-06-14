from app.coach.engine import CoachEngine


def test_coach_generates_recommendations():
    analytics = {
        "fitness_analytics": {
            "player_1": {
                "rally_intensity": 0.3,
                "fatigue_trend": "declining",
                "avg_recovery": 1.5,
            }
        },
        "tactical_analytics": {
            "player_1": {
                "shot_distribution": {"smash": 0.1, "clear": 0.4, "drop": 0.3, "net_shot": 0.2},
                "total_shots": 50,
            }
        },
        "footwork_analytics": {
            "player_1": {
                "distance_covered": 800.0,
                "avg_recovery": 1.5,
            }
        },
    }

    engine = CoachEngine()
    report = engine.generate(analytics, player_id="player_1")

    assert "strengths" in report
    assert "weaknesses" in report
    assert "top_3_improvements" in report
    assert "recommended_drills" in report
    assert "evidence" in report
    assert isinstance(report["evidence"], list)
    assert all("finding" in e for e in report["evidence"])
    assert all("metrics" in e for e in report["evidence"])


def test_coach_no_evidence_without_metrics():
    analytics = {
        "fitness_analytics": {"player_1": {"rally_intensity": 0.5, "fatigue_trend": "stable"}},
        "tactical_analytics": {"player_1": {"shot_distribution": {}, "total_shots": 0}},
    }

    engine = CoachEngine()
    report = engine.generate(analytics, player_id="player_1")

    for evidence in report["evidence"]:
        assert len(evidence["metrics"]) > 0
