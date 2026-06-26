from app.shuttle_coach.engine import analyze_from_pipeline


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

    report = analyze_from_pipeline(analytics, shuttle_metrics={}, player_id="player_1")

    assert "strengths" in report
    assert "weaknesses" in report
    assert "top_3_improvements" in report
    assert "recommended_drills" in report
    assert "evidence" in report
    assert isinstance(report["evidence"], list)


def test_coach_no_evidence_without_metrics():
    analytics = {
        "fitness_analytics": {"player_1": {"rally_intensity": 0.5, "fatigue_trend": "stable"}},
        "tactical_analytics": {"player_1": {"shot_distribution": {}, "total_shots": 0}},
    }

    report = analyze_from_pipeline(analytics, shuttle_metrics={}, player_id="player_1")

    assert "evidence" in report
