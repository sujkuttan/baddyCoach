import pytest

from app.shuttle_coach.narration.rag import retrieve_relevant_metrics
from app.shuttle_coach.narration.gemini import enforce_citations, format_metrics_for_rag


SAMPLE_METRICS = [
    {"metric_id": "movement.recovery_time", "player_id": "P1", "value": 0.92, "unit": "score", "context": {"avg_gap": "1.2s"}},
    {"metric_id": "movement.court_coverage", "player_id": "P1", "value": 0.85, "unit": "score", "context": {}},
    {"metric_id": "shots.effectiveness", "player_id": "P1", "value": {"smash": 0.45, "drop": 0.52}, "unit": "dict", "context": {}},
    {"metric_id": "errors.location_reason", "player_id": "P1", "value": {"unforced": 35.0, "forced": 65.0}, "unit": "pct", "context": {}},
    {"metric_id": "technique.backhand_clear", "player_id": "P2", "value": 0.7, "unit": "score", "context": {"quality": "good"}},
    {"metric_id": "movement.lateral_speed", "player_id": "P2", "value": 3.2, "unit": "m/s", "context": {}},
]


def test_retrieve_relevant_metrics():
    results = retrieve_relevant_metrics("What is the recovery time?", SAMPLE_METRICS, k=3)
    assert len(results) <= 3
    metric_ids = [r["metric_id"] for r in results]
    assert "movement.recovery_time" in metric_ids


def test_retrieve_relevant_metrics_empty():
    results = retrieve_relevant_metrics("random question xyz", [], k=5)
    assert results == []


def test_enforce_citations_valid():
    metrics = [{"metric_id": "movement.recovery_time"}, {"metric_id": "shots.effectiveness"}]
    text = "Player 1 has slow recovery [movement.recovery_time]. Their shot effectiveness is low [shots.effectiveness]."
    enforce_citations(text, metrics)


def test_enforce_citations_invalid():
    metrics = [{"metric_id": "movement.recovery_time"}]
    text = "Player 1 has slow recovery [errors.fictional_metric]."
    with pytest.raises(ValueError, match="unknown metrics"):
        enforce_citations(text, metrics)


def test_enforce_citations_uncited():
    metrics = [{"metric_id": "movement.recovery_time"}]
    text = "This sentence is long enough to trigger the uncited check and has no citation at all in it."
    with pytest.raises(ValueError, match="Ungrounded sentences"):
        enforce_citations(text, metrics)


def test_format_metrics_for_rag():
    result = format_metrics_for_rag(SAMPLE_METRICS, "recovery time")
    assert "movement.recovery_time" in result


def test_enforce_citations_empty_text():
    enforce_citations("Yes.", [{"metric_id": "x.y"}])
