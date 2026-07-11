import pandas as pd
import pytest

from scripts.evaluate_owner_attribution import compute_owner_metrics, recommend_deploy


def test_compute_owner_metrics_tracks_assigned_and_abstained_accuracy():
    matched = pd.DataFrame(
        {
            "label_side": ["near", "far", "near", "far"],
            "pred_side": ["near", "far", "unknown", "near"],
            "owner_source": ["local_anchor", "viterbi_bridge", "unknown", "local_anchor"],
        }
    )
    metrics = compute_owner_metrics(matched)
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["assigned_accuracy"] == pytest.approx(2 / 3)
    assert metrics["overall_accuracy"] == pytest.approx(0.5)
    assert metrics["abstention_rate"] == pytest.approx(0.25)


def test_recommendation_requires_accuracy_and_coverage_lift():
    recommendation = recommend_deploy(
        baseline={"assigned_accuracy": 0.70, "coverage": 0.60},
        candidate={"assigned_accuracy": 0.75, "coverage": 0.68},
        min_accuracy_lift=0.03,
        min_coverage_lift=0.05,
    )
    assert recommendation["deploy"] is True
