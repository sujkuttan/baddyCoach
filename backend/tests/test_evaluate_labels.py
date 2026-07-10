import pandas as pd

from scripts.evaluate_labels import summarize_bst_input_quality


def test_quality_summary_reports_accepted_accuracy_and_coverage():
    shots = pd.DataFrame({
        "stroke_type": ["smash", "drop", "lift"],
        "true_stroke": ["smash", "smash", "lift"],
        "bst_input_eligible": [True, True, False],
        "bst_input_quality_reasons": [[], [], ["long_shuttle_gap"]],
    })

    result = summarize_bst_input_quality(shots)

    assert result["total_labeled"] == 3
    assert result["eligible_labeled"] == 2
    assert result["coverage"] == 2 / 3
    assert result["accepted_accuracy"] == 0.5
    assert result["overall_accuracy"] == 2 / 3
    assert result["reason_counts"] == {"long_shuttle_gap": 1}
