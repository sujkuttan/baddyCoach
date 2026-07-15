"""Tests for the labels benchmark harness (Task 0.1).

Constructs tiny synthetic shots/labels DataFrames (no real files, no network,
no model) and asserts the core merge/score functions produce correct counts
and rates. The numbers are hand-computed below so the assertions are not
coupled to the real run.
"""

import numpy as np
import pandas as pd

from scripts.benchmark_labels import (
    load_stroke_groups,
    load_shots,
    load_labels,
    match_labels_to_shots,
    merge_and_score,
)


# Stroke-group map used by the synthetic test. drive/push are similar.
_SYNTH_GROUPS = {
    "smash": "attack_family",
    "rush": "attack_family",
    "drive": "flat_family",
    "push": "flat_family",
    "clear": "clear_family",
    "lift": "clear_family",
    "drop": "soft_family",
    "net_shot": "soft_family",
    "block": "block_family",
    "cross_court": "cross_family",
}


def _make_shots():
    # Four pipeline shots using NORMALIZED column names (as produced by
    # load_shots): frame, stroke_type, side, owner_uncertain, bst_eligible,
    # bst_quality_score.
    return pd.DataFrame({
        "shot_id": [1, 2, 3, 4],
        "frame": [100, 200, 300, 400],
        "stroke_type": ["smash", "drive", "clear", "net_shot"],
        "side": ["far", "near", "far", "near"],
        "owner_uncertain": [False, True, False, False],
        "bst_eligible": [True, True, False, True],
        "bst_quality_score": [0.9, 0.8, 0.1, 0.7],
    })


def _make_labels():
    # Four gold labels, all status 'labeled'.
    return pd.DataFrame({
        "label_frame": [100, 200, 305, 500],
        "side": ["far", "near", "far", "near"],
        "true_stroke": ["smash", "push", "clear", "drop"],
        "true_class_id": [9, 6, 4, 5],
        "label_status": ["labeled", "labeled", "labeled", "labeled"],
    })


def test_match_labels_to_shots_basic():
    shots = _make_shots()
    labels = _make_labels()
    matches = match_labels_to_shots(labels, shots, radius_frames=15)
    # label@100->shot1, @200->shot2, @305->shot3, @500->missed (dist 100)
    assert [m["shot_idx"] for m in matches] == [0, 1, 2, None]
    assert matches[0]["frame_diff"] == 0.0
    assert matches[2]["frame_diff"] == 5.0  # 305 - 300


def test_merge_and_score_counts():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, _SYNTH_GROUPS, radius_frames=15)

    # Coverage / recall
    assert m["n_labels"] == 4
    assert m["n_shots"] == 4
    assert m["n_matched"] == 3
    assert m["n_missed"] == 1
    assert abs(m["coverage"]["recall"] - 0.75) < 1e-9

    # Temporal: frame_diffs = [0, 0, 5] -> mean 5/3, median 0, max 5
    assert m["temporal"]["n"] == 3
    assert abs(m["temporal"]["mean"] - (5.0 / 3.0)) < 1e-9
    assert m["temporal"]["median"] == 0.0
    assert m["temporal"]["max"] == 5.0
    assert abs(m["temporal"]["mean_abs"] - (5.0 / 3.0)) < 1e-9


def test_merge_and_score_stroke():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, _SYNTH_GROUPS, radius_frames=15)
    s = m["stroke"]
    # exact: smash==smash (label1), clear==clear (label3) => 2
    # similar: drive vs push (label2) same group => +1 => 3
    assert s["exact"] == 2
    assert s["similar"] == 3
    assert abs(s["exact_rate"] - 2 / 3) < 1e-9
    assert abs(s["similar_rate"] - 1.0) < 1e-9
    # Only confusion is drive -> push (label2 is similar, not a confusion)
    assert s["top_confusions"] == []


def test_merge_and_score_attribution():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, _SYNTH_GROUPS, radius_frames=15)
    a = m["attribution"]
    # All three matched shots have side matching the label.
    assert a["all_n"] == 3
    assert a["all_correct"] == 3
    assert abs(a["all_rate"] - 1.0) < 1e-9
    # Committed-only excludes shot2 (owner_uncertain=True).
    # Remaining committed: shot1 (far, match), shot3 (far, match) => 2/2
    assert a["committed_n"] == 2
    assert a["committed_correct"] == 2
    assert abs(a["committed_rate"] - 1.0) < 1e-9


def test_merge_and_score_coverage_eligibility():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, _SYNTH_GROUPS, radius_frames=15)
    c = m["coverage"]
    # eligible matched: shot1 True, shot2 True, shot3 False => 2/3
    assert c["eligible_matched"] == 2
    assert abs(c["eligible_matched_rate"] - 2 / 3) < 1e-9
    # eligible among all labels: 2/4
    assert abs(c["eligible_all_rate"] - 0.5) < 1e-9


def test_load_stroke_groups_fallback():
    # Missing file -> embedded default map (non-empty dict).
    groups = load_stroke_groups("/nonexistent/path.yaml")
    assert isinstance(groups, dict) and groups
    assert groups["smash"] == "attack_family"
    assert groups["drive"] == "flat_family"


def test_column_normalization(tmp_path):
    # load_shots should rename raw columns (bst_input_eligible /
    # bst_input_quality_score) into the canonical names used by scoring.
    raw = _make_shots().rename(columns={
        "bst_eligible": "bst_input_eligible",
        "bst_quality_score": "bst_input_quality_score",
    })
    p = tmp_path / "shots.parquet"
    raw.to_parquet(p)
    shots = load_shots(p)
    assert "bst_eligible" in shots.columns
    assert "bst_quality_score" in shots.columns
    assert "bst_input_eligible" not in shots.columns

    # load_labels keeps only labeled rows.
    labels2 = pd.concat([_make_labels(), pd.DataFrame([{
        "label_frame": 999, "side": "far", "true_stroke": "smash",
        "true_class_id": 9, "label_status": "unlabeled"}])], ignore_index=True)
    kept = load_labels_from_df(labels2)
    assert len(kept) == 4


def load_labels_from_df(df):
    # Helper mirroring load_labels' filter without re-reading a file.
    df = df[df["label_status"].astype(str) == "labeled"].copy()
    df["label_frame"] = pd.to_numeric(df.get("label_frame"), errors="coerce")
    df["side"] = df["side"].astype(str).str.strip().str.lower()
    df["true_stroke"] = df["true_stroke"].astype(str).str.strip()
    return df.reset_index(drop=True)
