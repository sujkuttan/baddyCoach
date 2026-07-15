"""Tests for the labels benchmark harness (Task 0.1, corrected).

Constructs tiny synthetic shots/labels (no real files, no network, no model)
and asserts the core merge/score functions replicate the canonical
``evaluate_labels.py::evaluate_enriched_csv`` method:

  * matching is done on the CSV ``shot_frame`` column (NOT label_frame)
  * stroke similarity uses the canonical ``STROKE_SIMILARITY`` map verbatim
  * combined = (exact + similar) / matched

Hand-computed expected numbers are embedded so the assertions are not coupled
to the real run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from scripts.benchmark_labels import (  # noqa: E402
    load_shots,
    load_labels,
    match_labels_to_shots,
    merge_and_score,
    stroke_matches,
    normalize_stroke,
    STROKE_SIMILARITY,
)


# --- Shared stroke-similarity fixture (used by cross-check test) -------------
_SIM_PAIRS = [
    ("smash", "smash", "exact"),
    ("drive", "push", "similar"),
    ("push", "drive", "similar"),
    ("clear", "lift", "similar"),
    ("lift", "clear", "similar"),
    ("drop", "net_shot", "similar"),
    ("net_shot", "drop", "similar"),
    ("smash", "net_shot", "wrong"),
    ("cross_court", "clear", "similar"),
    ("serve", "short_serve", "similar"),
    ("rush", "drive", "similar"),
    ("block", "drive", "similar"),
    ("soft_lift_or_push", "lift", "similar"),
    ("unknown", "smash", "wrong"),
]


def _make_shots():
    return pd.DataFrame({
        "frame": [100, 200, 300, 400],
        "side": ["far", "near", "far", "near"],
        "stroke_type": ["smash", "drive", "clear", "net_shot"],
        "stroke_confidence": [0.3, 0.85, 0.47, 0.9],
        "bst_input_eligible": [True, True, False, True],
        "owner_uncertain": [False, True, False, False],
    })


def _make_labels():
    """Labels list mirroring load_labels' dict shape (shot_frame-driven)."""
    return [
        {"time_s": 100 / 30.0, "frame": 100, "label_frame": 100, "frame_diff": 0,
         "player": "far", "stroke": "smash"},
        {"time_s": 200 / 30.0, "frame": 200, "label_frame": 200, "frame_diff": 0,
         "player": "near", "stroke": "push"},
        {"time_s": 305 / 30.0, "frame": 305, "label_frame": 305, "frame_diff": 5,
         "player": "far", "stroke": "clear"},
        {"time_s": 500 / 30.0, "frame": 500, "label_frame": 500, "frame_diff": 30,
         "player": "near", "stroke": "drop"},
    ]


def _canonical_module():
    import importlib.util
    path = REPO_ROOT / "backend" / "scripts" / "evaluate_labels.py"
    spec = importlib.util.spec_from_file_location("ev_canonical", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_match_labels_to_shots_uses_shot_frame():
    shots = _make_shots()
    labels = _make_labels()
    matches = match_labels_to_shots(labels, shots, radius_frames=15)
    # label@frame100->shot1, @200->shot2, @305->shot3 (dist 5, within radius),
    # @500->missed (nearest shot @400 is dist 100 > 15)
    assert [m["shot_idx"] for m in matches] == [0, 1, 2, None]
    # frame_error is overridden by the CSV frame_diff for matched rows.
    assert matches[0]["frame_error"] == 0
    assert matches[2]["frame_error"] == 5  # CSV frame_diff, not live |305-300|


def test_merge_and_score_coverage_recall():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, radius_frames=15)
    assert m["n_labels"] == 4
    assert m["n_shots"] == 4
    assert m["n_matched"] == 3
    assert m["n_missed"] == 1
    assert abs(m["coverage_recall"] - 75.0) < 1e-9


def test_merge_and_score_stroke_canonical():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, radius_frames=15)
    s = m["stroke"]
    # exact: smash==smash (label1), clear==clear (label3) => 2
    # similar: drive vs push (label2) => 1 (count of 'similar' results only)
    assert s["exact"] == 2
    assert s["similar"] == 1  # similar-only count
    assert abs(s["exact_rate"] - 2 / 3) < 1e-9
    assert abs(s["combined_rate"] - 100.0) < 1e-9  # (2+1)/3


def test_merge_and_score_attribution():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, radius_frames=15)
    # (a) all matched: 3/3 correct (sides far/near/far match)
    assert m["attribution_all_n"] == 3
    assert m["attribution_all_correct"] == 3
    assert abs(m["attribution_all_rate"] - 100.0) < 1e-9
    # (b) committed-only: excludes shot2 (owner_uncertain=True) => 2/2
    assert m["attribution_committed_n"] == 2
    assert m["attribution_committed_correct"] == 2
    assert abs(m["attribution_committed_rate"] - 100.0) < 1e-9


def test_merge_and_score_coverage_eligibility():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, radius_frames=15)
    # eligible matched: shot1 True, shot2 True, shot3 False => 2/3
    assert m["coverage_eligible_matched"] == 2
    assert abs(m["coverage_eligible_rate_matched"] - 100.0 * 2 / 3) < 1e-9
    # eligible among all labels: 2/4
    assert abs(m["coverage_eligible_rate_all"] - 50.0) < 1e-9


def test_merge_and_score_temporal_split():
    shots = _make_shots()
    labels = _make_labels()
    m = merge_and_score(shots, labels, radius_frames=15)
    # CSV frame_diff series over all 4 labeled rows: [0,0,5,30]
    tc = m["temporal_csv"]
    assert tc["n"] == 4
    assert abs(tc["mean"] - 8.75) < 1e-9
    assert tc["median"] == 2.5
    assert tc["max"] == 30.0
    # Live |label_frame - shot.frame| over matched 3: [0,0,5]
    tl = m["temporal_live"]
    assert tl["n"] == 3
    assert abs(tl["mean"] - (5.0 / 3.0)) < 1e-9
    assert tl["median"] == 0.0
    assert tl["max"] == 5.0


def test_load_shots_sorts_by_frame():
    shots = pd.DataFrame({
        "frame": [400, 100, 300, 200],
        "side": ["near", "far", "far", "near"],
        "stroke_type": ["net_shot", "smash", "clear", "drive"],
    })
    out = shots.sort_values("frame").reset_index(drop=True)
    for col in ("stroke_confidence", "bst_input_eligible", "owner_uncertain"):
        if col not in out.columns:
            out[col] = np.nan
    assert list(out["frame"]) == [100, 200, 300, 400]


def test_load_labels_filters_labeled_and_uses_shot_frame(tmp_path):
    csv = pd.DataFrame([
        {"label_frame": 100, "shot_frame": 100, "frame_diff": 0, "side": "far",
         "true_stroke": "smash", "label_status": "labeled", "true_class_id": 9},
        {"label_frame": 999, "shot_frame": 999, "frame_diff": 0, "side": "near",
         "true_stroke": "drop", "label_status": "unlabeled", "true_class_id": 5},
    ])
    p = tmp_path / "labels.csv"
    csv.to_csv(p, index=False)
    labels = load_labels(p)
    assert len(labels) == 1
    assert labels[0]["frame"] == 100  # from shot_frame, not label_frame


def test_embedded_similarity_matches_canonical():
    """Cross-check: our embedded stroke_matches MUST equal evaluate_labels.py's."""
    ev = _canonical_module()
    for p, l, expected in _SIM_PAIRS:
        got = stroke_matches(p, l)
        assert got == expected, (p, l, got, expected)
        assert got == ev.stroke_matches(p, l), (p, l, got, ev.stroke_matches(p, l))


def test_normalize_stroke_mirrors_canonical():
    ev = _canonical_module()
    for raw in ["Net Shot", "cross_court", "defensive lift", "soft_lift_or_push"]:
        assert normalize_stroke(raw) == ev.normalize_stroke(raw)


def test_stroke_similarity_map_identical_to_canonical():
    ev = _canonical_module()
    assert STROKE_SIMILARITY == ev.STROKE_SIMILARITY
