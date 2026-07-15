"""Labels benchmark harness (Task 0.1) — canonical A/B replication.

This script REPLICATES ``backend/scripts/evaluate_labels.py::evaluate_enriched_csv``
so its baseline numbers match the repo's established A/B method
(``labels_enriched_new.csv`` vs ``results/hybrid_results/debug/shots.parquet``).

The labels were generated against an EARLIER pipeline run, so the canonical
method (matching on the CSV's stored ``shot_frame``) yields only ~50% recall
against the CURRENT ``shots.parquet``. The user-confirmed fair current-run
baseline matches on the human ground-truth ``label_frame`` instead, which
RE-ANCHORS the labels to the current run. That is the DEFAULT (``--match-key
label_frame``). Pass ``--match-key shot_frame`` to reproduce the canonical
evaluate_labels.py numbers (the CSV ``frame_diff`` series) for cross-checking.

The script is fully self-contained: it does NOT import from ``evaluate_labels.py``
except in a TEST-ONLY cross-check. The canonical ``STROKE_SIMILARITY`` map and
``stroke_matches`` / ``normalize_stroke`` are embedded VERBATIM below (clearly
marked) so the harness stays a standalone measurement tool.

Metric groups reported:
  * temporal    - in label_frame mode: label_frame - matched shot.frame (signed);
                  in shot_frame mode: stale CSV frame_diff (canonical)
  * stroke      - exact + similar match rate using canonical similarity map
  * attribution - all matched (matches evaluate_labels.py) + committed-only
  * coverage    - bst_input_eligible fraction (matched + all labels) and recall
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ===========================================================================
# Canonical stroke taxonomy — COPIED VERBATIM from
# backend/scripts/evaluate_labels.py (STROKE_SIMILARITY / normalize_stroke /
# stroke_matches). Do NOT edit independently; this mirrors the canonical method
# so the harness stays a self-contained replication of evaluate_enriched_csv.
# ===========================================================================
STROKE_SIMILARITY = {
    "lift": ["lift", "defensive_lift", "clear"],
    "clear": ["clear", "lift", "defensive_lift"],
    "drop": ["drop", "net_shot"],
    "net shot": ["net_shot", "drop", "push"],
    "netshot": ["net_shot", "drop", "push"],
    "drive": ["drive", "push", "rush"],
    "smash": ["smash", "rush", "drive"],
    "serve": ["short_serve", "long_serve", "serve"],
    "push": ["push", "drive", "net_shot"],
    "rush": ["rush", "drive", "smash", "push"],
    "block": ["block", "drive", "net_shot"],
    "defensive lift": ["lift", "defensive_lift", "clear"],
    "soft lift or push": ["soft_lift_or_push", "lift", "push"],
    "cross court": ["cross_court", "drive", "clear"],
}


def normalize_stroke(s: str) -> str:
    # Verbatim mirror of evaluate_labels.py::normalize_stroke
    return s.strip().lower().replace(" ", "").replace("_", "")


def stroke_matches(pipeline_stroke: str, label_stroke: str) -> str:
    """Check if pipeline stroke matches label stroke.

    Returns 'exact', 'similar', or 'wrong'. Copied VERBATIM from
    backend/scripts/evaluate_labels.py::stroke_matches.
    """
    p_norm = normalize_stroke(pipeline_stroke)
    l_norm = normalize_stroke(label_stroke)

    if p_norm == l_norm:
        return "exact"

    # Check similarity map
    for key, alternatives in STROKE_SIMILARITY.items():
        key_norm = normalize_stroke(key)
        alt_norm = [normalize_stroke(a) for a in alternatives]
        if l_norm == key_norm and p_norm in alt_norm:
            return "similar"
        if p_norm == key_norm and l_norm in alt_norm:
            return "similar"

    return "wrong"


# ===========================================================================
# Loading
# ===========================================================================
def load_shots(path):
    """Load the pipeline shots parquet, sorted by frame (mirrors
    ``load_pipeline_shots`` in evaluate_labels.py which sorts by frame)."""
    df = pd.read_parquet(path)
    # Normalise the eligibility column name if needed.
    if "bst_input_eligible" in df.columns:
        df = df.rename(columns={"bst_input_eligible": "bst_input_eligible"})
    df = df.sort_values("frame").reset_index(drop=True)
    # Ensure required columns exist for downstream access.
    for col in ("frame", "side", "stroke_type", "stroke_confidence",
                "bst_input_eligible", "owner_uncertain"):
        if col not in df.columns:
            df[col] = np.nan
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    return df


def load_labels(path, match_key="label_frame"):
    """Load the gold label CSV and keep only manually labelled rows.

    ``match_key`` selects which frame the label is matched against the
    CURRENT run's pipeline shots:

      * ``label_frame`` (DEFAULT) — the human ground-truth, video-anchored
        frame. This RE-ANCHORS the labels to the current run: labels were
        originally produced against an earlier pipeline run, so matching on
        their stored ``shot_frame`` is stale. Matching on ``label_frame`` is
        the fair current-run A/B baseline.
      * ``shot_frame`` — the pipeline frame recorded at labeling time. Use this
        to reproduce the canonical ``evaluate_labels.py`` numbers (which match
        on ``shot_frame``).
    """
    df = pd.read_csv(path)
    df = df[df["label_status"] == "labeled"].copy()
    df = df.reset_index(drop=True)

    labels = []
    for _, row in df.iterrows():
        label_frame = int(row["label_frame"]) if "label_frame" in row and not pd.isna(row.get("label_frame")) else None
        if match_key == "shot_frame":
            # Canonical: anchor on the recorded pipeline frame, fall back to label_frame.
            match_frame = int(row["shot_frame"]) if ("shot_frame" in row and not pd.isna(row.get("shot_frame"))) else label_frame
        else:
            # Default (label_frame): re-anchor labels to the current run.
            match_frame = label_frame if label_frame is not None else int(row["shot_frame"])
        if match_frame is None:
            continue
        labels.append({
            "time_s": match_frame / 30.0,
            "frame": int(match_frame),
            "label_frame": int(label_frame) if label_frame is not None else int(match_frame),
            "frame_diff": int(row["frame_diff"]) if "frame_diff" in row and not pd.isna(row.get("frame_diff")) else None,
            "player": str(row["side"]).strip(),
            "stroke": str(row["true_stroke"]).strip(),
        })
    return labels


# ===========================================================================
# Matching — greedy nearest, each pipeline shot used at most once.
# Mirrors evaluate_labels.py::match_labels_to_shots exactly.
# ===========================================================================
def match_labels_to_shots(labels, shots, radius_frames=15):
    shots_unused = set(shots.index)
    matches = []

    for lab in labels:
        best_idx = None
        best_dist = float("inf")
        for idx in shots_unused:
            dist = abs(shots.loc[idx, "frame"] - lab["frame"])
            if dist < best_dist and dist <= radius_frames:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            matches.append({
                "label": lab,
                "shot_idx": best_idx,
                "shot": shots.loc[best_idx].to_dict(),
                "frame_error": int(best_dist),
            })
            shots_unused.remove(best_idx)
        else:
            matches.append({
                "label": lab,
                "shot_idx": None,
                "shot": None,
                "frame_error": None,
            })
    return matches


# ===========================================================================
# Scoring — mirrors evaluate_enriched_csv / compute_metrics / summarize.
# ===========================================================================
def merge_and_score(shots_df, labels, radius_frames=15, match_key="label_frame"):
    matches = match_labels_to_shots(labels, shots_df, radius_frames=radius_frames)

    # Frame-error semantics depend on the match key:
    #   * shot_frame mode: use the stale CSV ``frame_diff`` (canonical
    #     evaluate_labels.py behaviour, mirrors evaluate_enriched_csv).
    #   * label_frame mode: temporal error = label_frame - matched shot.frame
    #     (signed). The CSV frame_diff is STALE here and must NOT be used.
    for i, m in enumerate(matches):
        if m["shot_idx"] is not None:
            if match_key == "shot_frame" and labels[i]["frame_diff"] is not None:
                m["frame_error"] = labels[i]["frame_diff"]
            elif match_key == "label_frame":
                m["frame_error"] = int(m["label"]["label_frame"] - m["shot"]["frame"])

    n_labels = len(labels)
    n_shots = len(shots_df)
    n_matched = sum(1 for m in matches if m["shot_idx"] is not None)
    n_missed = n_labels - n_matched

    # Per-shot stroke / player / frame results (mirrors compute_metrics).
    stroke_results = []
    player_results = []
    frame_errors = []
    live_frame_errors = []  # |label_frame - matched shot.frame| (label_frame mode)
    matched_rows = []
    for m in matches:
        if m["shot"] is None:
            stroke_results.append("missed")
            player_results.append("missed")
            continue
        s_match = stroke_matches(m["shot"]["stroke_type"], m["label"]["stroke"])
        stroke_results.append(s_match)
        p_match = "correct" if m["shot"]["side"] == m["label"]["player"].lower() else "wrong"
        player_results.append(p_match)
        frame_errors.append(m["frame_error"])
        live_frame_errors.append(abs(m["label"]["label_frame"] - m["shot"]["frame"]))

        row = dict(m["shot"])
        row["true_stroke"] = m["label"]["stroke"]
        row["label_player"] = m["label"]["player"].lower()
        row["label_frame"] = m["label"]["label_frame"]
        matched_rows.append(row)

    stroke_exact = sum(1 for s in stroke_results if s == "exact")
    stroke_similar = sum(1 for s in stroke_results if s == "similar")
    stroke_correct = stroke_exact + stroke_similar
    player_correct = sum(1 for p in player_results if p == "correct")

    # --- Temporal: the frame_error series IS the mode-appropriate temporal error.
    # In label_frame mode it is (label_frame - matched shot.frame) signed; in
    # shot_frame mode it is the CSV frame_diff. This is the primary temporal
    # metric reported under "Temporal Alignment". ---
    if frame_errors:
        temporal = {
            "n": int(len(frame_errors)),
            "mean": float(np.mean(frame_errors)),
            "median": float(np.median(frame_errors)),
            "max": float(np.max(frame_errors)),
            "mean_abs": float(np.mean(np.abs(frame_errors))),
            "median_abs": float(np.median(np.abs(frame_errors))),
        }
    else:
        temporal = {"n": 0, "mean": 0.0, "median": 0.0, "max": 0.0,
                    "mean_abs": 0.0, "median_abs": 0.0}

    # --- Temporal (secondary): CSV frame_diff series (all labeled rows, stale) ---
    csv_frame_diffs = [lab["frame_diff"] for lab in labels if lab["frame_diff"] is not None]
    if csv_frame_diffs:
        temporal_csv = {
            "n": int(len(csv_frame_diffs)),
            "mean": float(np.mean(csv_frame_diffs)),
            "median": float(np.median(csv_frame_diffs)),
            "max": float(np.max(csv_frame_diffs)),
            "mean_abs": float(np.mean(np.abs(csv_frame_diffs))),
        }
    else:
        temporal_csv = {"n": 0, "mean": 0.0, "median": 0.0, "max": 0.0, "mean_abs": 0.0}

    stroke = {
        "n": int(n_matched),
        "exact": int(stroke_exact),
        "similar": int(stroke_similar),
        "exact_rate": stroke_exact / n_matched if n_matched else 0.0,
        "similar_rate": (stroke_exact + stroke_similar) / n_matched * 100 if n_matched else 0.0,
        "combined_rate": (stroke_exact + stroke_similar) / n_matched * 100 if n_matched else 0.0,
    }

    # --- Attribution (a): all matched (mirrors compute_metrics player_accuracy) ---
    attr_all_n = n_matched
    attr_all_correct = player_correct
    attr_all_rate = player_correct / n_matched * 100 if n_matched else 0.0

    # --- Attribution (b): committed-only (non-null side AND owner_uncertain==False) ---
    committed = []
    for m in matches:
        if m["shot"] is None:
            continue
        side = m["shot"]["side"]
        ou = m["shot"]["owner_uncertain"]
        if side is None or (isinstance(side, float) and np.isnan(side)):
            continue
        if str(side).strip().lower() in ("", "nan", "none", "unknown"):
            continue
        if ou is None or (isinstance(ou, float) and np.isnan(ou)) or ou:
            continue
        committed.append(m)
    committed_n = len(committed)
    committed_correct = sum(
        1 for m in committed if m["shot"]["side"] == m["label"]["player"].lower()
    )
    committed_rate = committed_correct / committed_n * 100 if committed_n else 0.0

    # --- Coverage / recall ---
    recall = n_matched / n_labels * 100 if n_labels else 0.0
    eligible_matched = sum(
        1 for r in matched_rows if bool(r.get("bst_input_eligible", False))
    )
    eligible_rate_matched = eligible_matched / n_matched * 100 if n_matched else 0.0
    eligible_rate_all = eligible_matched / n_labels * 100 if n_labels else 0.0

    return {
        "n_labels": int(n_labels),
        "n_shots": int(n_shots),
        "n_matched": int(n_matched),
        "n_missed": int(n_missed),
        "temporal": temporal,
        "temporal_csv": temporal_csv,
        "stroke": stroke,
        "attribution_all_correct": int(attr_all_correct),
        "attribution_all_n": int(attr_all_n),
        "attribution_all_rate": float(attr_all_rate),
        "attribution_committed_correct": int(committed_correct),
        "attribution_committed_n": int(committed_n),
        "attribution_committed_rate": float(committed_rate),
        "coverage_recall": float(recall),
        "coverage_eligible_matched": int(eligible_matched),
        "coverage_eligible_rate_matched": float(eligible_rate_matched),
        "coverage_eligible_rate_all": float(eligible_rate_all),
        "matches": matches,
    }


# ===========================================================================
# Reporting
# ===========================================================================
def format_report(metrics, labels_path="?", shots_path="?", match_key="label_frame"):
    out = []
    w = out.append
    w("=" * 70)
    w("  BADDYCOACH LABELS BENCHMARK REPORT (Task 0.1)")
    w(f"  [match key = {match_key}]")
    w("=" * 70)
    w(f"  Labels file: {labels_path}")
    w(f"  Shots file : {shots_path}")
    w("")
    w("  Coverage / Recall:")
    w(f"    Labels:          {metrics['n_labels']}")
    w(f"    Pipeline shots:  {metrics['n_shots']}")
    w(f"    Matched:         {metrics['n_matched']} ({metrics['coverage_recall']:.1f}% recall)")
    w(f"    Missed:          {metrics['n_missed']} (label has no nearby pipeline shot)")
    w("")
    if match_key == "label_frame":
        w("  Temporal Alignment (label_frame - matched shot.frame, signed; CSV frame_diff ignored):")
    else:
        w("  Temporal Alignment (CSV frame_diff = label_frame - shot_frame, canonical):")
    t = metrics["temporal"]
    w(f"    n={t['n']}  mean={t['mean']:.2f}  median(signed)={t['median']:.2f}  "
      f"max={t['max']:.2f}  mean|diff|={t['mean_abs']:.2f}  median|diff|={t['median_abs']:.2f}")
    w("")
    w("  Stroke Accuracy (exact + similar via canonical STROKE_SIMILARITY):")
    s = metrics["stroke"]
    w(f"    n={s['n']}  exact={s['exact']}  similar={s['similar']}")
    w(f"    Exact rate:   {s['exact_rate']*100:.1f}%")
    w(f"    Similar rate: +{(s['similar'] - s['exact'])/s['n']*100 if s['n'] else 0:.1f}%")
    w(f"    Combined:     {s['combined_rate']:.1f}%  (=(exact+similar)/matched)")
    w("")
    w("  Player Attribution (side vs labeled side):")
    w(f"    (a) All matched (matches evaluate_labels.py): "
      f"{metrics['attribution_all_correct']}/{metrics['attribution_all_n']} "
      f"({metrics['attribution_all_rate']:.1f}%)")
    w(f"    (b) Committed-only (non-null side AND owner_uncertain==False): "
      f"{metrics['attribution_committed_correct']}/{metrics['attribution_committed_n']} "
      f"({metrics['attribution_committed_rate']:.1f}%)")
    w("")
    w("  BST Input Eligibility / Coverage:")
    w(f"    Eligible among matched: {metrics['coverage_eligible_matched']}/{metrics['n_matched']} "
      f"({metrics['coverage_eligible_rate_matched']:.1f}%)")
    w(f"    Eligible among all labels: {metrics['coverage_eligible_matched']}/{metrics['n_labels']} "
      f"({metrics['coverage_eligible_rate_all']:.1f}%)")
    w("=" * 70)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="BaddyCoach labels benchmark harness (canonical A/B)")
    parser.add_argument("--shots", default="results/hybrid_results/debug/shots.parquet",
                        help="Path to pipeline shots parquet")
    parser.add_argument("--labels", default="labels_enriched_new.csv",
                        help="Path to gold label CSV")
    parser.add_argument("--radius", type=int, default=15,
                        help="Max frame distance for label->shot match")
    parser.add_argument("--match-key", choices=["label_frame", "shot_frame"],
                        default="label_frame",
                        help="Frame to match labels against the current run. "
                             "label_frame (default) re-anchors labels to the current "
                             "run via the human ground-truth frame; shot_frame "
                             "reproduces the canonical evaluate_labels.py numbers.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console report (still writes file)")
    args = parser.parse_args()

    shots_df = load_shots(args.shots)
    labels = load_labels(args.labels, match_key=args.match_key)
    metrics = merge_and_score(shots_df, labels, radius_frames=args.radius,
                              match_key=args.match_key)

    report = format_report(metrics, labels_path=args.labels, shots_path=args.shots,
                           match_key=args.match_key)
    if not args.quiet:
        print(report)

    out_path = Path(args.shots).resolve().parent / "benchmark_report.txt"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    if not args.quiet:
        print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
