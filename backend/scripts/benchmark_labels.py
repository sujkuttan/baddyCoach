"""Labels benchmark harness (Task 0.1).

Loads a real pipeline ``shots`` parquet and the gold ``labels_enriched`` CSV,
merges them, and reports four metric groups:

  * temporal    - frame alignment error (frame_diff = label_frame - shot frame)
  * stroke      - exact + similar match rate, top confusions
  * attribution - committed-only (owner confident) and all (ignore uncertainty)
  * coverage    - bst_input_eligible fraction (matched + all) and recall

This script is fully self-contained: it does NOT import from
``evaluate_labels.py`` or any notebook. Only pandas, numpy and pyyaml are used.

Merge note
----------
The gold label CSV's ``shot_id`` column is empty for every row, so a direct
``shot_id`` join (as originally specced) is impossible. Instead we use the
repository's established temporal matching (see ``evaluate_labels.py``
``evaluate_enriched_csv`` / ``match_labels_to_shots``): each gold label is
matched to the nearest pipeline shot by |label_frame - shot.frame| within a
radius (default 15 frames), greedily (each shot used at most once). This is the
only defensible, non-guessed mapping given the data, and is documented in the
report output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --- Canonical internal column names ---------------------------------------
# The brief assumes pipeline columns named shot_frame / stroke / bst_eligible /
# bst_quality_score, but the real parquet uses frame / stroke_type /
# bst_input_eligible / bst_input_quality_score. We normalise on load so the
# harness tolerates both.
_COLUMN_RENAME = {
    "stroke": "stroke_type",
    "bst_eligible": "bst_eligible",
    "bst_input_eligible": "bst_eligible",
    "bst_quality_score": "bst_quality_score",
    "bst_input_quality_score": "bst_quality_score",
}

# Default embedded stroke-group map, used if the yaml cannot be found.
_DEFAULT_STROKE_GROUPS = {
    "clear": "clear_family",
    "lift": "clear_family",
    "defensive_lift": "clear_family",
    "smash": "attack_family",
    "rush": "attack_family",
    "drive": "flat_family",
    "push": "flat_family",
    "drop": "soft_family",
    "net_shot": "soft_family",
    "block": "block_family",
    "short_serve": "serve_family",
    "long_serve": "serve_family",
    "serve": "serve_family",
    "cross_court": "cross_family",
}

# Path to the shipped stroke_groups.yaml, relative to this script.
_STROKE_GROUPS_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "shuttle_coach"
    / "feedback"
    / "stroke_groups.yaml"
)


# --- Loading ----------------------------------------------------------------
def load_stroke_groups(path=None):
    """Load the stroke -> group map from yaml.

    Falls back to an embedded default map if the file is missing or cannot be
    parsed, so the harness (and its tests) never hard-depend on the file.
    """
    p = Path(path) if path else _STROKE_GROUPS_PATH
    if p and p.exists():
        try:
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
        except (yaml.YAMLError, OSError):
            pass
    return dict(_DEFAULT_STROKE_GROUPS)


def load_shots(path):
    """Load the pipeline shots parquet, normalising column names."""
    df = pd.read_parquet(path)
    df = df.rename(columns={k: v for k, v in _COLUMN_RENAME.items() if k in df.columns})
    if "frame" not in df.columns and "shot_frame" in df.columns:
        df = df.rename(columns={"shot_frame": "frame"})
    if "shot_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["shot_id"] = np.arange(1, len(df) + 1)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    return df


def load_labels(path):
    """Load the gold label CSV and keep only manually labelled rows."""
    df = pd.read_csv(path)
    if "label_status" in df.columns:
        df = df[df["label_status"].astype(str) == "labeled"].copy()
    df["label_frame"] = pd.to_numeric(df.get("label_frame"), errors="coerce")
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.strip().str.lower()
    if "true_stroke" in df.columns:
        df["true_stroke"] = df["true_stroke"].astype(str).str.strip()
    return df.reset_index(drop=True)


# --- Matching ---------------------------------------------------------------
def match_labels_to_shots(labels_df, shots_df, radius_frames=15):
    """Greedily match each label to the nearest pipeline shot by frame.

    Returns a list of dicts, one per label, with keys:
        label_idx, label_frame, side, true_stroke,
        shot_idx (or None), frame (or None), frame_diff (or None).
    Each pipeline shot is consumed by at most one label (greedy nearest).
    """
    shots = shots_df.sort_values("frame").reset_index(drop=True)
    used = set()
    matches = []
    for li, lab in labels_df.iterrows():
        lf = lab.get("label_frame")
        best_idx = None
        best_dist = np.inf
        if pd.notna(lf):
            lf = float(lf)
            for si in range(len(shots)):
                if si in used:
                    continue
                dist = abs(float(shots.loc[si, "frame"]) - lf)
                if dist < best_dist and dist <= radius_frames:
                    best_dist = dist
                    best_idx = si
        if best_idx is not None:
            used.add(best_idx)
            sframe = float(shots.loc[best_idx, "frame"])
            matches.append({
                "label_idx": li,
                "label_frame": lf,
                "side": lab.get("side"),
                "true_stroke": lab.get("true_stroke"),
                "shot_idx": int(best_idx),
                "frame": sframe,
                "frame_diff": lf - sframe,
            })
        else:
            matches.append({
                "label_idx": li,
                "label_frame": lf if pd.notna(lf) else None,
                "side": lab.get("side"),
                "true_stroke": lab.get("true_stroke"),
                "shot_idx": None,
                "frame": None,
                "frame_diff": None,
            })
    return matches


# --- Scoring ----------------------------------------------------------------
def _group_of(stroke, stroke_groups):
    if stroke is None:
        return None
    s = str(stroke).strip().lower()
    return stroke_groups.get(s, s)


def merge_and_score(shots_df, labels_df, stroke_groups=None, radius_frames=15):
    """Merge labels to shots and compute the four benchmark metric groups.

    Returns a dict with top-level keys: temporal, stroke, attribution, coverage,
    plus summary counts (n_labels, n_shots, n_matched, n_missed).
    """
    if stroke_groups is None:
        stroke_groups = load_stroke_groups()
    matches = match_labels_to_shots(labels_df, shots_df, radius_frames=radius_frames)

    n_labels = len(matches)
    n_matched = sum(1 for m in matches if m["shot_idx"] is not None)
    n_missed = n_labels - n_matched

    # Build augmented rows for matched shots.
    rows = []
    for m in matches:
        if m["shot_idx"] is None:
            continue
        s = shots_df.iloc[m["shot_idx"]]
        rows.append({
            "label_frame": m["label_frame"],
            "true_stroke": m["true_stroke"],
            "label_side": m["side"],
            "shot_frame": m["frame"],
            "frame_diff": m["frame_diff"],
            "pred_stroke": s.get("stroke_type"),
            "pred_side": s.get("side"),
            "owner_uncertain": bool(s.get("owner_uncertain")) if pd.notna(s.get("owner_uncertain")) else None,
            "bst_eligible": bool(s.get("bst_eligible")) if pd.notna(s.get("bst_eligible")) else False,
        })
    matched = pd.DataFrame(rows)

    # --- Temporal ---
    if n_matched:
        fd = matched["frame_diff"].astype(float)
        temporal = {
            "n": int(n_matched),
            "mean": float(fd.mean()),
            "median": float(fd.median()),
            "max": float(fd.max()),
            "mean_abs": float(fd.abs().mean()),
            "std": float(fd.std()) if n_matched > 1 else 0.0,
        }
    else:
        temporal = {"n": 0, "mean": 0.0, "median": 0.0, "max": 0.0, "mean_abs": 0.0, "std": 0.0}

    # --- Stroke ---
    exact = similar = 0
    confusions = Counter()
    if n_matched:
        for _, r in matched.iterrows():
            p = str(r["pred_stroke"]).strip().lower() if pd.notna(r["pred_stroke"]) else ""
            t = str(r["true_stroke"]).strip().lower() if pd.notna(r["true_stroke"]) else ""
            if p == t:
                exact += 1
                similar += 1
            elif _group_of(p, stroke_groups) is not None and _group_of(p, stroke_groups) == _group_of(t, stroke_groups):
                similar += 1
            else:
                confusions[(p, t)] += 1
    stroke = {
        "n": int(n_matched),
        "exact": int(exact),
        "similar": int(similar),
        "exact_rate": exact / n_matched if n_matched else 0.0,
        "similar_rate": similar / n_matched if n_matched else 0.0,
        "top_confusions": [
            {"pred": k[0], "true": k[1], "count": v}
            for k, v in confusions.most_common(10)
        ],
    }

    # --- Attribution ---
    # Treat a side of 'unknown' (or NaN) as missing: it carries no
    # attribution information and must not count as a wrong side.
    def _known_side(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        return str(v).strip().lower() not in ("", "nan", "none", "unknown")

    side_mask = matched["pred_side"].apply(_known_side) & matched["label_side"].apply(_known_side)
    side_rows = matched[side_mask]
    committed = side_rows[side_rows["owner_uncertain"] == False]  # noqa: E712
    attr = {
        "n_with_side": int(len(side_rows)),
        "committed_n": int(len(committed)),
        "committed_correct": int((committed["pred_side"] == committed["label_side"]).sum()) if len(committed) else 0,
        "committed_rate": float((committed["pred_side"] == committed["label_side"]).mean()) if len(committed) else 0.0,
        "all_n": int(len(side_rows)),
        "all_correct": int((side_rows["pred_side"] == side_rows["label_side"]).sum()) if len(side_rows) else 0,
        "all_rate": float((side_rows["pred_side"] == side_rows["label_side"]).mean()) if len(side_rows) else 0.0,
    }

    # --- Coverage / recall ---
    eligible_matched = int(matched["bst_eligible"].sum()) if n_matched else 0
    coverage = {
        "recall": n_matched / n_labels if n_labels else 0.0,
        "eligible_matched": eligible_matched,
        "eligible_matched_rate": eligible_matched / n_matched if n_matched else 0.0,
        "eligible_all_rate": eligible_matched / n_labels if n_labels else 0.0,
    }

    return {
        "n_labels": n_labels,
        "n_shots": int(len(shots_df)),
        "n_matched": n_matched,
        "n_missed": n_missed,
        "temporal": temporal,
        "stroke": stroke,
        "attribution": attr,
        "coverage": coverage,
    }


# --- Reporting --------------------------------------------------------------
def format_report(metrics, labels_path="?", shots_path="?"):
    out = []
    w = out.append
    w("=" * 70)
    w("  BADDYCOACH LABELS BENCHMARK REPORT (Task 0.1)")
    w("=" * 70)
    w(f"  Labels file: {labels_path}")
    w(f"  Shots file : {shots_path}")
    w("")
    w("  MERGE NOTE: label shot_id column is empty; matched by nearest")
    w("  pipeline frame within radius (temporal greedy match).")
    w("")
    w("  Coverage:")
    w(f"    Labels:        {metrics['n_labels']}")
    w(f"    Pipeline shots: {metrics['n_shots']}")
    w(f"    Matched:       {metrics['n_matched']} ({metrics['coverage']['recall']*100:.1f}% recall)")
    w(f"    Missed:        {metrics['n_missed']} (no nearby pipeline shot)")
    w("")
    w("  Temporal (frame_diff = label_frame - shot frame):")
    t = metrics["temporal"]
    w(f"    n={t['n']}  mean={t['mean']:.2f}  median={t['median']:.2f}  "
      f"max={t['max']:.2f}  mean|diff|={t['mean_abs']:.2f}  std={t['std']:.2f}")
    w("")
    w("  Stroke (exact + similar via stroke_groups.yaml):")
    s = metrics["stroke"]
    w(f"    n={s['n']}  exact={s['exact']} ({s['exact_rate']*100:.1f}%)  "
      f"similar={s['similar']} ({s['similar_rate']*100:.1f}%)")
    if s["top_confusions"]:
        w("    Top confusions (pred -> true : count):")
        for c in s["top_confusions"]:
            w(f"      {c['pred']} -> {c['true']} : {c['count']}")
    w("")
    w("  Attribution (side vs labeled side):")
    a = metrics["attribution"]
    w(f"    Committed-only (owner confident): {a['committed_correct']}/{a['committed_n']} "
      f"({a['committed_rate']*100:.1f}%)")
    w(f"    All (ignore uncertainty):         {a['all_correct']}/{a['all_n']} "
      f"({a['all_rate']*100:.1f}%)")
    w("")
    w("  BST input eligibility / coverage:")
    c = metrics["coverage"]
    w(f"    Eligible among matched: {c['eligible_matched']}/{metrics['n_matched']} "
      f"({c['eligible_matched_rate']*100:.1f}%)")
    w(f"    Eligible among all labels: {c['eligible_matched']}/{metrics['n_labels']} "
      f"({c['eligible_all_rate']*100:.1f}%)")
    w("=" * 70)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="BaddyCoach labels benchmark harness")
    parser.add_argument("--shots", default="results/hybrid_results/debug/shots.parquet",
                        help="Path to pipeline shots parquet")
    parser.add_argument("--labels", default="labels_enriched_new.csv",
                        help="Path to gold label CSV")
    parser.add_argument("--radius", type=int, default=15,
                        help="Max frame distance for label->shot match")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console report (still writes file)")
    args = parser.parse_args()

    shots_df = load_shots(args.shots)
    labels_df = load_labels(args.labels)
    stroke_groups = load_stroke_groups()
    metrics = merge_and_score(shots_df, labels_df, stroke_groups, radius_frames=args.radius)

    report = format_report(metrics, labels_path=args.labels, shots_path=args.shots)
    if not args.quiet:
        print(report)

    out_path = Path(args.shots).resolve().parent / "benchmark_report.txt"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    if not args.quiet:
        print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
