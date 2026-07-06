#!/usr/bin/env python3
"""Grid-search ownership scorer weights against manual labels.

Usage:
    python backend/scripts/tune_ownership_weights.py \\
        --labels labels_enriched.csv \\
        --shots results/hybrid_results/debug/shots.parquet
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# ── Viterbi decoder (copied from ownership_scorer.py) ──────────────

def assign_owners_viterbi(n_shots: int, emissions: list[dict[str, float]],
                          p_alt: float = 0.95, p_same: float = 0.05,
                          eps: float = 1e-10) -> list[str]:
    states = ["near", "far"]
    log_alt = np.log(p_alt + eps)
    log_same = np.log(p_same + eps)
    dp, backptr = [], []
    for i in range(n_shots):
        row, bptr = {}, {}
        for s in states:
            log_emit = np.log(emissions[i][s] + eps)
            if i == 0:
                row[s] = log_emit
                bptr[s] = None
                continue
            best_score, best_prev = -1e18, None
            for ps in states:
                trans = log_alt if s != ps else log_same
                score = dp[i - 1][ps] + trans + log_emit
                if score > best_score:
                    best_score = score
                    best_prev = ps
            row[s] = best_score
            bptr[s] = best_prev
        dp.append(row)
        backptr.append(bptr)

    last = n_shots - 1
    final_state = max(dp[last], key=dp[last].get)
    owners = [final_state]
    for i in range(last, 0, -1):
        owners.append(backptr[i][owners[-1]])
    owners.reverse()
    return owners


# ── Grid search ───────────────────────────────────────────────────

# Default weights (from settings.py)
DEFAULT_WEIGHTS = {
    'trajectory': 0.35,
    'court_side': 0.20,
    'proximity': 0.15,
    'motion': 0.15,
    'pose': 0.10,
    'turn': 0.05,
    'bst': 0.06,
}

SUB_SCORE_KEYS = ['trajectory', 'court_side', 'proximity', 'motion', 'pose', 'turn', 'bst']

# Grid candidates — vary one weight at a time, scale others proportionally
GRID = {
    'trajectory': [0.10, 0.20, 0.35, 0.50, 0.70],
    'court_side': [0.05, 0.10, 0.20, 0.30, 0.40],
    'proximity':  [0.05, 0.10, 0.15, 0.25, 0.35],
    'motion':     [0.05, 0.10, 0.15, 0.25, 0.35],
    'pose':       [0.05, 0.10, 0.10, 0.20, 0.30],
    'turn':       [0.02, 0.05, 0.08, 0.12, 0.15],
    'bst':        [0.02, 0.06, 0.10, 0.15, 0.20],
}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def compute_emissions(shots: list[dict], weights: dict[str, float]) -> list[dict[str, float]]:
    """Compute near/far emissions from sub-scores with given weights."""
    emissions = []
    for s in shots:
        near_score = sum(weights[k] * s.get(f'ownership_{k}_near', 0.5) for k in SUB_SCORE_KEYS)
        far_score = sum(weights[k] * s.get(f'ownership_{k}_far', 0.5) for k in SUB_SCORE_KEYS)
        # Calibration z-score (mean/std from the data itself)
        emissions.append({"near": near_score, "far": far_score})
    return emissions


def main():
    parser = argparse.ArgumentParser(description="Tune ownership scorer weights")
    parser.add_argument("--labels", type=str, default="labels_enriched.csv")
    parser.add_argument("--shots", type=str,
                        default="results/hybrid_results/debug/shots.parquet")
    parser.add_argument("--max-frame-diff", type=int, default=15,
                        help="Max frame difference for label-shot matching (default: 15)")
    parser.add_argument("--min-labels-per-rally", type=int, default=0)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    shots_path = Path(args.shots)
    if not labels_path.exists() or not shots_path.exists():
        print("ERROR: labels or shots file not found")
        sys.exit(1)

    # ── 1. Load data ──────────────────────────────────────────────────
    df_labels = pd.read_csv(labels_path)
    df_labels = df_labels[df_labels["label_status"] == "labeled"]
    shots = pd.read_parquet(shots_path)

    print(f"Loaded {len(df_labels)} labels, {len(shots)} shots")

    # Match labels to shots by frame
    labeled_frames = df_labels[["label_frame", "side", "frame_diff"]].copy()
    labeled_frames = labeled_frames[labeled_frames["frame_diff"] <= args.max_frame_diff]
    print(f"Labels within {args.max_frame_diff} frames: {len(labeled_frames)}")

    # Also check label's "side" vs "predicted_class_id" to assign "true side"
    # But our enriched CSV already has the side from the label
    true_side_map = dict(zip(labeled_frames["label_frame"], labeled_frames["side"]))

    # Match labels to shots — for each label, find matching shot
    # Since enriched CSV already has the shot_frame, use that
    matched = 0
    rally_shots = {}  # rally_id -> list of shots with ground truth side
    for _, label in labeled_frames.iterrows():
        lf = label["label_frame"]
        # Find the shot with closest frame
        diffs = (shots["frame"] - lf).abs()
        best_pos = int(diffs.argmin())
        shot = shots.iloc[best_pos]
        rally_id = shot.get("rally_id")
        if pd.isna(rally_id):
            continue
        rally_id = int(rally_id)
        if rally_id not in rally_shots:
            rally_shots[rally_id] = []
        rally_shots[rally_id].append({
            "shot_idx": best_pos,
            "label_frame": int(lf),
            "label_side": label["side"],
            "frame_diff": label["frame_diff"],
        })
        matched += 1

    print(f"Matched {matched} labels to shots in {len(rally_shots)} rallies")

    if matched < 5:
        print("ERROR: too few matched labels")
        sys.exit(1)

    # ── 2. Build full per-rally shot sequences ─────────────────────────
    # For each rally with labels, get ALL shots in that rally
    rally_sequences = {}
    for rally_id in rally_shots:
        rshots = shots[shots["rally_id"] == rally_id].sort_values("frame")
        if len(rshots) < 2:
            continue
        rally_sequences[rally_id] = []
        for _, s in rshots.iterrows():
            entry = {
                "frame": int(s.get("frame", -1)),
                **{k: s.get(f"ownership_{k}_near", 0.5) for k in SUB_SCORE_KEYS},
            }
            for k in SUB_SCORE_KEYS:
                entry[f"{k}_far"] = s.get(f"ownership_{k}_far", 0.5)
            rally_sequences[rally_id].append(entry)

    print(f"Full rally sequences: {len(rally_sequences)} rallies")

    # ── 3. Grid search ────────────────────────────────────────────────
    results = []

    # Default first
    base_weights = normalize_weights(DEFAULT_WEIGHTS)
    default_acc = evaluate_weights(base_weights, rally_shots, rally_sequences)
    results.append({
        "changed": "default",
        "value": 0,
        "accuracy": default_acc,
        **{k: round(v, 3) for k, v in base_weights.items()},
    })
    print(f"Default accuracy: {default_acc:.1%}")

    # Grid: vary each weight, keep defaults for others with adjustment
    weight_names = list(GRID.keys())
    trial = 0
    for wi, wname in enumerate(weight_names):
        for wval in GRID[wname]:
            weights = DEFAULT_WEIGHTS.copy()
            weights[wname] = wval
            # Scale other weights proportionally so total stays close to 1.06
            other_sum = sum(DEFAULT_WEIGHTS[k] for k in weight_names if k != wname)
            new_other_sum = sum(weights[k] for k in weight_names if k != wname)
            if new_other_sum > 0 and other_sum > 0:
                scale = other_sum / new_other_sum
                for k in weight_names:
                    if k != wname:
                        weights[k] *= scale

            weights = normalize_weights(weights)
            acc = evaluate_weights(weights, rally_shots, rally_sequences)

            results.append({
                "changed": wname,
                "value": wval,
                "accuracy": acc,
                **{k: round(v, 3) for k, v in weights.items()},
            })
            trial += 1

    # ── 4. Report ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    best = df.loc[df["accuracy"].idxmax()]

    print()
    print("=" * 65)
    print("  OWNERSHIP WEIGHT GRID SEARCH — REPORT")
    print("=" * 65)
    print()
    print(f"  Default accuracy: {default_acc:.1%}")
    print()
    print("  Top-10 weight combinations:")
    print(f"  {'#':>3s} {'acc':>6s}  {'traj':>6s} {'court':>6s} {'prox':>6s} {'motion':>6s} {'pose':>6s} {'turn':>6s} {'bst':>6s}")
    print("  " + "-" * 58)
    top10 = df.nlargest(10, "accuracy")
    for i, (_, row) in enumerate(top10.iterrows()):
        print(f"  {i + 1:>3d} {row['accuracy']:>5.1%}  "
              f"{row.get('trajectory', 0):>6.3f} {row.get('court_side', 0):>6.3f} "
              f"{row.get('proximity', 0):>6.3f} {row.get('motion', 0):>6.3f} "
              f"{row.get('pose', 0):>6.3f} {row.get('turn', 0):>6.3f} "
              f"{row.get('bst', 0):>6.3f}")

    print()
    print(f"  Best weights: trajectory={best.get('trajectory', 0):.3f}, "
          f"court_side={best.get('court_side', 0):.3f}, "
          f"proximity={best.get('proximity', 0):.3f}, "
          f"motion={best.get('motion', 0):.3f}, "
          f"pose={best.get('pose', 0):.3f}, "
          f"turn={best.get('turn', 0):.3f}, "
          f"bst={best.get('bst', 0):.3f}")
    print(f"  Best accuracy: {best['accuracy']:.1%}")

    avg_acc = weighted_pooled_accuracy(df)
    print(f"\n  Weighted pooled accuracy: {avg_acc:.1%}")
    print("=" * 65)


def evaluate_weights(weights: dict[str, float],
                     rally_shots: dict,
                     rally_sequences: dict) -> float:
    """Evaluate a weight combination against ground truth labels."""
    correct = 0
    total = 0

    for rally_id, label_shots in rally_shots.items():
        if rally_id not in rally_sequences:
            continue

        seq = rally_sequences[rally_id]
        # Compute emissions from sub-scores
        emissions = []
        for s in seq:
            near_score = sum(weights[k] * s.get(k, 0.5) for k in SUB_SCORE_KEYS)
            far_score = sum(weights[k] * s.get(f"{k}_far", 0.5) for k in SUB_SCORE_KEYS)
            emissions.append({"near": near_score, "far": far_score})

        # Run Viterbi
        owners = assign_owners_viterbi(len(seq), emissions)

        # Match labeled shots to rally positions by frame proximity
        seq_frames = [s.get("frame", -1) for s in seq] if "frame" in seq[0] else list(range(len(seq)))
        for ls in label_shots:
            lf = ls.get("label_frame", ls.get("shot_idx", -1))
            # Find closest shot in rally by frame
            if "frame" in seq[0]:
                diffs = [abs(sf - lf) for sf in seq_frames]
                best_j = int(np.argmin(diffs))
            else:
                best_j = min(ls["shot_idx"], len(seq) - 1)

            if best_j < len(owners):
                correct += int(owners[best_j] == ls["label_side"])
                total += 1

    return correct / max(total, 1)


def weighted_pooled_accuracy(df: pd.DataFrame) -> float:
    """Compute accuracy weighted by how much better than default."""
    default = df[df["changed"].isna() | (df["changed"] == "default")]["accuracy"].max() if "default" in df["changed"].values else 0.37
    weights = np.maximum(df["accuracy"] - default, 0.001)
    return float(np.average(df["accuracy"], weights=weights))


if __name__ == "__main__":
    main()
