#!/usr/bin/env python3
"""Match manual labels to pipeline shots and create enriched CSV with BST logits.

Usage:
    python backend/scripts/enrich_labels_with_logits.py \\
        --labels labels_import.csv \\
        --shots results/hybrid_results/debug/shots.parquet \\
        --bst-debug results/hybrid_results/debug/debug_bst_outputs.parquet \\
        --output labels_enriched.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COACH_CLASSES = [
    "net_shot", "block", "smash", "lift", "clear", "drive",
    "drop", "push", "rush", "cross_court", "short_serve", "long_serve",
]

SHUTTLESET_CLASSES = [
    "unknown", "Top_net_shot", "Top_block", "Top_smash", "Top_lift",
    "Top_clear", "Top_drive", "Top_drop", "Top_push", "Top_rush",
    "Top_cross_court", "Top_short_serve", "Top_long_serve",
    "Bottom_net_shot", "Bottom_block", "Bottom_smash", "Bottom_lift",
    "Bottom_clear", "Bottom_drive", "Bottom_drop", "Bottom_push",
    "Bottom_rush", "Bottom_cross_court", "Bottom_short_serve", "Bottom_long_serve",
]


def _stroke_to_shuttleset_id(stroke: str, side: str) -> int:
    """Map stroke name + side to ShuttleSet class ID (1-24)."""
    if stroke not in COACH_CLASSES:
        return 0
    class_idx = COACH_CLASSES.index(stroke)
    if side == "far":
        return class_idx + 1  # Top_* classes
    elif side == "near":
        return class_idx + 13  # Bottom_* classes
    return 0


def main():
    parser = argparse.ArgumentParser(description="Enrich manual labels with BST logits")
    parser.add_argument("--labels", type=str, default="labels_import.csv",
                        help="Manual labels CSV (default: labels_import.csv)")
    parser.add_argument("--shots", type=str,
                        default="results/hybrid_results/debug/shots.parquet",
                        help="Pipeline shots parquet")
    parser.add_argument("--bst-debug", type=str,
                        default="results/hybrid_results/debug/debug_bst_outputs.parquet",
                        help="BST debug outputs parquet")
    parser.add_argument("--output", type=str, default="labels_enriched.csv",
                        help="Output CSV path (default: labels_enriched.csv)")
    parser.add_argument("--max-frame-diff", type=int, default=90,
                        help="Max frame difference to accept match (default: 90)")
    args = parser.parse_args()

    # ── 1. Load inputs ──────────────────────────────────────────────────
    labels_path = Path(args.labels)
    shots_path = Path(args.shots)
    bst_path = Path(args.bst_debug)

    if not labels_path.exists():
        print(f"ERROR: labels CSV not found: {labels_path}")
        sys.exit(1)
    if not shots_path.exists():
        print(f"ERROR: shots parquet not found: {shots_path}")
        sys.exit(1)
    if not bst_path.exists():
        print(f"ERROR: BST debug parquet not found: {bst_path}")
        sys.exit(1)

    df_labels = pd.read_csv(labels_path)
    shots = pd.read_parquet(shots_path)
    bst_debug = pd.read_parquet(bst_path)

    # Filter to labeled rows only
    labeled = df_labels[df_labels["label_status"] == "labeled"].copy()
    if len(labeled) == 0:
        print("ERROR: No labeled rows found in CSV")
        sys.exit(1)
    print(f"Loaded {len(labeled)} labeled rows from {labels_path.name}")
    print(f"Loaded {len(shots)} pipeline shots, {len(bst_debug)} BST debug entries")

    # Detect bst_debug ↔ shots alignment offset
    # bst_debug may have extra rows at the end; first N should align positionally
    n_aligned = min(len(shots), len(bst_debug))
    print(f"  Using first {n_aligned} entries for alignment")

    # Build index: shot frame → bst_debug logits
    shot_frames = shots["frame"].values[:n_aligned]
    bst_logits = bst_debug["logits_all"].values[:n_aligned]
    bst_pred_class = bst_debug["pred_class_id"].values[:n_aligned]
    bst_confidence = bst_debug["pred_confidence"].values[:n_aligned]
    bst_is_rule_based = bst_debug["is_rule_based"].values[:n_aligned]

    # ── 2. Match each label to nearest shot ─────────────────────────────
    rows = []
    matched = 0
    for _, label in labeled.iterrows():
        lf = label["frame"]
        true_stroke = label.get("true_stroke", None)
        side = label.get("side", None)

        # Skip if missing stroke label
        if pd.isna(true_stroke) or not true_stroke:
            continue

        # Compute true ShuttleSet class ID
        true_class_id = _stroke_to_shuttleset_id(str(true_stroke).strip(), str(side).strip())
        if true_class_id == 0:
            print(f"  SKIP: unknown stroke '{true_stroke}' / side '{side}' at frame {lf}")
            continue

        # Find nearest shot by frame
        diffs = np.abs(shot_frames - lf)
        best_pos = int(diffs.argmin())
        frame_diff = int(diffs[best_pos])

        if frame_diff > args.max_frame_diff:
            continue  # too far — no corresponding pipeline shot

        # Extract logits
        logits_raw = bst_logits[best_pos]
        if pd.isna(logits_raw):
            continue

        if isinstance(logits_raw, str):
            logits_arr = json.loads(logits_raw)
        elif isinstance(logits_raw, (list, np.ndarray)):
            logits_arr = list(logits_raw)
        else:
            continue

        if len(logits_arr) != 25:
            continue

        matched += 1
        rows.append({
            "shot_id": label.get("shot_id", ""),
            "label_frame": int(lf),
            "shot_frame": int(shot_frames[best_pos]),
            "frame_diff": frame_diff,
            "side": str(side).strip(),
            "true_stroke": str(true_stroke).strip(),
            "true_class_id": int(true_class_id),
            "predicted_class_id": int(bst_pred_class[best_pos]),
            "predicted_confidence": float(bst_confidence[best_pos]),
            "is_rule_based": not pd.isna(bst_is_rule_based[best_pos]) and bool(bst_is_rule_based[best_pos]),
            "logits": json.dumps([float(v) for v in logits_arr]),
            "label_status": "labeled",
            "source": "manual",
        })

    if len(rows) == 0:
        print("ERROR: No label entries matched to pipeline shots")
        sys.exit(1)

    out = pd.DataFrame(rows)
    out = out.sort_values("label_frame").reset_index(drop=True)

    print(f"\nMatched {len(out)}/{matched} labels to pipeline shots")
    print(f"  Frame diff stats:")
    print(f"    min={out['frame_diff'].min()}, max={out['frame_diff'].max()}")
    print(f"    mean={out['frame_diff'].mean():.1f}, median={out['frame_diff'].median():.0f}")
    print(f"  Side balance: near={out['side'].eq('near').sum()}, far={out['side'].eq('far').sum()}")
    print(f"  Is rule-based: {out['is_rule_based'].sum()}/{len(out)}")

    # ── 3. Write output ─────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nWritten {len(out)} rows to {output_path}")


if __name__ == "__main__":
    main()
