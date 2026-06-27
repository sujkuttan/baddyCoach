#!/usr/bin/env python3
"""Pre-process labels CSV for BST temperature calibration.

Fills missing side from player_id, derives true_class_id from
true_stroke+side, merges logits_all from debug_bst_outputs.parquet,
and writes an enriched CSV that calibrate_bst.py can consume.

Usage:
    python scripts/prepare_calibration.py \
        --labels labels_import.csv \
        --debug path/to/debug_bst_outputs.parquet \
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

PLAYER_SIDE = {"player_1": "near", "player_2": "far"}


def shuttleset_id(stroke: str, side: str) -> int:
    """Map (coach_stroke, side) -> ShuttleSet class ID (0-24)."""
    if not stroke or stroke == "unknown":
        return 0
    idx = COACH_CLASSES.index(stroke) if stroke in COACH_CLASSES else -1
    if idx < 0:
        return 0
    if side == "far":
        return idx + 1   # Top_* range: 1-12
    return idx + 13       # Bottom_* range: 13-24


def main():
    parser = argparse.ArgumentParser(description="Pre-process labels CSV")
    parser.add_argument("--labels", required=True, help="Path to labels CSV")
    parser.add_argument("--debug", required=True, help="Path to debug_bst_outputs.parquet")
    parser.add_argument("--output", default="labels_enriched.csv", help="Output path")
    parser.add_argument("--no-side-fill", action="store_true", help="Skip filling missing side from player_id")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found")
        sys.exit(1)

    debug_path = Path(args.debug)
    if not debug_path.exists():
        print(f"ERROR: {debug_path} not found")
        sys.exit(1)

    # ── Read inputs ──────────────────────────────────────────────────
    df = pd.read_csv(labels_path)
    debug_df = pd.read_parquet(debug_path)

    print(f"Labels CSV: {len(df)} rows")
    print(f"Debug parquet: {len(debug_df)} rows, columns: {list(debug_df.columns)}")

    # ── Fill missing side from player_id ─────────────────────────────
    if not args.no_side_fill:
        n_side_before = df["side"].notna().sum()
        df["side"] = df["side"].fillna(df["player_id"].map(PLAYER_SIDE))
        n_side_after = df["side"].notna().sum()
        print(f"Side filled: {n_side_before} -> {n_side_after}")

    # ── Derive true_class_id from true_stroke + side ─────────────────
    n_cid_before = df["true_class_id"].notna().sum()
    for idx, row in df.iterrows():
        if pd.isna(row.get("true_class_id")) or not str(row["true_class_id"]).strip():
            stroke = str(row.get("true_stroke", "")).strip() if pd.notna(row.get("true_stroke")) else ""
            side = str(row.get("side", "")).strip() if pd.notna(row.get("side")) else ""
            derived = shuttleset_id(stroke, side)
            df.at[idx, "true_class_id"] = derived
    n_cid_after = df["true_class_id"].notna().sum()
    print(f"true_class_id filled: {n_cid_before} -> {n_cid_after}")

    # ── Merge logits_all from debug parquet ───────────────────────────
    # Both files have the same number of rows in the same order (shot_id order).
    # Join by position to avoid index alignment issues with filtered subsets.
    if "logits_all" in debug_df.columns:
        df["logits"] = debug_df["logits_all"].values
        n_logits = df["logits"].notna().sum()
        print(f"Logits merged: {n_logits}/{len(df)} rows")

        # For non-labeled rows, set logits to empty string to avoid confusion
        labeled_mask = df["label_status"] == "labeled"
        n_labeled_with_logits = labeled_mask.sum() if "logits" not in df.columns else df.loc[labeled_mask, "logits"].notna().sum()
        print(f"Labeled rows with logits: {n_labeled_with_logits}/{labeled_mask.sum()}")

    # ── Write enriched CSV ──────────────────────────────────────────
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(df)} rows)")

    # ── Summary ──────────────────────────────────────────────────────
    labeled = df[df["label_status"] == "labeled"]
    print(f"\nReady for calibration: {len(labeled)} labeled rows")
    side_counts = labeled["side"].value_counts().to_dict()
    print(f"  Side distribution: {side_counts}")
    stroke_counts = labeled["true_stroke"].value_counts().to_dict()
    print(f"  Stroke distribution: {sorted(stroke_counts.items(), key=lambda x: -x[1])}")


if __name__ == "__main__":
    main()
