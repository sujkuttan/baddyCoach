#!/usr/bin/env python3
"""Calibrate BST softmax temperature on a validation set.

Usage:
    python scripts/calibrate_bst.py \
        --logits path/to/logits.npy \
        --labels path/to/labels.npy \
        [--output path/to/bst_temperature.json]

    python scripts/calibrate_bst.py \
        --data-dir path/to/calibration_data/ \
        [--output ...]

If --logits/--labels are given, those arrays are loaded directly.
If --data-dir is given, the script loads all .npz files in that directory,
each containing 'logits' (N, n_classes) and 'labels' (N,) arrays.

If --output is omitted, the result is saved to the default CKPT_DIR/bst/ path.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Calibrate BST softmax temperature")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logits", type=str, help="Path to logits .npy file (N, n_classes)")
    group.add_argument("--data-dir", type=str, help="Directory with .npz files (each: logits + labels)")
    parser.add_argument("--labels", type=str, help="Path to labels .npy file (N,) — required with --logits")
    parser.add_argument("--output", type=str, help="Output path for temperature JSON (default: CKPT_DIR/bst/bst_temperature.json)")
    args = parser.parse_args()

    # ── Load logits + labels ────────────────────────────────────────────
    logits_list, labels_list = [], []

    if args.logits:
        if not args.labels:
            print("ERROR: --labels is required with --logits")
            sys.exit(1)
        logits = np.load(args.logits)
        labels = np.load(args.labels)
        logits_list.append(logits)
        labels_list.append(labels)
        print(f"Loaded logits {logits.shape}, labels {labels.shape}")
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        for f in sorted(data_dir.glob("*.npz")):
            data = np.load(f)
            logits_list.append(data["logits"])
            labels_list.append(data["labels"])
            print(f"  Loaded {f.name}: logits {data['logits'].shape}, labels {data['labels'].shape}")

    if not logits_list:
        print("ERROR: No data loaded")
        sys.exit(1)

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"\nTotal: {len(logits)} samples, {logits.shape[1]} classes")

    # ── Resolve output path ─────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        from app.pipeline.shared.models import CKPT_DIR
        output_path = CKPT_DIR / "bst" / "bst_temperature.json"

    # ── Compute optimal temperature ─────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.models.bst import BSTClassifier

    T_opt = BSTClassifier.compute_optimal_temperature(logits, labels)
    print(f"Optimal temperature: T = {T_opt:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"temperature": round(T_opt, 4)}, f)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
