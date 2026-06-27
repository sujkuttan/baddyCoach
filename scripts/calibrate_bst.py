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

    python scripts/calibrate_bst.py \
        --labels-csv labels_<job>.csv \
        [--logits-source results.json | debug_bst_outputs.parquet] \
        [--output ...]

If --labels-csv is given, the CSV must contain columns:
    shot_id, frame, ts_start, ts_end, player_id, side,
    predicted_stroke, predicted_class_id, true_stroke, true_class_id, label_status
Only rows with label_status == "labeled" are used.
Logits are resolved from the --logits-source (default: results.json via
report.json's per-shot logits, else join by frame/shot_id to
debug_bst_outputs.parquet).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


COACH_CLASSES = [
    "net_shot", "block", "smash", "lift", "clear", "drive",
    "drop", "push", "rush", "cross_court", "short_serve", "long_serve",
]


def _shuttleset_id(stroke: str, side: str) -> int:
    """Map (coach_stroke, side) -> ShuttleSet class ID (0-24)."""
    if side == "far":
        idx = COACH_CLASSES.index(stroke) if stroke in COACH_CLASSES else -1
        return idx + 1 if idx >= 0 else 0
    idx = COACH_CLASSES.index(stroke) if stroke in COACH_CLASSES else -1
    return idx + 13 if idx >= 0 else 0


def load_from_csv(csv_path: Path, logits_source: str = "results.json",
                  debug_parquet: str | None = None) -> tuple:
    """Load labeled data from CSV and resolve logits.

    Returns:
        (logits: np.ndarray, labels: np.ndarray, metadata: dict)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    labeled = df[df["label_status"] == "labeled"].copy()
    if len(labeled) == 0:
        print("ERROR: No 'labeled' rows found in CSV")
        sys.exit(1)

    print(f"CSV: {len(df)} rows, {len(labeled)} labeled")

    # Resolve logits per shot
    if logits_source == "results.json":
        logits_list = []
        for _, row in labeled.iterrows():
            raw = row.get("logits")
            if pd.isna(raw) or not raw:
                logits_list.append(None)
            elif isinstance(raw, str):
                try:
                    logits_list.append(np.array(json.loads(raw)))
                except (json.JSONDecodeError, TypeError):
                    logits_list.append(None)
            else:
                logits_list.append(None)
        missing = sum(1 for l in logits_list if l is None)
        if missing == len(logits_list):
            print("No logits in CSV, trying debug_bst_outputs.parquet fallback...")
            logits_list = [None] * len(labeled)
        elif missing > 0:
            print(f"WARNING: {missing}/{len(labeled)} rows missing logits in CSV")
    else:
        logits_list = [None] * len(labeled)

    # Fallback: join with debug_bst_outputs.parquet
    if all(l is None for l in logits_list):
        default_path = Path("backend/results/debug/debug_bst_outputs.parquet")
        parquet_path = Path(debug_parquet) if debug_parquet else default_path
        if not parquet_path.exists():
            print(f"ERROR: debug_bst_outputs.parquet not found at {parquet_path}")
            print("Specify --logits-source path to the parquet file, or embed logits in the CSV")
            sys.exit(1)
        debug_df = pd.read_parquet(parquet_path)
        # Join on frame (assuming debug_bst_outputs has no shot_id)
        merged = labeled.merge(
            debug_df[["logits_all"]], left_index=True, right_index=True,
            how="left",
        )
        logits_list = []
        for _, row in merged.iterrows():
            raw = row.get("logits_all")
            if pd.isna(raw) or not raw:
                logits_list.append(None)
            elif isinstance(raw, str):
                try:
                    logits_list.append(np.array(json.loads(raw)))
                except (json.JSONDecodeError, TypeError):
                    logits_list.append(None)
            else:
                logits_list.append(None)

    valid_mask = [l is not None for l in logits_list]
    if sum(valid_mask) < 30:
        print("ERROR: Need at least 30 labeled shots with logits to calibrate "
              f"(got {sum(valid_mask)})")
        sys.exit(1)

    logits = np.stack([logits_list[i] for i in range(len(logits_list)) if valid_mask[i]])
    labels = labeled["true_class_id"].values.astype(np.int64)[valid_mask]

    # Verify side → class_id mapping
    for i in range(len(labeled)):
        row = labeled.iloc[i]
        derived = _shuttleset_id(row["true_stroke"], row["side"])
        if derived != row["true_class_id"]:
            print(f"WARNING: row {row['shot_id']}: true_class_id={row['true_class_id']} "
                  f"but derived={derived} from stroke={row['true_stroke']} side={row['side']}")

    n_labeled = sum(valid_mask)
    n_total = len(logits)

    metadata = {
        "n_csv_rows": len(df),
        "n_labeled": n_labeled,
        "n_logits_found": n_total,
        "n_unlabeled": len(df) - len(labeled),
        "n_not_a_shot": len(df[df["label_status"] == "not_a_shot"]),
        "n_unsure": len(df[df["label_status"] == "unsure"]),
        "hit_precision": len(df[df["label_status"] != "not_a_shot"]) / max(1, len(df)),
    }

    return logits, labels, metadata


def calibration_report(logits: np.ndarray, labels: np.ndarray, T: float,
                       metadata: dict, classes: list[str]) -> str:
    """Generate a printable calibration report with accuracy/F1/ECE/NLL."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    probs = np.exp(logits / T)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    # Accuracy
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    # ECE (Expected Calibration Error)
    confidences = probs.max(axis=1)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.any():
            bin_acc = accuracy_score(labels[in_bin], preds[in_bin])
            bin_conf = confidences[in_bin].mean()
            ece += abs(bin_acc - bin_conf) * in_bin.sum() / len(labels)

    # NLL
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-15))

    # NLL before calibration (T=1)
    probs_raw = np.exp(logits)
    probs_raw = probs_raw / probs_raw.sum(axis=1, keepdims=True)
    nll_raw = -np.mean(np.log(probs_raw[np.arange(len(labels)), labels] + 1e-15))

    # Confusion matrix highlights
    cm = confusion_matrix(labels, preds)
    class_acc = {}
    n_classes = logits.shape[1]
    for c in range(n_classes):
        if c in labels:
            mask = labels == c
            class_acc[classes[c] if c < len(classes) else str(c)] = {
                "accuracy": (preds[mask] == c).sum() / mask.sum(),
                "count": mask.sum(),
            }

    hit_prec = metadata.get("hit_precision", "N/A")
    if isinstance(hit_prec, float):
        hit_prec = f"{hit_prec:.1%}"
    report_lines = [
        "=" * 60,
        "BST CALIBRATION REPORT",
        "=" * 60,
        f"Total labeled shots: {metadata.get('n_labeled', len(logits))}",
        f"Hit precision:       {hit_prec}",
        f"Not-a-shot excl:     {metadata.get('n_not_a_shot', 'N/A')}",
        "",
        f"Not-a-shot excl:     {metadata.get('n_not_a_shot', 'N/A')}",
        "",
        f"Optimal temperature: T = {T:.4f}",
        "",
        "Before (T=1.0):",
        f"  NLL: {nll_raw:.4f}",
        "",
        f"After (T={T:.4f}):",
        f"  Top-1 accuracy: {acc:.2%}",
        f"  Macro F1:       {f1:.4f}",
        f"  ECE:            {ece:.4f}",
        f"  NLL:            {nll:.4f}",
        "",
        "Per-class accuracy:",
    ]
    for cls, info in sorted(class_acc.items(), key=lambda x: -x[1]["count"]):
        report_lines.append(f"  {cls:15s} {info['accuracy']:.1%} (n={info['count']})")

    report_lines.append("=" * 60)
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Calibrate BST softmax temperature")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logits", type=str, help="Path to logits .npy file (N, n_classes)")
    group.add_argument("--data-dir", type=str, help="Directory with .npz files (each: logits + labels)")
    group.add_argument("--labels-csv", type=str, help="Path to labels CSV from labeling UI")
    parser.add_argument("--labels", type=str, help="Path to labels .npy file (N,) — required with --logits")
    parser.add_argument("--logits-source", type=str, default="results.json",
                        help="Logits source for CSV mode: 'results.json' or parquet path")
    parser.add_argument("--output", type=str, help="Output path for temperature JSON (default: CKPT_DIR/bst/bst_temperature.json)")
    args = parser.parse_args()

    # ── Resolve backend path ──────────────────────────────────────────────
    backend_root = Path(__file__).resolve().parent.parent / "backend"
    sys.path.insert(0, str(backend_root))

    # ── Load logits + labels ────────────────────────────────────────────
    logits_list, labels_list = [], []
    metadata = {}

    if args.logits:
        if not args.labels:
            print("ERROR: --labels is required with --logits")
            sys.exit(1)
        logits_list.append(np.load(args.logits))
        labels_list.append(np.load(args.labels))
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        for f in sorted(data_dir.glob("*.npz")):
            data = np.load(f)
            logits_list.append(data["logits"])
            labels_list.append(data["labels"])
            print(f"  Loaded {f.name}: logits {data['logits'].shape}, labels {data['labels'].shape}")
    elif args.labels_csv:
        csv_path = Path(args.labels_csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            sys.exit(1)
        logits, labels, metadata = load_from_csv(csv_path, args.logits_source)
        logits_list.append(logits)
        labels_list.append(labels)

    if not logits_list:
        print("ERROR: No data loaded")
        sys.exit(1)

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    n_classes = logits.shape[1]
    print(f"\nTotal: {len(logits)} samples, {n_classes} classes")

    if len(logits) < 30:
        print("ERROR: Need at least 30 labeled shots to calibrate "
              f"(got {len(logits)})")
        sys.exit(1)

    # Build coach class name list for the report
    coach_class_names = ["unknown"] + COACH_CLASSES + COACH_CLASSES
    coach_class_names = coach_class_names[:n_classes]

    # ── Resolve output path ─────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        from app.pipeline.shared.models import CKPT_DIR
        output_path = CKPT_DIR / "bst" / "bst_temperature.json"

    # ── Compute optimal temperature ─────────────────────────────────────
    from app.models.bst import BSTClassifier

    T_opt = BSTClassifier.compute_optimal_temperature(logits, labels)
    print(f"\nOptimal temperature: T = {T_opt:.4f}")

    # ── Calibration report ──────────────────────────────────────────────
    report = calibration_report(logits, labels, T_opt, metadata, coach_class_names)
    print("\n" + report)

    # ── Save ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"temperature": round(T_opt, 4)}, f)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
