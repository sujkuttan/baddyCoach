#!/usr/bin/env python3
"""Calibrate BST temperature + logit bias from labeled data.

Steps:
1. Load labels_enriched.csv
2. Deduplicate (one label per shot_frame, keep closest)
3. Filter by max_frame_diff
4. Compute optimal temperature (NLL minimization via LBFGS)
5. Fit per-class logit bias (CE + L2 via LBFGS)  
6. Save both JSON files
7. Print evaluation metrics

Usage:
    python backend/scripts/calibrate_bst.py
    python backend/scripts/calibrate_bst.py --max-frame-diff 10 --dry-run
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


def load_clean_labels(csv_path: str, max_frame_diff: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load, deduplicate, and filter labels. Returns (logits, labels, meta)."""
    df = pd.read_csv(csv_path)
    labeled = df[df["label_status"] == "labeled"].copy()
    print(f"  Raw labeled entries: {len(labeled)}")

    # Parse logits
    logits_list = []
    valid_rows = []
    for idx, row in labeled.iterrows():
        raw = row.get("logits")
        if pd.isna(raw) or not raw:
            continue
        try:
            arr = np.array(json.loads(raw) if isinstance(raw, str) else raw, dtype=np.float64)
            if arr.shape != (25,):
                continue
            logits_list.append(arr)
            valid_rows.append(idx)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    df_valid = labeled.loc[valid_rows].copy()
    df_valid["logits_arr"] = logits_list
    print(f"  With valid logits: {len(df_valid)}")

    # Filter by frame_diff
    before_filter = len(df_valid)
    df_valid = df_valid[df_valid["frame_diff"] <= max_frame_diff].copy()
    print(f"  After frame_diff ≤ {max_frame_diff}: {len(df_valid)} (removed {before_filter - len(df_valid)})")

    # Deduplicate: for each shot_frame, keep the label with smallest frame_diff
    df_valid = df_valid.loc[df_valid.groupby("shot_frame")["frame_diff"].idxmin()].copy()
    print(f"  After dedup (1 label/shot): {len(df_valid)}")

    logits = np.stack(df_valid["logits_arr"].values)
    labels = df_valid["true_class_id"].values.astype(np.int64)
    print(f"  Frame diff stats: min={df_valid['frame_diff'].min()}, median={df_valid['frame_diff'].median():.0f}, max={df_valid['frame_diff'].max()}")
    print(f"  Unique classes: {sorted(df_valid['true_class_id'].unique())}")

    return logits, labels, df_valid


def compute_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find optimal temperature via NLL minimization using LBFGS."""
    try:
        import torch
    except ImportError:
        print("  WARNING: PyTorch not available, keeping T=1.0")
        return 1.0
    try:
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()
        nll = torch.nn.CrossEntropyLoss()

        T = torch.ones(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=150)

        def closure():
            optimizer.zero_grad()
            loss = nll(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_opt = float(T.detach().item())
        if T_opt <= 0:
            return 1.0
        return max(0.01, min(T_opt, 100.0))
    except Exception as e:
        print(f"  WARNING: Temperature optimization failed: {e}")
        return 1.0


def softmax_ce(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy loss (mean)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(labels)
    return float(-np.mean(np.log(probs[np.arange(n), labels] + 1e-15)))


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_logit_bias(logits: np.ndarray, labels: np.ndarray,
                   reg_lambda: float = 1.0) -> np.ndarray:
    """Fit per-class logit bias via L-BFGS-B with L2 regularization."""
    from scipy.optimize import minimize
    n_classes = logits.shape[1]
    b = np.zeros(n_classes, dtype=np.float64)

    def loss(b_arr):
        corrected = logits - b_arr[np.newaxis, :]
        ce = softmax_ce(corrected, labels)
        reg = float(reg_lambda * np.mean(b_arr ** 2))
        return ce + reg

    def grad(b_arr):
        corrected = logits - b_arr[np.newaxis, :]
        shifted = corrected - corrected.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = len(labels)
        grad_ce = -probs.copy()
        grad_ce[np.arange(n), labels] += 1.0
        grad_ce /= n
        grad_reg = 2.0 * reg_lambda * b_arr / n_classes
        return grad_ce.sum(axis=0) + grad_reg

    result = minimize(loss, b, method="L-BFGS-B", jac=grad,
                      options={"maxiter": 500, "ftol": 1e-12})

    if not result.success:
        print(f"  WARNING: Optimizer did not converge: {result.message}")
    b_fitted = result.x
    # Mean-center
    b_fitted = b_fitted - b_fitted.mean()
    return b_fitted


def compute_accuracy(logits: np.ndarray, labels: np.ndarray,
                     bias: np.ndarray, T: float) -> tuple[float, float]:
    """Compute top-1 accuracy and macro F1 after bias + temperature."""
    corrected = (logits - bias[np.newaxis, :]) / T
    preds = np.argmax(corrected, axis=1)
    acc = float((preds == labels).mean())
    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Calibrate BST temperature + logit bias")
    parser.add_argument("--labels-csv", default="labels_enriched.csv")
    parser.add_argument("--max-frame-diff", type=int, default=10,
                        help="Max frame diff for label-to-shot match (default: 10)")
    parser.add_argument("--reg-lambda", type=float, default=0.5,
                        help="L2 regularization strength for bias fit (default: 0.5)")
    parser.add_argument("--temperature-out", default="ckpts/bst/bst_temperature.json")
    parser.add_argument("--bias-out", default="ckpts/bst/bst_logit_bias.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    csv_path = root / args.labels_csv
    temp_path = root / args.temperature_out
    bias_path = root / args.bias_out

    print("=" * 60)
    print("  BST CALIBRATION")
    print("=" * 60)

    # ── 1. Load clean labels ──────────────────────────────────────────
    print("\n[1/5] Loading labels...")
    logits, labels, meta = load_clean_labels(str(csv_path), args.max_frame_diff)
    if len(logits) < 10:
        print(f"ERROR: Only {len(logits)} valid samples (need ≥ 10)")
        sys.exit(1)

    # ── 2. Compute temperature ────────────────────────────────────────
    print("\n[2/5] Computing optimal temperature...")
    T_opt = compute_optimal_temperature(logits, labels)
    print(f"  T = {T_opt:.4f}")
    print(f"  (cross-entropy before: {softmax_ce(logits, labels):.4f})")
    print(f"  (cross-entropy  after: {softmax_ce(logits / T_opt, labels):.4f})")

    # ── 3. Fit logit bias (on temperature-scaled logits) ─────────────
    print("\n[3/5] Fitting logit bias...")
    logits_scaled = logits / T_opt
    bias = fit_logit_bias(logits_scaled, labels, reg_lambda=args.reg_lambda)
    print(f"  Bias range: [{bias.min():.4f}, {bias.max():.4f}]")
    print(f"  sum(bias) = {bias.sum():.2e}")

    # ── 4. Evaluate ───────────────────────────────────────────────────
    print("\n[4/5] Evaluating...")
    acc_before, f1_before = compute_accuracy(logits, labels, np.zeros(25), 1.0)
    acc_after, f1_after = compute_accuracy(logits, labels, bias, T_opt)

    print(f"  {'':>25s}  {'Before':>10s}  {'After':>10s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}")
    print(f"  {'Top-1 accuracy':>25s}  {acc_before:>8.1%}  {acc_after:>8.1%}")
    print(f"  {'Macro F1':>25s}  {f1_before:>10.4f}  {f1_after:>10.4f}")

    # Confusion
    corrected = (logits - bias[np.newaxis, :]) / T_opt
    preds = np.argmax(corrected, axis=1)
    print(f"\n  Per-class accuracy:")
    for cid in sorted(set(labels)):
        mask = labels == cid
        n = mask.sum()
        correct = (preds[mask] == labels[mask]).sum()
        name = SHUTTLESET_CLASSES[cid]
        print(f"    {name:24s}  {correct}/{n} ({correct/n*100:.0f}%)")

    # ── 5. Save ───────────────────────────────────────────────────────
    print("\n[5/5] Saving...")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return

    # Temperature
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_data = {
        "temperature": T_opt,
        "temperature_far": T_opt,
        "temperature_near": T_opt,
        "mean_conf_t1": float(softmax_probs(logits).max(axis=1).mean()),
        "mean_conf_cal": float(softmax_probs(corrected).max(axis=1).mean()),
        "n_model": len(logits),
        "method": f"nll_minimization_from_{csv_path.name}_framediff<={args.max_frame_diff}",
    }
    with open(temp_path, "w") as f:
        json.dump(temp_data, f, indent=2)
    print(f"  Temperature -> {temp_path}")

    # Logit bias
    bias_data = {
        "bias": bias.tolist(),
        "n_clips": int(len(logits)),
        "source": f"supervised fit from {csv_path.name} (frame_diff ≤ {args.max_frame_diff})",
        "method": "supervised",
        "reg_lambda": args.reg_lambda,
        "min_support": 1,
        "temperature": T_opt,
        "temperature_far": T_opt,
        "temperature_near": T_opt,
        "before_accuracy": acc_before,
        "before_macro_f1": f1_before,
        "after_accuracy": acc_after,
        "after_macro_f1": f1_after,
    }
    with open(bias_path, "w") as f:
        json.dump(bias_data, f, indent=2)
    print(f"  Bias -> {bias_path}")

    print("\n" + "=" * 60)
    print("  Done. Apply with:")
    print("    bst_prior_correction_enabled = True")
    print("    bst_prior_correction_strength = 1.0")
    print("    (settings are already at these defaults)")
    print("=" * 60)


if __name__ == "__main__":
    main()
