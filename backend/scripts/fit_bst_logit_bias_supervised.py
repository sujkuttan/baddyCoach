#!/usr/bin/env python3
"""Fit per-class additive logit bias from labeled shots.

Uses the 82 labeled shots in labels_enriched.csv to correct BST's systematic
class bias via convex optimisation.  Output JSON is consumed by the existing
``_load_logit_bias`` path — no inference code change needed.

Usage:
    python backend/scripts/fit_bst_logit_bias_supervised.py
    python backend/scripts/fit_bst_logit_bias_supervised.py --dry-run --reg-lambda 2.0
    python backend/scripts/fit_bst_logit_bias_supervised.py --fit-temperature

Output fields in the JSON match ``bst_logit_bias.json`` so it can be loaded
by ``BSTClassifier._load_logit_bias``.  After fitting, set::

    bst_prior_correction_enabled = True
    bst_prior_correction_strength = 1.0   # fitted bias is the full correction
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Helpers ──────────────────────────────────────────────────────────

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


def _class_name(cid: int) -> str:
    return SHUTTLESET_CLASSES[cid] if 0 <= cid < len(SHUTTLESET_CLASSES) else str(cid)


def _load_existing_bias(path: Path) -> np.ndarray:
    """Load existing bias JSON, return mean-centred (n_classes,) or zeros."""
    if not path.exists():
        print(f"  Prior file not found at {path} — using zeros")
        return np.zeros(25, dtype=np.float64)
    with open(path) as f:
        data = json.load(f)
    bias = np.array(data["bias"], dtype=np.float64)
    if bias.shape != (25,):
        print(f"  WARNING: prior bias has shape {bias.shape}, expected (25,). Using zeros.")
        return np.zeros(25, dtype=np.float64)
    bias = bias - bias.mean()
    source = data.get("source", "?")
    print(f"  Loaded prior bias ({data.get('n_clips', '?')} clips, source={source})")
    print(f"    range [{bias.min():.4f}, {bias.max():.4f}]")
    return bias


def _load_labels(csv_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load labeled rows from CSV.  Returns (logits, labels, metadata_df)."""
    df = pd.read_csv(csv_path)
    labeled = df[df["label_status"] == "labeled"].copy()
    if len(labeled) == 0:
        print("ERROR: No 'labeled' rows found in CSV")
        sys.exit(1)

    # Parse embedded JSON logits
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

    if len(logits_list) < 10:
        print(f"ERROR: Only {len(logits_list)} valid logit rows (need ≥ 10)")
        sys.exit(1)

    logits = np.stack(logits_list)
    labels = labeled.loc[valid_rows, "true_class_id"].values.astype(np.int64)
    meta = labeled.loc[valid_rows].copy()
    return logits, labels, meta


def _decorrect_logits(logits: np.ndarray, meta: pd.DataFrame,
                       prior: np.ndarray, strength: float) -> np.ndarray:
    """If embedded logits are post-correction, revert to raw.

    Checks whether ``argmax(embedded) ≈ predicted_class_id`` for most rows.
    If so, the logits are already prior-corrected and must be de-corrected
    before fitting a *new* bias on top.
    """
    pred_ids = meta["predicted_class_id"].values.astype(np.int64)
    embedded_pred = np.argmax(logits, axis=1)
    match_rate = (embedded_pred == pred_ids).mean()

    if match_rate >= 0.90:
        raw = logits + strength * prior[np.newaxis, :]
        print(f"  De-correct guard: argmax match rate {match_rate:.0%} ≥ 90%")
        print(f"    → embedded logits are post-correction (reverting with strength={strength})")
        return raw
    else:
        print(f"  De-correct guard: argmax match rate {match_rate:.0%} < 90%")
        print(f"    → treating embedded logits as raw (no de-correction)")
        return logits


def _class_support_table(labels: np.ndarray) -> dict[int, int]:
    """Count labeled samples per class ID."""
    counts = {}
    for cid in range(25):
        n = int((labels == cid).sum())
        if n > 0:
            counts[cid] = n
    return counts


# ── Loss & fitting ───────────────────────────────────────────────────

def _softmax_ce(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy loss (mean, with numerical stability)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(labels)
    return float(-np.mean(np.log(probs[np.arange(n), labels] + 1e-15)))


def _loss(b: np.ndarray, logits: np.ndarray, labels: np.ndarray,
           prior: np.ndarray, reg_lambda: float) -> float:
    """CE + L2 regularisation toward prior."""
    corrected = logits - b[np.newaxis, :]
    ce = _softmax_ce(corrected, labels)
    reg = float(reg_lambda * np.mean((b - prior) ** 2))
    return ce + reg


def _grad(b: np.ndarray, logits: np.ndarray, labels: np.ndarray,
           prior: np.ndarray, reg_lambda: float) -> np.ndarray:
    """Gradient of the loss w.r.t. b."""
    corrected = logits - b[np.newaxis, :]
    shifted = corrected - corrected.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(labels)
    # grad_ce = -(1/n) * (probs - one_hot)
    grad_ce = -probs.copy()
    grad_ce[np.arange(n), labels] += 1.0
    grad_ce /= n
    grad_reg = 2.0 * reg_lambda * (b - prior) / len(b)
    return grad_ce.sum(axis=0) + grad_reg


def _fit_bias(logits: np.ndarray, labels: np.ndarray,
               prior: np.ndarray, reg_lambda: float,
               frozen: np.ndarray) -> np.ndarray:
    """Fit per-class bias via L-BFGS-B.

    Parameters
    ----------
    frozen : (25,) bool array — True means the class bias is frozen to prior.
    """
    n_classes = logits.shape[1]
    b = prior.copy()

    free_mask = ~frozen
    free_idx = np.where(free_mask)[0]

    if len(free_idx) == 0:
        print("  All classes frozen — keeping prior unchanged")
        return prior

    def objective(b_free):
        b_full = b.copy()
        b_full[free_idx] = b_free
        return _loss(b_full, logits, labels, prior, reg_lambda)

    def gradient(b_free):
        b_full = b.copy()
        b_full[free_idx] = b_free
        g_full = _grad(b_full, logits, labels, prior, reg_lambda)
        return g_full[free_idx]

    x0 = prior[free_idx].copy()
    result = minimize(
        objective, x0, method="L-BFGS-B", jac=gradient,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        print(f"  WARNING: Optimiser did not converge ({result.message})")

    b[free_idx] = result.x
    # Enforce sum(b) = 0 while keeping frozen at prior values.
    # This preserves the relative differences among free classes (shift is uniform).
    frozen_sum = prior[frozen].sum()
    free_count = len(free_idx)
    current_free_sum = b[free_idx].sum()
    target_free_sum = -frozen_sum
    b[free_idx] -= (current_free_sum - target_free_sum) / free_count
    return b


# ── Metrics ──────────────────────────────────────────────────────────

def _compute_metrics(logits: np.ndarray, labels: np.ndarray,
                      bias: np.ndarray, classes: list[str]) -> dict:
    """Compute accuracy, macro-F1, per-class predicted rate."""
    corrected = logits - bias[np.newaxis, :]
    preds = np.argmax(corrected, axis=1)

    acc = float((preds == labels).mean())
    # Macro F1
    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels, preds, average="macro"))

    # Per-class predicted rate (how often each class is predicted)
    n_total = len(preds)
    pred_rate = {}
    for cid in range(25):
        count = int((preds == cid).sum())
        if count > 0:
            name = classes[cid] if cid < len(classes) else str(cid)
            pred_rate[name] = count / n_total

    return {"accuracy": acc, "macro_f1": f1, "pred_rate": pred_rate}


def _cv_metrics(logits: np.ndarray, labels: np.ndarray,
                 prior: np.ndarray, reg_lambda: float,
                 frozen: np.ndarray, n_folds: int = 5) -> dict:
    """k-fold cross-validated metrics."""
    from sklearn.model_selection import StratifiedKFold

    # Can't stratify on very rare classes — use KFold with class-stratified
    # fallback to simple shuffle
    n_classes = logits.shape[1]
    n_samples = len(labels)

    try:
        skf = StratifiedKFold(n_splits=min(n_folds, n_samples), shuffle=True, random_state=42)
        splits = list(skf.split(logits, labels))
    except Exception:
        # Fallback: random shuffle split
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(n_folds, n_samples), shuffle=True, random_state=42)
        splits = list(kf.split(logits))

    accs, f1s = [], []
    for train_idx, val_idx in splits:
        l_train, l_val = logits[train_idx], logits[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        b_fold = _fit_bias(l_train, y_train, prior, reg_lambda, frozen)
        corrected = l_val - b_fold[np.newaxis, :]
        preds = np.argmax(corrected, axis=1)
        accs.append(float((preds == y_val).mean()))
        from sklearn.metrics import f1_score
        f1s.append(float(f1_score(y_val, preds, average="macro")))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_folds": len(splits),
    }


def _print_report(before: dict, after: dict, cv: dict,
                   support: dict, frozen: list[int],
                   classes: list[str]):
    """Print formatted metrics report."""
    print()
    print("=" * 62)
    print("  SUPERVISED LOGIT-BIAS FIT — REPORT")
    print("=" * 62)
    print(f"  Samples:              {sum(support.values())}")
    print(f"  Classes with labels:  {len(support)} / 25")
    print(f"  Frozen classes:       {len(frozen)}",
          f"({', '.join(_class_name(c) for c in frozen)})" if frozen else "")
    print()
    print("  ┌────────────────────┬──────────┬──────────┐")
    print("  │ Metric             │ Before   │ After    │")
    print("  ├────────────────────┼──────────┼──────────┤")
    print(f"  │ Top-1 accuracy     │ {before['accuracy']:>6.1%}  │ {after['accuracy']:>6.1%}  │")
    print(f"  │ Macro F1           │ {before['macro_f1']:>8.4f} │ {after['macro_f1']:>8.4f} │")
    print("  └────────────────────┴──────────┴──────────┘")
    print()
    print(f"  k-fold CV (n={cv['n_folds']}):")
    print(f"    Accuracy: {cv['accuracy_mean']:.1%} ± {cv['accuracy_std']:.1%}")
    print(f"    Macro F1: {cv['macro_f1_mean']:.4f} ± {cv['macro_f1_std']:.4f}")
    print()
    print("  Per-class predicted rate (top changes):")
    rates_before = before.get("pred_rate", {})
    rates_after = after.get("pred_rate", {})
    all_classes = set(rates_before.keys()) | set(rates_after.keys())
    for cname in sorted(all_classes):
        rb = rates_before.get(cname, 0.0)
        ra = rates_after.get(cname, 0.0)
        if abs(rb - ra) > 0.01:
            print(f"    {cname:20s}  {rb:.1%}  →  {ra:.1%}")
    print("=" * 62)
    print()
    print("  Apply with:")
    print("    bst_prior_correction_enabled = True")
    print("    bst_prior_correction_strength = 1.0   # fitted bias is full correction")
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit BST logit bias from labeled shots")
    parser.add_argument("--labels-csv", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / "labels_enriched.csv"),
                        help="Path to labels CSV with embedded logits + true_class_id")
    parser.add_argument("--prior", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / "ckpts/bst/bst_logit_bias.json"),
                        help="Prior bias JSON (regularisation target); zeros if absent")
    parser.add_argument("--reg-lambda", type=float, default=1.0,
                        help="L2 regularisation strength toward prior (default: 1.0)")
    parser.add_argument("--min-support", type=int, default=5,
                        help="Classes with fewer labels are frozen to prior (default: 5)")
    parser.add_argument("--fit-temperature", action="store_true",
                        help="Also fit softmax temperature (alternating optimisation)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds for held-out metrics (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: same as --prior)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print metrics, don't write output")
    args = parser.parse_args()

    csv_path = Path(args.labels_csv)
    if not csv_path.exists():
        print(f"ERROR: labels CSV not found: {csv_path}")
        sys.exit(1)

    prior_path = Path(args.prior)
    output_path = Path(args.output) if args.output else prior_path

    # ── 1. Load labels ────────────────────────────────────────────────
    print("─" * 62)
    print("  Loading labeled data...")
    print(f"    CSV:  {csv_path}")
    logits, labels, meta = _load_labels(csv_path)
    print(f"    Loaded {len(logits)} labeled shots with valid logits")

    support = _class_support_table(labels)
    print(f"    Class support:")
    for cid in sorted(support):
        n = support[cid]
        bar = "█" * min(n, 20)
        print(f"      {_class_name(cid):24s}  {n:3d}  {bar}")

    # ── 2. Load prior & de-correct ─────────────────────────────────────
    print()
    print("─" * 62)
    print("  Prior & de-correct check...")
    prior = _load_existing_bias(prior_path)

    # De-correct: revert logits to raw if they're post-correction
    from app.config.settings import settings
    strength = settings.bst_prior_correction_strength if hasattr(settings, 'bst_prior_correction_strength') else 0.75
    logits_raw = _decorrect_logits(logits, meta, prior, strength)

    # ── 3. Metrics before (using raw logits with prior) ────────────────
    print()
    print("─" * 62)
    print("  Metrics BEFORE fit (raw logits + existing prior)...")
    before = _compute_metrics(logits_raw, labels, prior, SHUTTLESET_CLASSES)
    print(f"    Top-1 accuracy: {before['accuracy']:.1%}")
    print(f"    Macro F1:       {before['macro_f1']:.4f}")

    # ── 4. Determine frozen classes ────────────────────────────────────
    frozen = np.zeros(25, dtype=bool)
    frozen_list = []
    for cid in range(25):
        n = support.get(cid, 0)
        if 0 < n < args.min_support:
            frozen[cid] = True
            frozen_list.append(cid)
    if frozen_list:
        print()
        print(f"  Frozen classes (< {args.min_support} labels):")
        for cid in frozen_list:
            print(f"    {_class_name(cid):24s}  (n={support[cid]})")
    # Classes with zero support stay at prior (already zero in prior diff)
    zero_support = [cid for cid in range(25) if cid not in support]
    for cid in zero_support:
        frozen[cid] = True
    if zero_support:
        print(f"    (plus {len(zero_support)} classes with zero support — kept at prior)")

    # ── 5. Fit ─────────────────────────────────────────────────────────
    print()
    print("─" * 62)
    print("  Fitting bias...")
    print(f"    reg_lambda = {args.reg_lambda}")
    print(f"    {int((~frozen).sum())} free classes, {int(frozen.sum())} frozen")
    b_fitted = _fit_bias(logits_raw, labels, prior, args.reg_lambda, frozen)
    print(f"    Fitted bias range: [{b_fitted.min():.4f}, {b_fitted.max():.4f}]")
    print(f"    sum(b_fitted) = {b_fitted.sum():.2e} (should be ≈ 0)")

    # ── Optional temperature fit ───────────────────────────────────────
    temperature = 1.0
    if args.fit_temperature:
        print()
        print("  Fitting temperature (alternating with bias)...")
        try:
            import torch
            from app.models.bst import BSTClassifier
            for _ in range(3):
                corrected = logits_raw - b_fitted[np.newaxis, :]
                T = BSTClassifier.compute_optimal_temperature(corrected, labels)
                corrected /= T
                b_fitted = _fit_bias(corrected * T, labels, prior, args.reg_lambda, frozen)
            temperature = float(T)
            print(f"    Fitted temperature: T = {temperature:.4f}")
        except Exception as e:
            print(f"    WARNING: Temperature fit failed ({e}), keeping T=1.0")

    # ── 6. Metrics after ───────────────────────────────────────────────
    print()
    print("─" * 62)
    print("  Metrics AFTER fit (corrected logits)...")
    # Build the full correction: apply fitted bias then temperature
    bias_combined = b_fitted.copy()
    # If temperature was fitted, scale the bias accordingly
    # (bias is applied as logits - b before temp scaling)
    after = _compute_metrics(logits_raw, labels, bias_combined, SHUTTLESET_CLASSES)
    print(f"    Top-1 accuracy: {after['accuracy']:.1%}")
    print(f"    Macro F1:       {after['macro_f1']:.4f}")

    # ── 7. Cross-validated metrics ─────────────────────────────────────
    print()
    print("─" * 62)
    print("  Cross-validated metrics...")
    cv = _cv_metrics(logits_raw, labels, prior, args.reg_lambda, frozen, args.folds)

    # ── 8. Report ──────────────────────────────────────────────────────
    _print_report(before, after, cv, support, frozen_list, SHUTTLESET_CLASSES)

    # ── 9. Write output ────────────────────────────────────────────────
    if args.dry_run:
        print("  (dry-run — no file written)")
        return

    output_data = {
        "bias": b_fitted.tolist(),
        "n_clips": int(len(logits_raw)),
        "source": f"supervised fit from {csv_path.name}",
        "method": "supervised",
        "reg_lambda": args.reg_lambda,
        "min_support": args.min_support,
        "temperature": temperature,
        "temperature_far": temperature,
        "temperature_near": temperature,
        "before_accuracy": before["accuracy"],
        "before_macro_f1": before["macro_f1"],
        "after_accuracy": after["accuracy"],
        "after_macro_f1": after["macro_f1"],
        "cv_accuracy_mean": cv["accuracy_mean"],
        "cv_accuracy_std": cv["accuracy_std"],
        "cv_macro_f1_mean": cv["macro_f1_mean"],
        "cv_macro_f1_std": cv["macro_f1_std"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Written to {output_path}")


if __name__ == "__main__":
    main()
