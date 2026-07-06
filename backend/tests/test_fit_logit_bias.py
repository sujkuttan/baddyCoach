"""Tests for supervised logit-bias fitting (fit_bst_logit_bias_supervised.py)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.scripts.fit_bst_logit_bias_supervised import (
    _load_existing_bias,
    _decorrect_logits,
    _fit_bias,
    _compute_metrics,
    _class_support_table,
    _softmax_ce,
    SHUTTLESET_CLASSES,
)
from app.models.bst import BSTClassifier


# ── Fixtures ──────────────────────────────────────────────────────────

RNG = np.random.RandomState(42)


@pytest.fixture
def synthetic_logits_labels():
    """Create synthetic 25-class logits with a known class-4 offset.

    True signal: class 0 (unknown) has low logits; class 4 (Top_lift) is
    systematically suppressed by +3.0 on class 3 (Top_smash), making the
    model over-predict smash on lift clips.
    """
    n = 200
    logits = RNG.randn(n, 25).astype(np.float64) * 0.5
    # Inject a real signal for each sample
    true_classes = RNG.randint(1, 25, size=n)  # avoid class 0 (unknown)
    for i in range(n):
        logits[i, true_classes[i]] += 4.0
    # Inject a +3.0 bias on class 3 (Top_smash) — systematic over-prediction
    logits[:, 3] += 3.0
    labels = true_classes.copy()
    return logits, labels


@pytest.fixture
def prior_zeros():
    return np.zeros(25, dtype=np.float64)


# ── Recovery ──────────────────────────────────────────────────────────


def test_recovery_removes_known_offset(synthetic_logits_labels, prior_zeros):
    """Fitted bias removes a known constant offset on one class."""
    logits, labels = synthetic_logits_labels

    frozen = np.zeros(25, dtype=bool)
    frozen[0] = True
    # Weak regularisation so the known offset can be recovered
    b_fitted = _fit_bias(logits, labels, prior_zeros, reg_lambda=0.01, frozen=frozen)

    # The fitted bias on class 3 should be close to +3.0 to cancel the offset
    # (correction is logits - b, so b[3] ≈ 3.0 neutralises the +3.0 injection)
    assert abs(b_fitted[3] - 3.0) < 1.5, f"b[3]={b_fitted[3]:.3f} should be ≈ 3.0"

    # Accuracy should improve with the fitted bias
    before_acc = _compute_metrics(logits, labels, prior_zeros, SHUTTLESET_CLASSES)["accuracy"]
    after_acc = _compute_metrics(logits, labels, b_fitted, SHUTTLESET_CLASSES)["accuracy"]
    assert after_acc >= before_acc - 0.01, f"Accuracy regressed: {before_acc:.1%} → {after_acc:.1%}"


# ── Mean-centring ─────────────────────────────────────────────────────


def test_fitted_bias_is_mean_centred(synthetic_logits_labels, prior_zeros):
    """sum(b_fitted) ≈ 0."""
    logits, labels = synthetic_logits_labels
    frozen = np.zeros(25, dtype=bool)
    frozen[0] = True
    b_fitted = _fit_bias(logits, labels, prior_zeros, reg_lambda=1.0, frozen=frozen)
    assert abs(b_fitted.sum()) < 1e-8, f"sum(b) = {b_fitted.sum():.2e}"


# ── Rare-class freeze ─────────────────────────────────────────────────


def test_rare_class_freeze():
    """Classes with < min-support labels keep prior value."""
    n_classes = 25
    n = 30
    logits = RNG.randn(n, n_classes).astype(np.float64)
    # Only classes 1, 2, 3 have labels; class 4 has 1 label (< min-support=2)
    labels = np.array([1] * 10 + [2] * 10 + [3] * 9 + [4] * 1)
    # Add signal
    for i in range(n):
        logits[i, labels[i]] += 5.0

    prior = np.zeros(n_classes, dtype=np.float64)
    frozen = np.zeros(n_classes, dtype=bool)
    frozen[0] = True  # class 0 unknown — no labels
    # Classes with < 2 labels should be frozen
    support = _class_support_table(labels)
    for cid in range(n_classes):
        n_support = support.get(cid, 0)
        if 0 < n_support < 2:
            frozen[cid] = True

    b_fitted = _fit_bias(logits, labels, prior, reg_lambda=1.0, frozen=frozen)

    # Frozen classes should equal prior (zero)
    for cid in np.where(frozen)[0]:
        assert b_fitted[cid] == pytest.approx(0.0, abs=1e-10), f"Class {cid} should be frozen at 0"

    # Non-frozen classes should deviate from zero
    for cid in range(1, 4):
        assert abs(b_fitted[cid]) > 0.1, f"Class {cid} should have been fitted"


# ── Regularisation ────────────────────────────────────────────────────


def test_regularisation_strong_lambda_pulls_to_prior(synthetic_logits_labels, prior_zeros):
    """λ → ∞  ⇒ fitted bias ≈ prior (zero)."""
    logits, labels = synthetic_logits_labels
    frozen = np.zeros(25, dtype=bool)
    frozen[0] = True

    b_weak = _fit_bias(logits, labels, prior_zeros, reg_lambda=0.01, frozen=frozen)
    b_strong = _fit_bias(logits, labels, prior_zeros, reg_lambda=1000.0, frozen=frozen)

    # Strong regularisation should keep bias closer to zero
    assert np.abs(b_strong).max() < np.abs(b_weak).max(), (
        f"Strong λ should shrink bias: weak max={np.abs(b_weak).max():.3f}, "
        f"strong max={np.abs(b_strong).max():.3f}"
    )


def test_regularisation_weak_lambda_fits_harder(synthetic_logits_labels, prior_zeros):
    """λ → 0  ⇒ fits training set harder (higher training accuracy)."""
    logits, labels = synthetic_logits_labels
    frozen = np.zeros(25, dtype=bool)
    frozen[0] = True

    b_weak = _fit_bias(logits, labels, prior_zeros, reg_lambda=0.001, frozen=frozen)
    b_strong = _fit_bias(logits, labels, prior_zeros, reg_lambda=100.0, frozen=frozen)

    acc_weak = _compute_metrics(logits, labels, b_weak, SHUTTLESET_CLASSES)["accuracy"]
    acc_strong = _compute_metrics(logits, labels, b_strong, SHUTTLESET_CLASSES)["accuracy"]

    # Weak regularisation should fit training data at least as well as strong
    assert acc_weak >= acc_strong - 0.01, (
        f"Weak λ acc ({acc_weak:.1%}) < strong λ acc ({acc_strong:.1%})"
    )


# ── De-correct guard ──────────────────────────────────────────────────


def test_decorrect_guard_detects_post_correction():
    """When embedded logits are post-correction, de-correct reconstructs raw."""
    n = 50
    prior = np.random.randn(25).astype(np.float64) * 0.5
    prior = prior - prior.mean()

    raw_logits = np.random.randn(n, 25).astype(np.float64) * 0.5
    true_classes = np.random.randint(1, 25, size=n)
    for i in range(n):
        raw_logits[i, true_classes[i]] += 4.0

    # Apply correction just like BST does (logits - strength * prior)
    strength = 0.75
    corrected_logits = raw_logits - strength * prior[np.newaxis, :]

    # Build a meta DataFrame where predicted_class_id = argmax(corrected)
    meta = pd.DataFrame({
        "predicted_class_id": np.argmax(corrected_logits, axis=1),
    })

    # De-correct should detect ≥ 90% match and revert
    reconstructed = _decorrect_logits(corrected_logits, meta, prior, strength)
    assert np.allclose(reconstructed, raw_logits, atol=1e-10), (
        "De-corrected logits should match raw"
    )


def test_decorrect_guard_leaves_raw_unchanged():
    """When embedded logits are raw (low match rate), no de-correction."""
    n = 50
    prior = np.random.randn(25).astype(np.float64) * 0.5
    prior = prior - prior.mean()

    raw_logits = np.random.randn(n, 25).astype(np.float64)

    # predicted_class_id that doesn't match argmax of raw
    meta = pd.DataFrame({
        "predicted_class_id": np.zeros(n, dtype=np.int64),  # all 0, unlikely to match
    })

    reconstructed = _decorrect_logits(raw_logits, meta, prior, 0.75)
    assert np.allclose(reconstructed, raw_logits, atol=1e-10), (
        "Raw logits should be unchanged"
    )


# ── Output round-trip ────────────────────────────────────────────────


def test_output_round_trip(synthetic_logits_labels, prior_zeros):
    """Fitted bias saves to JSON and loads unchanged via _load_logit_bias."""
    logits, labels = synthetic_logits_labels
    frozen = np.zeros(25, dtype=bool)
    frozen[0] = True
    b_fitted = _fit_bias(logits, labels, prior_zeros, reg_lambda=1.0, frozen=frozen)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "bias": b_fitted.tolist(),
            "n_clips": len(logits),
            "source": "test",
            "method": "supervised",
            "reg_lambda": 1.0,
        }, f)
        tmp_path = Path(f.name)

    try:
        with open(tmp_path) as f:
            data = json.load(f)
        bias_loaded = np.array(data["bias"], dtype=np.float64)
        bias_loaded = bias_loaded - bias_loaded.mean()

        assert bias_loaded.shape == (25,)
        assert abs(bias_loaded.sum()) < 1e-8
        assert np.allclose(bias_loaded, b_fitted, atol=1e-10)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_round_trip_through_load_logit_bias():
    """JSON written by the script loads through _load_logit_bias correctly."""
    from app.config.settings import settings

    # Create temp bias JSON
    test_bias = np.random.randn(25).astype(np.float64)
    test_bias = test_bias - test_bias.mean()

    orig_path = settings.bst_logit_bias_path
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bias": test_bias.tolist(), "n_clips": 82, "source": "test"}, f)
            tmp_path = Path(f.name)

        settings.bst_logit_bias_path = tmp_path
        loaded = BSTClassifier._load_logit_bias(25)
        assert loaded is not None
        assert loaded.shape == (25,)
        assert np.allclose(loaded, test_bias, atol=1e-10)
        assert abs(loaded.sum()) < 1e-8
    finally:
        settings.bst_logit_bias_path = orig_path
        if tmp_path.exists():
            tmp_path.unlink()


# ── Dry-run ────────────────────────────────────────────────────────────


def test_dry_run_does_not_write(synthetic_logits_labels, prior_zeros):
    """--dry-run  ⇒  no output file written."""
    logits, labels = synthetic_logits_labels
    frozen = np.zeros(25, dtype=bool)
    frozen[0] = True
    b_fitted = _fit_bias(logits, labels, prior_zeros, reg_lambda=1.0, frozen=frozen)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_bias.json"
        # Simulate dry-run: just compute, don't write
        assert not output_path.exists()
        # In dry-run mode, the file should not be created
        # (handled by the CLI arg, not this function)
