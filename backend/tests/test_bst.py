import json
import numpy as np
import pytest
from pathlib import Path


def test_bst_predict_returns_class():
    from app.models.bst import BSTClassifier, COACH_STROKE_CLASSES

    classifier = BSTClassifier()

    # Test rule-based fallback (no model loaded)
    clip = {
        'JnB': np.random.rand(30, 2, 72).astype(np.float32),
        'shuttle': np.random.rand(30, 2).astype(np.float32),
        'pos': np.random.rand(30, 2, 2).astype(np.float32),
        'video_len': 30,
    }
    
    stroke_type, confidence, raw_class_id = classifier.predict_single(clip)

    assert stroke_type in COACH_STROKE_CLASSES or stroke_type == "unknown"
    assert 0 <= confidence <= 1
    assert isinstance(raw_class_id, int)


def test_bst_normalize_shuttle():
    from app.models.bst import normalize_shuttlecock

    shuttle = np.array([[100, 200], [150, 250], [200, 300]], dtype=np.float32)
    normalized = normalize_shuttlecock(shuttle, v_width=640, v_height=480)

    assert normalized.shape == (3, 2)
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)


def test_bst_normalize_joints():
    from app.models.bst import normalize_joints

    joints = np.random.rand(2, 17, 2).astype(np.float32) * 500
    bbox = np.array([[100, 100, 300, 400], [400, 100, 600, 400]], dtype=np.float32)

    normalized = normalize_joints(joints, bbox, center_align=True)

    assert normalized.shape == (2, 17, 2)


def test_bst_default_temperature():
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier()
    assert classifier.temperature == 1.0


def test_bst_custom_temperature():
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier(temperature=2.5)
    assert classifier.temperature == 2.5
    classifier2 = BSTClassifier(temperature=0.5)
    assert classifier2.temperature == 0.5


def test_temperature_affects_probs():
    """Verify T != 1.0 changes softmax output."""
    import torch
    from app.models.bst import BSTClassifier, COACH_STROKE_CLASSES

    classifier = BSTClassifier()
    logits = torch.randn(1, 12) * 3.0

    probs_T1 = torch.softmax(logits / 1.0, dim=1).numpy()
    probs_T2 = torch.softmax(logits / 2.0, dim=1).numpy()
    probs_T05 = torch.softmax(logits / 0.5, dim=1).numpy()

    # T=2.0 should have lower max prob than T=1.0 (softer)
    assert probs_T2.max() < probs_T1.max()
    # T=0.5 should have higher max prob than T=1.0 (sharper)
    assert probs_T05.max() > probs_T1.max()


def test_compute_optimal_temperature():
    """Verify temperature calibration returns a reasonable value."""
    pytest.importorskip("sympy", reason="torch LBFGS requires sympy")
    from app.models.bst import BSTClassifier

    rng = np.random.RandomState(42)
    n_classes = 12
    n_samples = 1000

    logits = rng.randn(n_samples, n_classes).astype(np.float32)
    labels = rng.randint(0, n_classes, size=n_samples).astype(np.int64)

    T_opt = BSTClassifier.compute_optimal_temperature(logits, labels)
    assert 0 < T_opt <= 100
    assert isinstance(T_opt, float)


def test_temperature_cache_roundtrip(tmp_path):
    """Verify temperature cache save and reload."""
    from app.models.bst import BSTClassifier

    cache_path = tmp_path / "bst" / "bst_temperature.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    original = BSTClassifier._temperature_cache_path
    BSTClassifier._temperature_cache_path = classmethod(lambda cls: cache_path)

    try:
        BSTClassifier._save_temperature(2.5)
        assert cache_path.exists()

        data = json.loads(cache_path.read_text())
        assert data["temperature"] == 2.5
    finally:
        BSTClassifier._temperature_cache_path = original


def test_temperature_calibration_script_runs():
    """Verify calibration script can run end-to-end with synthetic data."""
    pytest.importorskip("sympy", reason="torch LBFGS requires sympy")
    import subprocess
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    cal_script = scripts_dir / "calibrate_bst.py"
    if not cal_script.exists():
        pytest.skip("calibrate_bst.py not found")

    # Synthetic logits/labels
    tmp = Path(__file__).parent / "_calib_test_data"
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    logits = rng.randn(500, 12).astype(np.float32)
    labels = rng.randint(0, 12, size=500).astype(np.int64)
    logits_path = tmp / "logits.npy"
    labels_path = tmp / "labels.npy"
    np.save(str(logits_path), logits)
    np.save(str(labels_path), labels)

    out_path = tmp / "bst_temperature.json"
    result = subprocess.run(
        [sys.executable, str(cal_script), "--logits", str(logits_path),
         "--labels", str(labels_path), "--output", str(out_path)],
        capture_output=True, text=True, cwd=scripts_dir.parent
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "temperature" in data
    assert 0 < data["temperature"] <= 10

    import shutil
    shutil.rmtree(tmp)


def test_prior_correction_alpha_zero_is_noop():
    """α=0 reproduces the original logits exactly."""
    from app.models.bst import BSTClassifier
    from app.config.settings import settings

    rng = np.random.RandomState(42)
    n_clips = 50
    logits = rng.randn(n_clips, 25).astype(np.float64)

    orig_a = settings.bst_prior_correction_strength
    orig_enabled = settings.bst_prior_correction_enabled
    try:
        settings.bst_prior_correction_strength = 0.0
        settings.bst_prior_correction_enabled = True

        classifier = BSTClassifier()
        corrected = classifier._apply_prior_correction(logits)
        np.testing.assert_array_equal(corrected, logits)
    finally:
        settings.bst_prior_correction_strength = orig_a
        settings.bst_prior_correction_enabled = orig_enabled


def test_prior_correction_removes_constant_bias():
    """Synthetic logits with known bias → bias is removed, per-clip evidence drives argmax."""
    from app.models.bst import BSTClassifier
    from app.config.settings import settings

    rng = np.random.RandomState(42)
    n_clips = 50
    n_classes = 25

    # Create logits where each clip has a distinct true class (signal)
    signal = np.zeros((n_clips, n_classes), dtype=np.float64)
    for i in range(n_clips):
        signal[i, i % n_classes] = 3.0  # per-clip evidence

    # Add a constant class bias (independent of clip)
    bias = rng.randn(n_classes).astype(np.float64) * 2.0
    bias_mc = bias - bias.mean()
    logits = signal + bias_mc[np.newaxis, :]

    orig_a = settings.bst_prior_correction_strength
    orig_enabled = settings.bst_prior_correction_enabled
    orig_min = settings.bst_prior_min_clips
    orig_path = settings.bst_logit_bias_path
    try:
        settings.bst_prior_correction_strength = 1.0
        settings.bst_prior_correction_enabled = True
        settings.bst_prior_min_clips = 10  # low threshold for test
        settings.bst_logit_bias_path = Path("/nonexistent/bst_logit_bias.json")

        classifier = BSTClassifier()

        # Without a bias file, this should self-calibrate (50 clips >= 10)
        corrected = classifier._apply_prior_correction(logits)

        # After correction, each clip's argmax should match the signal's argmax
        pred_signal = np.argmax(signal, axis=1)
        pred_corrected = np.argmax(corrected, axis=1)
        accuracy = (pred_signal == pred_corrected).mean()
        assert accuracy > 0.9, f"Accuracy after correction: {accuracy:.2f}"
    finally:
        settings.bst_prior_correction_strength = orig_a
        settings.bst_prior_correction_enabled = orig_enabled
        settings.bst_prior_min_clips = orig_min
        settings.bst_logit_bias_path = orig_path


def test_prior_correction_skipped_too_few_clips_no_bias_file():
    """No bias file and too few clips → correction is skipped."""
    from app.models.bst import BSTClassifier
    from app.config.settings import settings

    # Temporarily point bias path to a non-existent file
    orig_path = settings.bst_logit_bias_path
    orig_enabled = settings.bst_prior_correction_enabled
    orig_min = settings.bst_prior_min_clips
    try:
        settings.bst_logit_bias_path = Path("/nonexistent/bst_logit_bias.json")
        settings.bst_prior_correction_enabled = True
        settings.bst_prior_min_clips = 100  # more than 5 clips

        classifier = BSTClassifier()
        logits = np.random.randn(5, 25).astype(np.float64)
        corrected = classifier._apply_prior_correction(logits)
        np.testing.assert_array_equal(corrected, logits)
    finally:
        settings.bst_logit_bias_path = orig_path
        settings.bst_prior_correction_enabled = orig_enabled
        settings.bst_prior_min_clips = orig_min


def test_prior_correction_disabled_is_noop():
    """bst_prior_correction_enabled=False → logits pass through unchanged."""
    from app.models.bst import BSTClassifier
    from app.config.settings import settings

    orig_enabled = settings.bst_prior_correction_enabled
    try:
        settings.bst_prior_correction_enabled = False
        classifier = BSTClassifier()
        logits = np.random.randn(10, 25).astype(np.float64)
        corrected = classifier._apply_prior_correction(logits)
        np.testing.assert_array_equal(corrected, logits)
    finally:
        settings.bst_prior_correction_enabled = orig_enabled
