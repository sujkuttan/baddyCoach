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
    
    result = classifier.predict_single(clip)
    stroke_type, confidence, raw_class_id, alpha, aim_attention_p0, aim_attention_p1 = result

    assert isinstance(stroke_type, str) and len(stroke_type) > 0
    assert 0 <= confidence <= 1
    assert isinstance(raw_class_id, int)
    assert 0 <= alpha <= 1
    assert isinstance(aim_attention_p0, float)
    assert isinstance(aim_attention_p1, float)


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


def test_bst_temperature_always_1():
    """BSTClassifier always uses T=1.0 (raw probs); calibration is post-hoc."""
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier(temperature=2.5)
    assert classifier.temperature == 1.0
    classifier2 = BSTClassifier(temperature=0.5)
    assert classifier2.temperature == 1.0


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


def test_calibration_cache_roundtrip(tmp_path):
    """Verify calibration cache save and load_calibration_cache."""
    import json as json_mod
    from app.models.bst import BSTClassifier
    import app.pipeline.shared.models as models_mod

    cache_path = tmp_path / "bst" / "bst_temperature.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    original_path = models_mod.CKPT_DIR
    models_mod.CKPT_DIR = tmp_path

    try:
        BSTClassifier._save_temperature(2.5)
        T_far, T_near = BSTClassifier.load_calibration_cache()
        assert T_far == 2.5
        assert T_near == 2.5

        # Per-side cache
        with open(cache_path, "w") as f:
            json_mod.dump({"temperature_far": 1.3, "temperature_near": 1.8}, f)
        T_far, T_near = BSTClassifier.load_calibration_cache()
        assert T_far == 1.3
        assert T_near == 1.8
    finally:
        models_mod.CKPT_DIR = original_path


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


# ── Calibration tests ────────────────────────────────────────────

def test_calibrate_probs_identity_at_T1():
    """T=1.0 should preserve the argmax of raw logits."""
    from app.models.bst import BSTClassifier
    rng = np.random.RandomState(42)
    logits = rng.randn(25).astype(np.float64)
    raw_probs = np.exp(logits)
    raw_probs = raw_probs / raw_probs.sum()
    raw_top = int(np.argmax(raw_probs))

    calibrated_probs, conf, top3 = BSTClassifier.calibrate_probs(logits, T=1.0)
    assert int(np.argmax(calibrated_probs)) == raw_top
    assert len(top3) == 3
    assert top3[0][1] == conf


def test_calibrate_probs_T2_softer():
    """T > 1 produces softer distribution (lower max confidence)."""
    from app.models.bst import BSTClassifier
    rng = np.random.RandomState(42)
    logits = rng.randn(25).astype(np.float64)
    _, conf_T1, _ = BSTClassifier.calibrate_probs(logits, T=1.0)
    _, conf_T2, _ = BSTClassifier.calibrate_probs(logits, T=2.0)
    assert conf_T2 < conf_T1


def test_calibrate_probs_T05_sharper():
    """T < 1 produces sharper distribution (higher max confidence)."""
    from app.models.bst import BSTClassifier
    rng = np.random.RandomState(42)
    logits = rng.randn(25).astype(np.float64)
    _, conf_T1, _ = BSTClassifier.calibrate_probs(logits, T=1.0)
    _, conf_T05, _ = BSTClassifier.calibrate_probs(logits, T=0.5)
    assert conf_T05 > conf_T1


def test_calibrate_probs_top3_ordering():
    """Top3 returned in descending confidence order."""
    from app.models.bst import BSTClassifier
    rng = np.random.RandomState(42)
    logits = rng.randn(25).astype(np.float64)
    _, _, top3 = BSTClassifier.calibrate_probs(logits, T=1.0)
    assert len(top3) == 3
    assert top3[0][1] >= top3[1][1] >= top3[2][1]


def test_calibrate_probs_valid_probs():
    """Calibrated probabilities sum to 1 and are non-negative."""
    from app.models.bst import BSTClassifier
    rng = np.random.RandomState(42)
    logits = rng.randn(25).astype(np.float64)
    probs, _, _ = BSTClassifier.calibrate_probs(logits, T=1.5)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_normalize_joints_keypoint_bbox():
    """det_bbox=None with margin produces centered in-range values."""
    from app.pipeline.shared.bst_preproc import normalize_joints

    rng = np.random.RandomState(42)
    coords = rng.uniform(200, 800, (17, 2)).astype(np.float32)
    normalized = normalize_joints(coords, det_bbox=None, bbox_margin=0.15)

    assert normalized.shape == (17, 2)
    mean = normalized.mean()
    assert abs(float(mean)) < 0.05, f"Expected centered near 0, got {mean:.4f}"
    vmax = float(np.abs(normalized).max())
    assert vmax < 0.6, f"Expected range < 0.6, got {vmax:.4f}"


def test_normalize_joints_det_bbox():
    """det_bbox provided uses detection bbox (backward compat)."""
    from app.pipeline.shared.bst_preproc import normalize_joints

    rng = np.random.RandomState(42)
    coords = rng.uniform(200, 800, (17, 2)).astype(np.float32)
    det_bbox = (100, 100, 900, 900)
    normalized = normalize_joints(coords, det_bbox=det_bbox, bbox_margin=0.15)

    assert normalized.shape == (17, 2)


def test_normalize_joints_conf_masking():
    """Low-confidence keypoint at origin is zeroed after normalization."""
    from app.pipeline.shared.bst_preproc import normalize_joints

    rng = np.random.RandomState(42)
    coords = rng.uniform(200, 800, (17, 2)).astype(np.float32)
    coords[5] = [0.0, 0.0]  # spurious undetected joint at origin
    conf = np.ones(17, dtype=np.float32)
    conf[5] = 0.0  # zero confidence → masked

    norm_with = normalize_joints(coords, det_bbox=None, bbox_margin=0.15, conf=conf)
    norm_without = normalize_joints(coords, det_bbox=None, bbox_margin=0.15, conf=None)

    # Masked keypoint must be zeroed after normalization
    assert np.all(norm_with[5] == 0.0), f"Masked keypoint should be zeroed, got {norm_with[5]}"

    # Without conf: the all-zero coord is excluded by the zero-coord check,
    # but WITH conf also excludes it. The difference is that conf provides
    # an additional mask for non-zero spurious keypoints.
    # Verify the always-on zero-coord guard works too:
    assert np.all(norm_without[5] == 0.0), (
        "Zero-coord keypoint should be zeroed even without conf"
    )


def test_normalize_joints_conf_masks_nonzero():
    """Low-confidence keypoint at non-zero position is zeroed when conf provided."""
    from app.pipeline.shared.bst_preproc import normalize_joints

    rng = np.random.RandomState(42)
    coords = rng.uniform(200, 800, (17, 2)).astype(np.float32)
    coords[5] = [9999.0, 9999.0]  # spurious keypoint far from body, but non-zero
    conf = np.ones(17, dtype=np.float32)
    conf[5] = 0.0  # zero confidence → masked

    norm_with = normalize_joints(coords, det_bbox=None, bbox_margin=0.15, conf=conf)
    norm_without = normalize_joints(coords, det_bbox=None, bbox_margin=0.15, conf=None)

    # With conf: keypoint 5 zeroed (masked by low confidence)
    assert np.all(norm_with[5] == 0.0), f"Low-conf keypoint should be zeroed, got {norm_with[5]}"

    # Without conf: keypoint 5 NOT zeroed (included since coords are non-zero)
    assert not np.all(norm_without[5] == 0.0), "Non-zero keypoint without conf should NOT be zeroed"


def test_normalize_joints_masks_low_confidence_joint_with_detection_bbox():
    from app.pipeline.shared.bst_preproc import normalize_joints

    coords = np.full((17, 2), [50.0, 50.0], dtype=np.float32)
    coords[9] = [60.0, 60.0]
    coords[10] = [999.0, 999.0]
    confidence = np.ones(17, dtype=np.float32)
    confidence[10] = 0.1

    normalized = normalize_joints(
        coords, det_bbox=(0.0, 0.0, 100.0, 100.0), conf=confidence,
        min_confidence=0.35,
    )

    np.testing.assert_array_equal(normalized[10], [0.0, 0.0])
    assert np.any(normalized[9] != 0.0)


def test_normalize_joints_regression_not_other_player_bbox():
    """Pose of player A should never be normalized by player B's bbox.

    Regression: keypoint-bbox is computed from the coords argument itself,
    so cross-player contamination is structurally impossible. Each player's
    pose maps to its own bbox center regardless of where in the image it is.
    """
    from app.pipeline.shared.bst_preproc import normalize_joints

    # Symmetric keypoint sets in different image regions
    xs_a = np.linspace(200, 400, 17)[:, None]
    ys_a = np.linspace(300, 500, 17)[:, None]
    coords_a = np.concatenate([xs_a, ys_a], axis=1).astype(np.float32)

    xs_b = np.linspace(700, 900, 17)[:, None]
    ys_b = np.linspace(200, 600, 17)[:, None]
    coords_b = np.concatenate([xs_b, ys_b], axis=1).astype(np.float32)

    norm_a = normalize_joints(coords_a, det_bbox=None, bbox_margin=0.15)
    norm_b = normalize_joints(coords_b, det_bbox=None, bbox_margin=0.15)

    # Each normalizes to its own bbox center → both centered near 0
    assert abs(norm_a.mean()) < 0.05, f"Player A not centered: mean={norm_a.mean():.4f}"
    assert abs(norm_b.mean()) < 0.05, f"Player B not centered: mean={norm_b.mean():.4f}"

    # Neither can have extreme values (> 1.0 is always wrong for center-aligned bbox-norm)
    assert np.abs(norm_a).max() < 1.0, f"Player A range too large: {np.abs(norm_a).max():.4f}"
    assert np.abs(norm_b).max() < 1.0, f"Player B range too large: {np.abs(norm_b).max():.4f}"
