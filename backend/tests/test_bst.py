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
