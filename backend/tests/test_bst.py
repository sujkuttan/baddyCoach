import numpy as np


def test_bst_predict_returns_class():
    from app.models.bst import BSTClassifier, STROKE_CLASSES

    classifier = BSTClassifier()

    features = np.random.rand(144).astype(np.float32)

    stroke_type, confidence = classifier.predict(features)

    assert stroke_type in STROKE_CLASSES
    assert 0 <= confidence <= 1


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
