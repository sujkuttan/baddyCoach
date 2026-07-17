import numpy as np
import pytest
from pathlib import Path


@pytest.mark.model
@pytest.mark.memory_intensive
def test_rtmpose_estimate_keypoints():
    from app.models.rtmpose import RTMPoseEstimator

    model_path = Path("ckpts/rtmpose/rtmpose-m_8xb64-270e_coco-256x192.onnx")
    if not model_path.exists():
        pytest.skip("RTMPose checkpoint not found")
    estimator = RTMPoseEstimator(str(model_path), device="cpu")

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = (100, 100, 300, 400)

    keypoints = estimator.estimate(frame, bbox)

    assert keypoints.shape == (17, 3)
    assert np.all(keypoints[:, 2] >= 0)
    assert np.all(keypoints[:, 2] <= 1)


@pytest.mark.model
@pytest.mark.memory_intensive
def test_rtmpose_estimate_batch():
    from app.models.rtmpose import RTMPoseEstimator

    model_path = Path("ckpts/rtmpose/rtmpose-m_8xb64-270e_coco-256x192.onnx")
    if not model_path.exists():
        pytest.skip("RTMPose checkpoint not found")
    estimator = RTMPoseEstimator(str(model_path), device="cpu")

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [(100, 100, 300, 400), (400, 100, 600, 400)]

    keypoints_list = estimator.estimate_batch(frame, bboxes)

    assert len(keypoints_list) == 2
    for kps in keypoints_list:
        assert kps.shape == (17, 3)


def test_decode_simcc_location_matches_reference():
    from app.models.rtmpose import RTMPoseEstimator

    estimator = RTMPoseEstimator(device="cpu")
    K, Wx, Wy = 17, 384, 512
    rng = np.random.default_rng(0)
    simcc_x = rng.random((K, Wx)).astype(np.float32)
    simcc_y = rng.random((K, Wy)).astype(np.float32)

    keypoints = estimator._decode_simcc(simcc_x, simcc_y)

    x_locs = np.argmax(simcc_x, axis=1).astype(np.float32) / estimator.simcc_split_ratio
    y_locs = np.argmax(simcc_y, axis=1).astype(np.float32) / estimator.simcc_split_ratio
    assert np.allclose(keypoints[:, 0], x_locs)
    assert np.allclose(keypoints[:, 1], y_locs)


def test_decode_simcc_score_uses_mean_of_max_like_reference():
    from app.models.rtmpose import RTMPoseEstimator

    estimator = RTMPoseEstimator(device="cpu")
    K, Wx, Wy = 17, 384, 512
    rng = np.random.default_rng(1)
    simcc_x = rng.random((K, Wx)).astype(np.float32)
    simcc_y = rng.random((K, Wy)).astype(np.float32)

    keypoints = estimator._decode_simcc(simcc_x, simcc_y)

    max_x = np.max(simcc_x, axis=1)
    max_y = np.max(simcc_y, axis=1)
    expected_vals = 0.5 * (max_x + max_y)
    expected_scores = 1.0 / (1.0 + np.exp(-expected_vals))
    assert np.allclose(keypoints[:, 2], expected_scores)
    assert np.all(keypoints[:, 2] >= 0) and np.all(keypoints[:, 2] <= 1)
