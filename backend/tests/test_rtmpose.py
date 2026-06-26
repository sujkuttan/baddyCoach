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
