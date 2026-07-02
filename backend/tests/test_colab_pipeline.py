import inspect
import cv2
import numpy as np


def _synthetic_court_frame(corners):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for a, b in [(corners[0], corners[1]), (corners[2], corners[3]), (corners[0], corners[2]), (corners[1], corners[3])]:
        cv2.line(frame, tuple(a), tuple(b), (255, 255, 255), 5, cv2.LINE_AA)
    return frame


def test_colab_pipeline_accepts_manual_court_corners():
    import colab.pipeline as pipeline

    assert "court_corners" in inspect.signature(pipeline.run_pipeline).parameters
    corners = pipeline._parse_court_corners_arg("100,500,1180,500,250,150,1030,150")
    assert corners == [(100, 500), (1180, 500), (250, 150), (1030, 150)]


def test_colab_court_detection_uses_shared_hough_trapezoid_detector():
    import colab.pipeline as pipeline
    from app.pipeline.shared.court import compute_homography

    frame = _synthetic_court_frame([[160, 650], [1120, 650], [390, 170], [890, 170]])

    corners = pipeline.detect_court_from_frame(frame)

    assert corners is not None
    H, valid = compute_homography(corners)
    assert H is not None
    assert valid is True
