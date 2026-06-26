import numpy as np
import pytest


@pytest.mark.model
@pytest.mark.memory_intensive
def test_yolov8_detect_persons():
    from app.models.yolov8 import YOLOv8Detector

    detector = YOLOv8Detector(conf_threshold=0.5)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    detections = detector.detect_persons(frame, frame_idx=0)

    assert isinstance(detections, list)


@pytest.mark.model
@pytest.mark.memory_intensive
def test_yolov8_tracker():
    from app.models.yolov8 import YOLOv8Tracker

    tracker = YOLOv8Tracker(conf_threshold=0.5)

    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]

    results = tracker.track_frames(frames)

    assert isinstance(results, dict)
    assert 'frames' in results
    assert 'tracks' in results
