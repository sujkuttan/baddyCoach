import numpy as np
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
import cv2


def create_test_video(path: Path, num_frames=30, fps=30, width=640, height=480):
    """Create a simple test video with movement."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int(100 + (width - 200) * i / num_frames)
        y = int(height / 2 + 100 * np.sin(i * 0.2))
        cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 200), (150, 350), (0, 0, 255), -1)
        cv2.rectangle(frame, (500, 200), (550, 350), (255, 0, 0), -1)
        out.write(frame)
    out.release()
    return path


@pytest.mark.model
@pytest.mark.integration
@pytest.mark.memory_intensive
def test_real_pipeline_with_models():
    """Test the full pipeline with real models (if available)."""
    from app.pipeline.video_utils import extract_frames, get_video_info

    tracknet_path = Path("ckpts/TrackNet_best.pt")
    if not tracknet_path.exists():
        pytest.skip("TrackNet checkpoint not found")

    with NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)

    create_test_video(video_path, num_frames=30)

    frames = extract_frames(video_path, max_frames=20)
    assert len(frames) > 0

    info = get_video_info(video_path)
    assert info['width'] == 640
    assert info['height'] == 480

    from app.models.tracknet import TrackNetV3
    model = TrackNetV3(str(tracknet_path), device="cpu")
    predictions = model.predict_batch(frames[:10], original_size=(640, 480))
    assert len(predictions) == 8

    from app.models.yolov8 import YOLOv8Detector
    detector = YOLOv8Detector(conf_threshold=0.3)
    detections = detector.detect_persons(frames[0], 0)
    assert isinstance(detections, list)

    rtmpose_path = Path("ckpts/rtmpose/rtmpose-m_8xb64-270e_coco-256x192.onnx")
    if rtmpose_path.exists():
        from app.models.rtmpose import RTMPoseEstimator
        estimator = RTMPoseEstimator(str(rtmpose_path), device="cpu")
        if detections:
            kps = estimator.estimate(frames[0], detections[0].bbox)
            assert kps.shape == (17, 3)

    video_path.unlink()
