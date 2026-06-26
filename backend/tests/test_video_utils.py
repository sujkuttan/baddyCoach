import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
import cv2


def create_test_video(path: Path, num_frames=30, fps=30, width=640, height=480):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_extract_frames():
    from app.pipeline.video_utils import extract_frames
    with NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)
    create_test_video(video_path, num_frames=30)
    frames = extract_frames(video_path, max_frames=10)
    assert len(frames) == 10
    assert frames[0].shape == (480, 640, 3)
    video_path.unlink()


def test_get_video_info():
    from app.pipeline.video_utils import get_video_info
    with NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = Path(f.name)
    create_test_video(video_path, num_frames=30, fps=30, width=1920, height=1080)
    info = get_video_info(video_path)
    assert info['width'] == 1920
    assert info['height'] == 1080
    assert info['fps'] == 30
    video_path.unlink()
