import cv2
import numpy as np
from pathlib import Path


def extract_frames(
    video_path: Path,
    max_frames: int = 200,
    target_fps: int | None = None
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if target_fps and target_fps < source_fps:
        skip = int(source_fps / target_fps)
    else:
        skip = 1
    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['duration'] = info['total_frames'] / info['fps']
    cap.release()
    return info
