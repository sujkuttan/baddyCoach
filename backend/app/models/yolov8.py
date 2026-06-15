import os
import numpy as np
from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class Detection:
    frame: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    track_id: int | None = None


class YOLOv8Detector:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        from ultralytics import YOLO
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")

    def detect_persons(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        if self.model is None:
            return []
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False, device=self.device)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(Detection(
                    frame=frame_idx,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                ))
        return detections


class YOLOv8Tracker:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        from ultralytics import YOLO
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")

    def track_frames(self, frames: list[np.ndarray]) -> dict:
        """Track persons across multiple frames.

        Args:
            frames: List of BGR frames

        Returns:
            Dictionary with 'frames' (per-frame detections) and 'tracks' (track IDs)
        """
        all_detections = {}

        for frame_idx, frame in enumerate(frames):
            results = self.model.track(
                frame,
                classes=[0],
                conf=self.conf_threshold,
                verbose=False,
                persist=True,
                device=self.device
            )

            frame_detections = []
            for r in results:
                if r.boxes is not None and r.boxes.id is not None:
                    for i, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        track_id = int(box.id[0].item()) if box.id is not None else None

                        frame_detections.append(Detection(
                            frame=frame_idx,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf,
                            track_id=track_id,
                        ))

            all_detections[frame_idx] = frame_detections

        return {
            "frames": all_detections,
            "tracks": self._extract_tracks(all_detections),
        }

    def _extract_tracks(self, all_detections: dict) -> dict:
        """Extract track trajectories from per-frame detections."""
        tracks = {}
        for frame_idx, detections in all_detections.items():
            for det in detections:
                if det.track_id is not None:
                    if det.track_id not in tracks:
                        tracks[det.track_id] = []
                    tracks[det.track_id].append({
                        "frame": frame_idx,
                        "bbox": det.bbox,
                        "confidence": det.confidence,
                    })
        return tracks
