import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    frame: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0


class YOLOv8Detector:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.model = None
        if model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    def detect_persons(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        if self.model is None:
            return []
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
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
