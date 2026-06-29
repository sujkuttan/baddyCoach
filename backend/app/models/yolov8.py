import numpy as np
from dataclasses import dataclass


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
            self.model = YOLO("yolov8s.pt")

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
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.7, device: str = "cpu",
                 chunk_size: int | None = None, batch_size: int | None = None):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        from ultralytics import YOLO
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8s.pt")
        if chunk_size is not None:
            self._chunk_size = chunk_size
        else:
            from app.config.gpu_batch import get_gpu_batch_config
            cfg = get_gpu_batch_config(device)
            self._chunk_size = cfg["yolo_chunk"]
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            from app.config.gpu_batch import get_gpu_batch_config
            cfg = get_gpu_batch_config(device)
            self._batch_size = cfg["yolo_batch"]
        
        # ByteTrack is built into Ultralytics' model.track() with persist=True.
        # Custom config at backend/app/config/bytetrack_badminton.yaml tunes it for badminton.

    def track_frames(self, frames: list[np.ndarray]) -> dict:
        all_detections = {}
        chunk_size = self._chunk_size
        batch_size = self._batch_size

        for chunk_start in range(0, len(frames), chunk_size):
            chunk = frames[chunk_start:chunk_start + chunk_size]
            from app.config.settings import settings
            results = self.model.track(
                chunk,
                classes=[0],
                conf=self.conf_threshold,
                verbose=False,
                persist=True,
                tracker=str(settings.tracker_config_path),
                batch=batch_size,
                stream=True,
                device=self.device,
            )

            for local_idx, r in enumerate(results):
                frame_idx = chunk_start + local_idx
                frame_detections = []
                if r.boxes is not None and r.boxes.id is not None:
                    frame_area = frames[frame_idx].shape[0] * frames[frame_idx].shape[1]
                    dets = []
                    for i, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        track_id = int(box.id[0].item()) if box.id is not None else None
                        bbox_area = (x2 - x1) * (y2 - y1)
                        if bbox_area < frame_area * 0.001 or bbox_area > frame_area * 0.5:
                            continue
                        dets.append(Detection(
                            frame=frame_idx,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf,
                            track_id=track_id,
                        ))
                    dets.sort(key=lambda d: d.confidence, reverse=True)
                    dets = dets[:2]
                    frame_detections = dets

                all_detections[frame_idx] = frame_detections

            import gc, torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return {
            "frames": all_detections,
            "tracks": self._extract_tracks(all_detections),
        }

    def _extract_tracks(self, all_detections: dict) -> dict:
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
