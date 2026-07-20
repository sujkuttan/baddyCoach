"""Racket detection tracker (YOLOv8 on RacketDB weights).

Produces per-frame, per-player racket detections with a derived
racket-head point. Returns None gracefully when weights are missing.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.config.settings import settings

logger = logging.getLogger("racket_tracker")


class RacketTracker:
    """Single-class YOLOv8 racket detector with player association."""

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.4,
                 device: str = "cpu"):
        from ultralytics import YOLO
        self.model_path = model_path or settings.racket_model_path
        self.conf = conf
        self.device = device
        self.model = YOLO(self.model_path)

    @staticmethod
    def _head_point(bbox: tuple, margin: float = 0.1) -> tuple:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        h = max(y2 - y1, 1.0)
        head_y = y1 - margin * h
        return (float(cx), float(head_y))

    def detect(self, frames: List[np.ndarray], player_bboxes: dict) -> List[dict]:
        """Detect rackets per frame and associate to nearer player.

        frames: list of BGR images (one per frame index 0..N-1)
        player_bboxes: {frame: {side: bbox_tuple}}
        Returns list of {"frame","player_side","bbox","conf","head_point"}.
        """
        results = self.model(frames, conf=self.conf, device=self.device, verbose=False)
        out: List[dict] = []
        for fi, res in enumerate(results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item()) if box.conf is not None else 1.0
                bbox = (x1, y1, x2, y2)
                head = self._head_point(bbox, margin=settings.racket_head_margin)
                side = self._associate(fi, (x1, y1, x2, y2), player_bboxes)
                out.append({
                    "frame": fi,
                    "player_side": side or "near",
                    "bbox": bbox,
                    "conf": conf,
                    "head_point": head,
                })
        return out

    @staticmethod
    def _associate(frame: int, rbbox: tuple, player_bboxes: dict) -> Optional[str]:
        cands = player_bboxes.get(frame, {})
        if not cands:
            return None
        rcx, rcy = (rbbox[0] + rbbox[2]) / 2.0, (rbbox[1] + rbbox[3]) / 2.0
        best_side, best_d = None, None
        for side, pb in cands.items():
            pcx, pcy = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
            d = (pcx - rcx) ** 2 + (pcy - rcy) ** 2
            if best_d is None or d < best_d:
                best_d, best_side = d, side
        return best_side
