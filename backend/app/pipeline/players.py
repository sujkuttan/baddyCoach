import numpy as np
from pathlib import Path

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, NET_HEIGHT
from app.config.settings import settings


class PlayerTrackingStage:
    name = "player_tracking"
    input_keys = ["court"]
    output_keys = ["players"]

    def run(
        self,
        artifacts: ArtifactStore,
        config: StageConfig,
        frames: list[np.ndarray] | None = None,
        detections: list[dict] | None = None
    ) -> StageResult:
        """Run player tracking.

        If frames provided, runs YOLOv8 inference.
        If detections provided, uses pre-computed data.
        """
        court = artifacts.get("court")
        if court is None:
            return StageResult.from_error("Court data required")
        
        # Check if court is valid
        if not court.get("valid", False):
            return StageResult.from_error("Court detection is invalid, cannot track players")

        court_corners = court.get("corners_pixel", []) if court else []
        if court_corners:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = settings.default_frame_height / 2

        if detections:
            return self._process_detections(artifacts, detections, court_mid_y)

        if frames:
            detections = self._run_yolov8(frames)
            if detections:
                detections = self._filter_by_court_region(detections, court_corners)
            if not detections:
                return StageResult.from_error(
                    "YOLOv8 failed to detect any players in the video frames. "
                    "Check video quality, camera angle, or model checkpoint."
                )
            return self._process_detections(artifacts, detections, court_mid_y)

        return StageResult.from_error("No frames or detections provided")

    def _run_yolov8(self, frames: list[np.ndarray]) -> list[dict]:
        """Run YOLOv8 on video frames."""
        from app.models.yolov8 import YOLOv8Tracker
        from app.config.settings import settings

        model_path = str(settings.yolov8_model_path) if settings.yolov8_model_path else None
        device = settings.device
        tracker = YOLOv8Tracker(model_path, conf_threshold=0.5, device=device)

        results = tracker.track_frames(frames)

        detections = []
        for frame_idx, frame_dets in results["frames"].items():
            for det in frame_dets:
                detections.append({
                    "frame": frame_idx,
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "track_id": det.track_id,
                })

        return detections

    def _filter_by_court_region(self, detections: list[dict], court_corners: list) -> list[dict]:
        """Filter detections to keep only those near the court area.

        Uses court corner pixel coordinates to define a court bounding box
        with margin, filtering out detections (umpire, audience) far from court.
        """
        if not court_corners or len(court_corners) < 4:
            return detections
        xs = [c[0] for c in court_corners[:4]]
        ys = [c[1] for c in court_corners[:4]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin_x = (max_x - min_x) * 0.3
        margin_y = (max_y - min_y) * 0.3
        court_min_x = min_x - margin_x
        court_max_x = max_x + margin_x
        court_min_y = min_y - margin_y
        court_max_y = max_y + margin_y
        filtered = []
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if court_min_x <= cx <= court_max_x and court_min_y <= cy <= court_max_y:
                filtered.append(det)
        return filtered if filtered else detections

    def _process_detections(
        self,
        artifacts: ArtifactStore,
        detections: list[dict],
        court_mid_y: float
    ) -> StageResult:
        """Process detections and assign players to sides.

        Uses relative comparison: the detection with larger center_y is 'near',
        smaller center_y is 'far'. This is robust to camera angles where both
        players' bboxes may be on the same side of the court midline.
        """
        if not detections:
            return StageResult.from_error("No player detections provided")

        from collections import defaultdict
        by_frame = defaultdict(list)
        for det in detections:
            by_frame[det["frame"]].append(det)

        players = {}
        for frame_dets in by_frame.values():
            if len(frame_dets) < 2:
                if len(frame_dets) == 1:
                    d = frame_dets[0]
                    center_y = (d["bbox"][1] + d["bbox"][3]) / 2
                    side = "near" if center_y >= court_mid_y else "far"
                    self._add_to_player(players, d, side)
                continue

            sorted_dets = sorted(frame_dets, key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2, reverse=True)
            near_det = sorted_dets[0]
            far_det = sorted_dets[1]
            self._add_to_player(players, near_det, "near")
            self._add_to_player(players, far_det, "far")

        if not players:
            return StageResult.from_error("No player detections grouped")

        players_data = {
            "players": [
                {"id": p["id"], "side": p["side"], "detection_count": len(p["detections"]),
                 "is_synthetic": p.get("is_synthetic", False),
                 "detections": [{"frame": d["frame"], "bbox": d["bbox"], "confidence": d["confidence"],
                                 "is_synthetic": d.get("is_synthetic", False)}
                                for d in p["detections"]]}
                for p in players.values()
            ],
            "total_frames": max(d["frame"] for d in detections) + 1,
        }

        has_synthetic = any(p.get("is_synthetic", False) for p in players_data["players"])

        artifacts.set("players", players_data)

        return StageResult.success(
            artifacts={"players": artifacts.path("players")},
            metadata={"player_count": len(players), "has_synthetic": has_synthetic}
        )

    @staticmethod
    def _add_to_player(players: dict, det: dict, side: str):
        track_id = det.get("track_id")

        if track_id is not None:
            for pid, player in players.items():
                if player.get("track_id") == track_id:
                    player["detections"].append(det)
                    return

        for pid, player in players.items():
            if player["side"] != side:
                continue
            last_bbox = player["detections"][-1]["bbox"]
            iou = PlayerTrackingStage._compute_iou(det["bbox"], last_bbox)
            if iou > 0.3:
                player["detections"].append(det)
                return

        pid = f"player_{len(players) + 1}"
        players[pid] = {
            "id": pid,
            "side": side,
            "track_id": track_id,
            "detections": [det],
            "is_synthetic": det.get("is_synthetic", False),
        }

    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
