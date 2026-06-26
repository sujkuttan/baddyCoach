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
        from app.pipeline.shared.models import get_yolov8
        tracker = get_yolov8()
        if tracker is None:
            return []

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

        Stitches track-ID fragments into exactly 2 persistent identities
        (near/far) using per-frame side assignment + centroid distance continuity.
        """
        if not detections:
            return StageResult.from_error("No player detections provided")

        if settings.track_stitch_enabled:
            players_data = self._stitch_tracks(detections, court_mid_y)
        else:
            players_data = self._group_by_track_id(detections, court_mid_y)

        artifacts.set("players", players_data)

        has_synthetic = any(p.get("is_synthetic", False) for p in players_data["players"])
        return StageResult.success(
            artifacts={"players": artifacts.path("players")},
            metadata={"player_count": len(players_data["players"]), "has_synthetic": has_synthetic}
        )

    def _stitch_tracks(
        self, detections: list[dict], court_mid_y: float
    ) -> dict:
        """Stitch track-ID fragments into exactly 2 persistent players.

        Per-frame side test (cy >= court_mid_y → near) is the primary signal;
        centroid distance continuity prevents side-flips when a player briefly
        crosses the midline.
        """
        # Assign side per detection and sort by frame
        for d in detections:
            cy = (d["bbox"][1] + d["bbox"][3]) / 2
            d["side"] = "near" if cy >= court_mid_y else "far"

        detections = sorted(detections, key=lambda d: d["frame"])

        # Two persistent tracks
        tracks = {
            "near": {"id": "player_1", "side": "near", "detections": [], "last_center": None},
            "far": {"id": "player_2", "side": "far", "detections": [], "last_center": None},
        }

        for det in detections:
            side = det["side"]
            track = tracks[side]
            cx = (det["bbox"][0] + det["bbox"][2]) / 2
            cy = (det["bbox"][1] + det["bbox"][3]) / 2

            # Distance check: skip assignment to opposite side if within threshold
            # (prevents brief midline crossings from flipping identity)
            if track["last_center"] is not None:
                last_cx, last_cy = track["last_center"]
                dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                if dist > settings.track_stitch_max_dist_px:
                    # Try the other track — it might be the correct match
                    other_side = "far" if side == "near" else "near"
                    other = tracks[other_side]
                    if other["last_center"] is not None:
                        olx, oly = other["last_center"]
                        odist = np.sqrt((cx - olx) ** 2 + (cy - oly) ** 2)
                        if odist <= settings.track_stitch_max_dist_px:
                            track = other
                            side = other_side

            track["detections"].append(det)
            track["last_center"] = (cx, cy)

        total_frames = max(d["frame"] for d in detections) + 1

        players_list = []
        for side in ("near", "far"):
            t = tracks[side]
            dets = sorted(t["detections"], key=lambda d: d["frame"])
            players_list.append({
                "id": t["id"],
                "side": side,
                "detection_count": len(dets),
                "is_synthetic": any(d.get("is_synthetic", False) for d in dets),
                "detections": [
                    {"frame": d["frame"], "bbox": d["bbox"],
                     "confidence": d["confidence"],
                     "is_synthetic": d.get("is_synthetic", False)}
                    for d in dets
                ],
            })

        return {"players": players_list, "total_frames": total_frames}

    def _group_by_track_id(
        self, detections: list[dict], court_mid_y: float
    ) -> dict:
        """Original grouping logic: group by track_id, up to max_players.

        Used when stitching is disabled (track_stitch_enabled=False).
        """
        from collections import defaultdict
        track_groups = defaultdict(list)
        for det in detections:
            tid = det.get("track_id")
            if tid is not None:
                gid = f"track_{tid}"
            else:
                cy = (det["bbox"][1] + det["bbox"][3]) / 2
                side = "near" if cy >= court_mid_y else "far"
                gid = f"frame_{det['frame']}_{side}"
            track_groups[gid].append(det)

        players = {}
        max_players = settings.max_players

        for gid, group in track_groups.items():
            if len(players) >= max_players:
                break
            center_ys = [(d["bbox"][1] + d["bbox"][3]) / 2 for d in group]
            median_cy = float(np.median(center_ys))
            side = "near" if median_cy >= court_mid_y else "far"
            pid = f"player_{len(players) + 1}"
            players[pid] = {
                "id": pid,
                "side": side,
                "track_id": gid if isinstance(gid, int) else None,
                "detections": sorted(group, key=lambda d: d["frame"]),
                "is_synthetic": any(d.get("is_synthetic", False) for d in group),
            }

        if not players:
            return {"players": [], "total_frames": 0}

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
        return players_data
