import numpy as np
from pathlib import Path
from collections import Counter

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, NET_HEIGHT
from app.config.settings import settings
from app.pipeline.shared.logging import logger


def stitch_tracks(
    detections: list[dict],
    court_mid_y: float,
    scene_cut_gap_frames: int | None = None,
    scene_cut_jump_px: float | None = None,
) -> dict:
    """Stitch track-ID fragments into exactly 2 persistent players.

    Joint per-frame assignment: for each frame with ≤2 detections, assign
    all detections to tracks simultaneously by nearest-centroid matching.
    This guarantees 1 detection per track per frame whenever 2 are present
    (unlike independent side tests which lose detections when both players
    are on the same side of the midline).

    Scene-cut hardening: when a discontinuity is detected (a frame gap larger
    than ``scene_cut_gap_frames`` or a teleport where both detections jump
    farther than ``scene_cut_jump_px`` from every track's last center), the
    matching context is reset so stale centroids from before the cut cannot
    swap the two players' identities. After a reset the tracks are re-seeded
    by **court side** (centroid y vs ``court_mid_y``), not by left/right
    order, so a player who has physically swapped halves across a cut keeps
    the same persistent track rather than flipping with the opposite player.

    Side labels are resolved globally from each track's median court-y at the
    end, so ``player_1`` is always the near-side (lower) player and
    ``player_2`` the far-side player — independent of the arbitrary initial
    track index. This preserves the downstream invariant
    (player_1 == near, player_2 == far) even when the near player starts on
    the right.

    Args:
        detections: List of detection dicts, each with 'frame', 'bbox', 'confidence'.
        court_mid_y: Y-pixel coordinate of the court midline.
        scene_cut_gap_frames: Frame-gap threshold for a discontinuity (default
            ``settings.player_stitch_scene_cut_gap_frames``).
        scene_cut_jump_px: Teleport threshold in pixels (default
            ``settings.player_stitch_scene_cut_jump_px``).

    Returns:
        dict with 'players' list (2 entries, id/side/detection_count/detections),
        'total_frames', and 'scene_cut_count'.
    """
    if scene_cut_gap_frames is None:
        scene_cut_gap_frames = settings.player_stitch_scene_cut_gap_frames
    if scene_cut_jump_px is None:
        scene_cut_jump_px = settings.player_stitch_scene_cut_jump_px

    # Group by frame
    frames: dict[int, list[dict]] = {}
    for d in detections:
        frames.setdefault(d["frame"], []).append(d)

    # Two persistent tracks (side resolved globally at the end)
    tracks = [
        {"id": "player_1", "side": None, "detections": [], "last_center": None},
        {"id": "player_2", "side": None, "detections": [], "last_center": None},
    ]

    def _centroid(det):
        return np.array([(det["bbox"][0] + det["bbox"][2]) / 2,
                          (det["bbox"][1] + det["bbox"][3]) / 2])

    def _is_scene_cut(prev_frame, frame_idx, cand_centroids):
        """True if the gap from the previous frame is a discontinuity."""
        if prev_frame is None:
            return True
        if (frame_idx - prev_frame) > scene_cut_gap_frames:
            return True
        if (scene_cut_jump_px > 0 and
                tracks[0]["last_center"] is not None and
                tracks[1]["last_center"] is not None):
            # Teleport: every candidate is far from BOTH tracks' last centers.
            for c in cand_centroids:
                min_d = min(
                    np.linalg.norm(c - tracks[0]["last_center"]),
                    np.linalg.norm(c - tracks[1]["last_center"]),
                )
                if min_d <= scene_cut_jump_px:
                    return False
            return True
        return False

    scene_cut_count = 0
    prev_frame = None
    for frame_idx in sorted(frames.keys()):
        frame_dets = frames[frame_idx]
        n = len(frame_dets)

        if n == 1:
            cands = [(_centroid(frame_dets[0]), frame_dets[0])]
        else:
            sorted_dets = sorted(frame_dets, key=lambda x: x.get("confidence", 0), reverse=True)
            cands = [(_centroid(d), d) for d in sorted_dets[:2]]

        is_cut = _is_scene_cut(prev_frame, frame_idx, [c for c, _ in cands])
        if is_cut:
            tracks[0]["last_center"] = None
            tracks[1]["last_center"] = None
            if prev_frame is not None:
                scene_cut_count += 1
        prev_frame = frame_idx

        if n == 1:
            c, det = cands[0]
            if tracks[0]["last_center"] is not None and tracks[1]["last_center"] is not None:
                d0 = np.linalg.norm(c - tracks[0]["last_center"])
                d1 = np.linalg.norm(c - tracks[1]["last_center"])
                idx = 0 if d0 <= d1 else 1
            else:
                idx = 0 if c[1] >= court_mid_y else 1
            tracks[idx]["detections"].append(det)
            tracks[idx]["last_center"] = c

        elif n >= 2:
            (ca, det_a), (cb, det_b) = cands
            if tracks[0]["last_center"] is not None and tracks[1]["last_center"] is not None and not is_cut:
                d_an = np.linalg.norm(ca - tracks[0]["last_center"])
                d_af = np.linalg.norm(ca - tracks[1]["last_center"])
                d_bn = np.linalg.norm(cb - tracks[0]["last_center"])
                d_bf = np.linalg.norm(cb - tracks[1]["last_center"])

                if d_an + d_bf <= d_af + d_bn:
                    pairs = [(det_a, 0), (det_b, 1)]
                else:
                    pairs = [(det_a, 1), (det_b, 0)]
            else:
                # Re-seed by court side (lower y == farther) so a player who
                # swapped halves across a cut keeps the same persistent track.
                if ca[1] >= cb[1]:
                    pairs = [(det_a, 0), (det_b, 1)]
                else:
                    pairs = [(det_a, 1), (det_b, 0)]

            for det, idx in pairs:
                tracks[idx]["detections"].append(det)
                tracks[idx]["last_center"] = _centroid(det)

    # Global side resolution: side follows court position, not track index.
    tracks = _resolve_sides(tracks, court_mid_y)

    total_frames = max(frames.keys()) + 1 if frames else 0

    players_list = []
    for track in tracks:
        dets = sorted(track["detections"], key=lambda d: d["frame"])
        players_list.append({
            "id": track["id"],
            "side": track["side"],
            "detection_count": len(dets),
            "is_synthetic": any(d.get("is_synthetic", False) for d in dets),
            "detections": [
                {
                    "frame": d["frame"],
                    "bbox": d["bbox"],
                    "confidence": d["confidence"],
                    "is_synthetic": d.get("is_synthetic", False),
                    "center_px": [(d["bbox"][0] + d["bbox"][2]) / 2.0,
                                  (d["bbox"][1] + d["bbox"][3]) / 2.0],
                    "foot_point_px": [(d["bbox"][0] + d["bbox"][2]) / 2.0,
                                      float(d["bbox"][3])],
                    "bbox_height": float(d["bbox"][3] - d["bbox"][1]),
                }
                for d in dets
            ],
        })

    return {"players": players_list, "total_frames": total_frames,
            "scene_cut_count": scene_cut_count}


def _resolve_sides(tracks: list[dict], court_mid_y: float) -> list[dict]:
    """Assign near/far sides from each track's median court-y and keep the
    player_1 == near / player_2 == far invariant.

    The two tracks are reordered (and their ``id`` fields rewritten) so that
    the lower (nearer) track becomes ``player_1``. This makes side a property
    of the physical player's court position rather than the arbitrary initial
    track index, fixing cases where the near player starts on the right.
    """
    meds = []
    for t in tracks:
        ys = [(d["bbox"][1] + d["bbox"][3]) / 2.0 for d in t["detections"]]
        meds.append(float(np.median(ys)) if ys else court_mid_y)

    # Larger median y == nearer to camera == "near" side; it leads.
    order = sorted(range(len(tracks)), key=lambda i: meds[i], reverse=True)
    for rank, i in enumerate(order):
        tracks[i]["side"] = "near" if rank == 0 else "far"

    reordered = [tracks[order[0]], tracks[order[1]]]
    reordered[0]["id"] = "player_1"
    reordered[1]["id"] = "player_2"
    return reordered


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

        if not court.get("valid", False):
            logger.warning("Court geometry invalid; continuing player tracking with pixel midline fallback")
        
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

        # Log raw ByteTrack fragmentation
        id_counts = Counter(
            det.track_id for frame_dets in results["frames"].values()
            for det in frame_dets if det.track_id is not None
        )
        n_frames_with_dets = sum(1 for v in results["frames"].values() if v)
        logger.info(
            "ByteTrack raw tracking",
            unique_ids=len(id_counts),
            frames_with_detections=n_frames_with_dets,
            detections=sum(id_counts.values()),
        )
        if id_counts:
            # Show the most fragmented IDs (fewest detections each)
            small_ids = {k: v for k, v in id_counts.items() if v < 10}
            if small_ids:
                logger.info(
                    "ByteTrack fragmentation",
                    short_lived_id_count=len(small_ids),
                    id_detection_counts=dict(sorted(small_ids.items(), key=lambda item: item[1])),
                )

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
        Enriches each detection with derived fields (center, foot, height, court coords).
        """
        if not detections:
            return StageResult.from_error("No player detections provided")

        if settings.track_stitch_enabled:
            players_data = self._stitch_tracks(detections, court_mid_y)
            logger.info(
                "Player tracking stitched",
                scene_cut_count=players_data.get("scene_cut_count", 0),
                players=len(players_data.get("players", [])),
            )
        else:
            players_data = self._group_by_track_id(detections, court_mid_y)

        # Enrich with court-space coordinates when homography is available
        court = artifacts.get("court")
        if court and court.get("valid", False) and court.get("homography") is not None:
            H = np.array(court["homography"])
            from app.pipeline.shared.court import image_to_court
            for player in players_data.get("players", []):
                for det in player.get("detections", []):
                    bbox = det.get("bbox")
                    if bbox and len(bbox) == 4:
                        foot_px = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
                        try:
                            cx, cy = image_to_court(H, foot_px)
                            det["x_court"] = round(cx, 3)
                            det["y_court"] = round(cy, 3)
                        except Exception:
                            pass

        artifacts.set("players", players_data)

        has_synthetic = any(p.get("is_synthetic", False) for p in players_data["players"])
        return StageResult.success(
            artifacts={"players": artifacts.path("players")},
            metadata={"player_count": len(players_data["players"]), "has_synthetic": has_synthetic}
        )

    def _stitch_tracks(
        self, detections: list[dict], court_mid_y: float
    ) -> dict:
        return stitch_tracks(detections, court_mid_y)

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
                {
                    "id": p["id"], "side": p["side"], "detection_count": len(p["detections"]),
                    "is_synthetic": p.get("is_synthetic", False),
                    "detections": [
                        {
                            "frame": d["frame"], "bbox": d["bbox"],
                            "confidence": d["confidence"],
                            "is_synthetic": d.get("is_synthetic", False),
                            "center_px": [(d["bbox"][0] + d["bbox"][2]) / 2.0,
                                          (d["bbox"][1] + d["bbox"][3]) / 2.0],
                            "foot_point_px": [(d["bbox"][0] + d["bbox"][2]) / 2.0,
                                              float(d["bbox"][3])],
                            "bbox_height": float(d["bbox"][3] - d["bbox"][1]),
                        }
                        for d in p["detections"]
                    ],
                }
                for p in players.values()
            ],
            "total_frames": max(d["frame"] for d in detections) + 1,
        }
        return players_data
