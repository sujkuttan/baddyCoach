from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PlayerTrackingStage:
    name = "player_tracking"
    input_keys = ["court"]
    output_keys = ["players"]

    def run(self, artifacts: ArtifactStore, config: StageConfig, detections: list[dict] | None = None) -> StageResult:
        if not detections:
            return StageResult.from_error("No player detections provided")

        court = artifacts.get("court")
        court_corners = court.get("corners_pixel", []) if court else []
        if court_corners:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = 300

        players = {}
        for det in detections:
            bbox = det["bbox"]
            center_y = (bbox[1] + bbox[3]) / 2
            side = "near" if center_y > court_mid_y else "far"

            matched = False
            for pid, player in players.items():
                last_bbox = player["detections"][-1]["bbox"]
                iou = self._compute_iou(bbox, last_bbox)
                if iou > 0.3 and player["side"] == side:
                    player["detections"].append(det)
                    matched = True
                    break

            if not matched:
                pid = f"player_{len(players) + 1}"
                players[pid] = {
                    "id": pid,
                    "side": side,
                    "detections": [det],
                }

        players_data = {
            "players": [
                {"id": p["id"], "side": p["side"], "detection_count": len(p["detections"])}
                for p in players.values()
            ],
            "total_frames": max(d["frame"] for d in detections) + 1,
        }

        artifacts.set("players", players_data)

        return StageResult.success(
            artifacts={"players": artifacts.path("players")},
            metadata={"player_count": len(players)}
        )

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
