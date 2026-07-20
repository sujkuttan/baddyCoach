"""Racket detection pipeline stage (Scope A feature channel).

Runs the RacketTracker over the (already-YOLO-stitched) frames and stores a
``racket_detections`` artifact consumed by hit-frame localization, stroke
classification, and ownership scoring. Degrades gracefully when racket
detection is disabled or weights are missing.
"""

from __future__ import annotations

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings


class RacketDetectionStage:
    name = "racket_detection"
    input_keys = ["players"]
    output_keys = ["racket_detections"]

    def run(self, artifacts: ArtifactStore, config: StageConfig,
            frames: list | None = None) -> StageResult:
        # Frames are passed by the orchestrator (mirrors player/shuttle/pose
        # stages). When absent (e.g. no video) we cannot run detection.
        if frames is None:
            artifacts.set("racket_detections", [])
            return StageResult.skipped("no frames")

        from app.pipeline.shared.models import get_racket

        tr = get_racket()
        if tr is None:
            artifacts.set("racket_detections", [])
            return StageResult.skipped("racket disabled")

        # Build the per-frame, per-side bbox map RacketTracker._associate needs,
        # reusing the already-stitched players_data so we don't re-run YOLO.
        players_data = artifacts.get("players") or {}
        player_bboxes: dict[int, dict[str, tuple]] = {}
        for p in players_data.get("players", []):
            side = p.get("side", "near")
            for det in p.get("detections", []):
                frame = det.get("frame")
                bbox = det.get("bbox")
                if frame is None or bbox is None:
                    continue
                player_bboxes.setdefault(int(frame), {})[side] = tuple(bbox)

        try:
            dets = tr.detect(frames, player_bboxes)
        except Exception as e:  # defensive: never hard-fail the pipeline on racket
            logger.warning("Racket detection failed (non-fatal)", error=str(e))
            artifacts.set("racket_detections", [])
            return StageResult.skipped("racket detection error")

        artifacts.set("racket_detections", dets)
        logger.info("Racket detection complete", n=len(dets))
        return StageResult.success(metadata={"n_racket_detections": len(dets)})
