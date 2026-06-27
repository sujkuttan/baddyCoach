import asyncio
import numpy as np
import cv2
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.storage.jobs import job_manager
from app.config.settings import settings

router = APIRouter(prefix="/api")


def run_pipeline(job_id: str):
    from app.pipeline.base import StageConfig
    from app.pipeline import CourtDetectionStage
    from app.pipeline import PlayerTrackingStage
    from app.pipeline import ShuttleTrackingStage
    from app.pipeline import PoseEstimationStage
    from app.pipeline import HitFrameLocalizationStage
    from app.pipeline import StrokeClassificationStage
    from app.pipeline import PlayerAttributionStage
    from app.pipeline import RallySegmentationStage
    from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage
    from app.pipeline.analytics.footwork import FootworkAnalyticsStage
    from app.pipeline.analytics.fitness import FitnessAnalyticsStage
    from app.pipeline.analytics.tactical import TacticalAnalyticsStage
    from app.pipeline.analytics.technical import TechnicalAnalyticsStage
    from app.pipeline.analytics.shot_context import ShotContextStage
    from app.pipeline.quality import DataQualityStage
    from app.shuttle_coach.engine import analyze_from_pipeline
    from app.storage.artifacts import ArtifactStore
    from fastapi.responses import Response
    from app.api.websocket import ws_manager
    from app.pipeline.shared.utils import get_video_info
    from app.pipeline.rallies import finalize_rally_outcomes

    job = job_manager.get_job(job_id)
    if not job:
        return

    job_dir = settings.job_dir(job_id)
    store = ArtifactStore(job_dir)

    # Get pose_model and sample_rate from job (set by process endpoint)
    pose_model = job.get("pose_model", "rtmpose")
    sample_rate = job.get("sample_rate", 0)

    # Update settings for this pipeline run
    settings.pose_model = pose_model
    settings.sample_rate = sample_rate

    job_manager.update_job(job_id, status="processing", current_stage="court_detection")

    def emit_progress(event):
        ws_manager.broadcast_sync(job_id, event)

    # Get video info (resolution, actual FPS) and extract frames
    video_path = job.get("video_path", "")
    if video_path and Path(video_path).exists():
        vid_w, vid_h, video_fps = get_video_info(video_path)
        video_fps = int(video_fps) if video_fps > 0 else 30
        store.set("video_resolution", {"width": vid_w, "height": vid_h})
        # Use sample_rate from job, default to 3 (~10fps for 30fps source)
        sample_interval = sample_rate if sample_rate > 0 else max(1, int(video_fps / 10))
        effective_fps = video_fps / sample_interval
        frames = _extract_frames(video_path, sample_interval=sample_interval)
    else:
        frames = []
        effective_fps = 30.0

    config = StageConfig(gpu_enabled=settings.gpu_enabled, processing_fps=max(1, int(effective_fps)))

    # Extract a sample frame for court detection
    court_frame = None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        ret, court_frame = cap.read()
        cap.release()
        if not ret:
            court_frame = None

    # Use manual corners if available, otherwise fall back to auto-detection
    manual_corners = job.get("manual_corners")
    court_kwargs = {"frame": court_frame} if manual_corners is None else {"corners": manual_corners}

    def _run_rally_finalization(store, config):
        rallies_df = store.get("rallies")
        shots_df = store.get("shots")
        shuttle_df = store.get("shuttle")
        court = store.get("court")
        players = store.get("players")
        if rallies_df is None or len(rallies_df) == 0:
            return StageResult.success(artifacts={})
        rallies_df = finalize_rally_outcomes(
            rallies_df, shots_df, shuttle_raw=shuttle_df,
            court=court, players=players, fps=config.processing_fps or 30.0,
        )
        store.set_parquet("rallies", rallies_df)
        return StageResult.success(artifacts={"rallies": store.path("rallies")})

    stages = [
        ("court_detection", lambda: CourtDetectionStage().run(store, config, **court_kwargs)),
        ("player_tracking", lambda: PlayerTrackingStage().run(store, config, frames=frames if frames else None)),
        ("shuttle_tracking", lambda: ShuttleTrackingStage().run(store, config, frames=frames if frames else None)),
        ("pose_estimation", lambda: PoseEstimationStage().run(store, config, frames=frames if frames else None)),
        ("hit_frame_localization", lambda: HitFrameLocalizationStage().run(store, config)),
        ("stroke_classification", lambda: StrokeClassificationStage().run(store, config)),
        ("rally_segmentation", lambda: RallySegmentationStage().run(store, config)),
        ("player_attribution", lambda: PlayerAttributionStage().run(store, config)),
        ("rally_finalization", lambda: _run_rally_finalization(store, config)),
        ("shot_context", lambda: ShotContextStage().run(store, config)),
        ("court_position_analytics", lambda: CourtPositionAnalyticsStage().run(store, config)),
        ("footwork_analytics", lambda: FootworkAnalyticsStage().run(store, config)),
        ("fitness_analytics", lambda: FitnessAnalyticsStage().run(store, config)),
        ("tactical_analytics", lambda: TacticalAnalyticsStage().run(store, config)),
        ("technical_analytics", lambda: TechnicalAnalyticsStage().run(store, config)),
        ("data_quality", lambda: DataQualityStage().run(store, config)),
    ]

    for stage_name, stage_fn in stages:
        try:
            job_manager.update_job(job_id, current_stage=stage_name)
            emit_progress({"stage": stage_name, "status": "running"})
            result = stage_fn()
            if result.status == "error":
                job_manager.update_job(job_id, status="error", error=result.error, current_stage=None)
                emit_progress({"stage": stage_name, "status": "failed", "error": result.error})
                return
            emit_progress({"stage": stage_name, "status": "complete", "metadata": result.metadata})
        except Exception as e:
            job_manager.update_job(job_id, status="error", error=str(e), current_stage=None)
            emit_progress({"stage": stage_name, "status": "failed", "error": str(e)})
            return

    # Generate coach report
    analytics = {
        "fitness_analytics": store.get("fitness_analytics") or {},
        "tactical_analytics": store.get("tactical_analytics") or {},
        "footwork_analytics": store.get("footwork_analytics") or {},
        "court_analytics": store.get("court_analytics") or {},
        "_rallies_df": store.get_parquet("rallies"),
        "_shots_df": store.get_parquet("shots"),
    }

    # Get data quality from the DataQualityStage
    data_quality = store.get("data_quality") or {}

    # Try to get shuttle_coach metrics for richer coaching
    shuttle_metrics = {}
    try:
        from app.shuttle_coach.engine import analyze
        sc_result = analyze(str(job_dir))
        for m in sc_result.get("metrics", []):
            pid = m.get("player_id", "player_1")
            mid = m.get("metric_id", "")
            val = m.get("value")
            if pid not in shuttle_metrics:
                shuttle_metrics[pid] = {}
            shuttle_metrics[pid][mid] = val
    except Exception:
        pass

    all_players = set(list(analytics["tactical_analytics"].keys()) + list(analytics["fitness_analytics"].keys()))
    if not all_players:
        all_players = {"player_1"}

    report = {
        "strengths": [], "weaknesses": [], "top_3_improvements": [],
        "recommended_drills": [], "recommended_drills_detailed": [],
        "evidence": [], "rally_stats": None, "patterns": [],
        "technique_reference": [], "progress": [],
    }
    for pid in sorted(all_players):
        result = analyze_from_pipeline(analytics, shuttle_metrics, player_id=pid, data_quality=data_quality)
        report["strengths"].extend(result["strengths"])
        report["weaknesses"].extend(result["weaknesses"])
        report["top_3_improvements"].extend(result["top_3_improvements"])
        report["recommended_drills"].extend(result["recommended_drills"])
        report["recommended_drills_detailed"].extend(result.get("recommended_drills_detailed", []))
        report["evidence"].extend(result["evidence"])
        if result.get("rally_stats") and report["rally_stats"] is None:
            report["rally_stats"] = result["rally_stats"]

    # ── Patterns section ─────────────────────────────────────────
    try:
        sc_result = analyze(str(job_dir))
        pattern_findings = [f for f in sc_result.get("findings", [])
                           if f.get("code", "").startswith("pattern::")]
        report["patterns"] = pattern_findings
    except Exception:
        pass

    # ── Technique reference section ──────────────────────────────
    tech_ref = []
    for pid in sorted(all_players):
        for mid, val in shuttle_metrics.get(pid, {}).items():
            if mid == "technique.reference" and isinstance(val, dict):
                tech_ref.append({"player_id": pid, **val})
    report["technique_reference"] = tech_ref

    # ── Progress section ─────────────────────────────────────────
    try:
        from app.storage.progress import compare_last_n
        progress_data = {}
        for pid in sorted(all_players):
            headlines = compare_last_n(pid, n=5, player_id=pid)
            if headlines:
                progress_data[pid] = headlines
        report["progress"] = progress_data
    except Exception:
        pass

    # Include data quality in the report
    if data_quality:
        report["data_quality"] = data_quality

    # Try to generate Gemini narration (rule-aware) and add to report
    narration_text = _generate_narration(job_id, store, report)
    if narration_text:
        report["narration"] = narration_text

    store.set("report", report)

    # Save cross-session progress data with structured snapshots
    try:
        from app.storage.progress import save_player_session
        player_key = job.get("player_key", "") or job_id[:8]
        for pid in sorted(all_players):
            save_player_session(f"{player_key}_{pid}", job_id, analytics, data_quality=data_quality)
    except Exception:
        pass

    from app.report.generator import ReportGenerator
    ReportGenerator().generate(job_dir)

    job_manager.update_job(job_id, status="completed", current_stage=None, stages_completed=[s[0] for s in stages])
    emit_progress({"stage": "coach_recommendations", "status": "complete", "metadata": report})


def _extract_frames(video_path: str, sample_interval: int = 3) -> list[np.ndarray]:
    """Extract all frames from video, sampling every Nth frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def _transcode_to_h264(input_path: str, output_path: str) -> bool:
    """Re-encode video to H.264 for browser compatibility. Returns True on success."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-y", output_path],
            capture_output=True, timeout=600
        )
        return result.returncode == 0
    except Exception:
        return False


def _generate_narration(job_id: str, store, rule_report: dict | None = None) -> str | None:
    """Generate Gemini narration for the main coaching report.

    Incorporates rule-based findings as context so Gemini can reference them.
    """
    import os
    api_key = os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key
    if not api_key:
        return None
    try:
        from app.shuttle_coach.engine import analyze, narrate
        result = analyze(str(settings.job_dir(job_id)))
        metrics = result.get("metrics", [])
        if not metrics:
            return None

        # Build a question that includes the rule-based findings as context
        strengths_text = ""
        weaknesses_text = ""
        if rule_report:
            ss = rule_report.get("strengths", [])
            ws = rule_report.get("weaknesses", [])
            strengths_text = "Strengths detected: " + "; ".join(ss[:3]) if ss else ""
            weaknesses_text = "Areas to improve: " + "; ".join(ws[:3]) if ws else ""

        question = (
            "Provide a concise coaching summary of this badminton match. "
            "Highlight key strengths and areas for improvement. "
            "Be specific and actionable. "
        )
        if strengths_text or weaknesses_text:
            question += f"\n\nRule-based analysis context:\n{strengths_text}\n{weaknesses_text}"

        return narrate(question, metrics, api_key)
    except Exception:
        return None


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...), player_key: str = ""):
    if not file.filename:
        raise HTTPException(400, "No filename")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.supported_formats:
        raise HTTPException(400, f"Unsupported format: {ext}")

    # Validate file size (max 500MB)
    MAX_SIZE = 500 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(400, f"File too large: {len(content)} bytes (max {MAX_SIZE} bytes)")
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    # Validate content is a video via magic bytes
    MAGIC_BYTES = {
        b"\x00\x00\x00\x18ftypmp4": "mp4",
        b"\x00\x00\x00\x20ftyp": "mp4",
        b"\x00\x00\x00\x1cftyp": "mp4",
        b"\x1a\x45\xdf\xa3": "mkv/webm",
        b"\x52\x49\x46\x46": "avi",
    }
    is_valid = False
    for magic, fmt in MAGIC_BYTES.items():
        if content[:len(magic)] == magic:
            is_valid = True
            break
    if not is_valid and not (content[:3] == b"\x00\x00\x00" and b"ftyp" in content[:32]):
        is_valid = True  # lenient — accept unknown containers

    import uuid
    player_key = player_key.strip() or f"player_{uuid.uuid4().hex[:8]}"
    job_id = job_manager.create_job(video_path="", filename=file.filename)
    job_manager.update_job(job_id, player_key=player_key)

    job_dir = settings.job_dir(job_id)
    video_path = job_dir / f"video.{ext}"
    video_path.write_bytes(content)

    h264_path = job_dir / "video_h264.mp4"
    if _transcode_to_h264(str(video_path), str(h264_path)):
        job_manager.update_job(job_id, video_path=str(h264_path), status="uploaded")
    else:
        job_manager.update_job(job_id, video_path=str(video_path), status="uploaded")

    return {"job_id": job_id, "status": "uploaded", "filename": file.filename, "player_key": player_key}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs")
async def list_jobs():
    return {"jobs": job_manager.list_jobs()}


@router.post("/jobs/{job_id}/process")
async def process_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    pose_model: str = "rtmpose",
    sample_rate: int = 0,
    player_key: str = "",
):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("uploaded", "error"):
        raise HTTPException(400, f"Job is already {job['status']}")

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Video not found")

    # Use player_key from process param, then from upload, then fallback
    existing_key = job.get("player_key", "")
    resolved_key = player_key.strip() or existing_key or f"player_{job_id[:8]}"
    job_manager.update_job(job_id, pose_model=pose_model, sample_rate=sample_rate, player_key=resolved_key)

    background_tasks.add_task(run_pipeline, job_id)
    return {"job_id": job_id, "status": "processing", "pose_model": pose_model, "sample_rate": sample_rate, "player_key": resolved_key}


from app.report.generator import ReportGenerator


report_generator = ReportGenerator()


@router.get("/jobs/{job_id}/report")
async def get_report(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    job_dir = settings.job_dir(job_id)
    report_path = job_dir / "report.json"

    if report_path.exists():
        import json
        return json.loads(report_path.read_text())

    report = report_generator.generate(job_dir)
    return report


@router.get("/jobs/{job_id}/video")
async def stream_video(job_id: str):
    from fastapi.responses import FileResponse
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Video not found")

    return FileResponse(video_path)


@router.get("/jobs/{job_id}/frame")
async def get_first_frame(job_id: str):
    """Return the first video frame as JPEG for court-corner setup."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Video not found")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(500, "Could not read first frame")

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.put("/jobs/{job_id}/court-corners")
async def set_court_corners(job_id: str, corners: list[list[int]]):
    """Store manual court corners for a job. Expected: [[bl_x,bl_y], [br_x,br_y], [tl_x,tl_y], [tr_x,tr_y]]."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if len(corners) != 4:
        raise HTTPException(400, "Exactly 4 corner points required: [bl, br, tl, tr]")

    job_manager.update_job(job_id, manual_corners=corners)
    return {"job_id": job_id, "corners": corners}


@router.get("/shuttle-coach/analyze/{job_id}")
async def analyze_shuttle_coach(job_id: str, question: str = None):
    """Run shuttle-coach analysis on a completed job."""
    import os
    from app.shuttle_coach.engine import analyze, narrate

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    job_dir = settings.job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(404, "Job directory not found")

    try:
        result = analyze(str(job_dir))
    except FileNotFoundError as e:
        raise HTTPException(400, f"Missing parquet files: {e}")
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    if question and os.environ.get("GEMINI_API_KEY"):
        try:
            result["narration"] = narrate(
                question, result["metrics"], os.environ["GEMINI_API_KEY"]
            )
        except Exception as e:
            result["narration_error"] = str(e)

    return result


@router.get("/players/{player_key}/progress")
async def get_player_progress(player_key: str, window: int = 5):
    """Get cross-session progress trends for a player."""
    from app.storage.progress import (
        get_player_history, compute_metric_trend, compare_last_n,
    )

    history = get_player_history(player_key)
    if len(history) < 2:
        return {"n_sessions": len(history), "trends": [], "headlines": [],
                "detail": "Analyze at least 2 sessions to see progress trends"}

    trends = {}
    tracked_metrics = [
        "fitness.rally_intensity", "fitness.peak_intensity",
        "fitness.total_distance", "footwork.avg_recovery",
        "fitness.late_rally_fatigue", "tactical.total_shots",
    ]
    for kp in tracked_metrics:
        trend = compute_metric_trend(player_key, kp, window=window)
        if trend["direction"] != "insufficient_data":
            trends[kp] = trend

    headlines = compare_last_n(player_key, n=window)

    # Compute simple sparklines per metric
    sparklines = {}
    for kp, trend in trends.items():
        sparklines[kp] = trend.get("sparkline", [])

    return {
        "n_sessions": len(history),
        "trends": trends,
        "headlines": headlines[:5],
        "sparklines": sparklines,
        "player_key": player_key,
    }
