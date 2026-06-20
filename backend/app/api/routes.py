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
    from app.pipeline.court import CourtDetectionStage
    from app.pipeline.players import PlayerTrackingStage
    from app.pipeline.shuttle import ShuttleTrackingStage
    from app.pipeline.pose import PoseEstimationStage
    from app.pipeline.hits import HitFrameLocalizationStage
    from app.pipeline.strokes import StrokeClassificationStage
    from app.pipeline.attribution import PlayerAttributionStage
    from app.pipeline.rallies import RallySegmentationStage
    from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage
    from app.pipeline.analytics.footwork import FootworkAnalyticsStage
    from app.pipeline.analytics.fitness import FitnessAnalyticsStage
    from app.pipeline.analytics.tactical import TacticalAnalyticsStage
    from app.pipeline.analytics.technical import TechnicalAnalyticsStage
    from app.coach.engine import CoachEngine
    from app.storage.artifacts import ArtifactStore
    from app.api.websocket import ws_manager

    job = job_manager.get_job(job_id)
    if not job:
        return

    job_dir = settings.job_dir(job_id)
    store = ArtifactStore(job_dir)
    config = StageConfig(gpu_enabled=False)

    # Get pose_model and sample_rate from job (set by process endpoint)
    pose_model = job.get("pose_model", "rtmpose")
    sample_rate = job.get("sample_rate", 0)

    # Update settings for this pipeline run
    settings.pose_model = pose_model
    settings.sample_rate = sample_rate

    job_manager.update_job(job_id, status="processing", current_stage="court_detection")

    def emit_progress(event):
        ws_manager.broadcast_sync(job_id, event)

    # Get video resolution and extract frames
    video_path = job.get("video_path", "")
    if video_path and Path(video_path).exists():
        vid_w, vid_h = _get_video_resolution(video_path)
        store.set("video_resolution", {"width": vid_w, "height": vid_h})
        # Use sample_rate from job, default to 3 (10fps)
        sample_interval = sample_rate if sample_rate > 0 else 3
        frames = _extract_frames(video_path, sample_interval=sample_interval)
    else:
        frames = []

    # Extract a sample frame for court detection
    court_frame = None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        ret, court_frame = cap.read()
        cap.release()
        if not ret:
            court_frame = None

    stages = [
        ("court_detection", lambda: CourtDetectionStage().run(store, config, frame=court_frame)),
        ("player_tracking", lambda: PlayerTrackingStage().run(store, config, frames=frames if frames else None)),
        ("shuttle_tracking", lambda: ShuttleTrackingStage().run(store, config, frames=frames if frames else None)),
        ("pose_estimation", lambda: PoseEstimationStage().run(store, config, frames=frames if frames else None)),
        ("hit_frame_localization", lambda: HitFrameLocalizationStage().run(store, config)),
        ("stroke_classification", lambda: StrokeClassificationStage().run(store, config)),
        ("rally_segmentation", lambda: RallySegmentationStage().run(store, config)),
        ("player_attribution", lambda: PlayerAttributionStage().run(store, config)),
        ("court_position_analytics", lambda: CourtPositionAnalyticsStage().run(store, config)),
        ("footwork_analytics", lambda: FootworkAnalyticsStage().run(store, config)),
        ("fitness_analytics", lambda: FitnessAnalyticsStage().run(store, config)),
        ("tactical_analytics", lambda: TacticalAnalyticsStage().run(store, config)),
        ("technical_analytics", lambda: TechnicalAnalyticsStage().run(store, config)),
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
    }
    engine = CoachEngine()
    report = engine.generate(analytics, player_id="player_1")

    from app.report.generator import ReportGenerator
    ReportGenerator().generate(job_dir)

    job_manager.update_job(job_id, status="completed", current_stage=None, stages_completed=[s[0] for s in stages])
    emit_progress({"stage": "coach_recommendations", "status": "complete", "metadata": report})


def _get_video_resolution(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


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


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.supported_formats:
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = job_manager.create_job(video_path="", filename=file.filename)

    job_dir = settings.job_dir(job_id)
    video_path = job_dir / f"video.{ext}"
    content = await file.read()
    video_path.write_bytes(content)

    h264_path = job_dir / "video_h264.mp4"
    if _transcode_to_h264(str(video_path), str(h264_path)):
        job_manager.update_job(job_id, video_path=str(h264_path), status="uploaded")
    else:
        job_manager.update_job(job_id, video_path=str(video_path), status="uploaded")

    return {"job_id": job_id, "status": "uploaded", "filename": file.filename}


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
    sample_rate: int = 0
):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("uploaded", "error"):
        raise HTTPException(400, f"Job is already {job['status']}")

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Video not found")

    # Store pose_model and sample_rate in job for pipeline to use
    job_manager.update_job(job_id, pose_model=pose_model, sample_rate=sample_rate)

    background_tasks.add_task(run_pipeline, job_id)
    return {"job_id": job_id, "status": "processing", "pose_model": pose_model, "sample_rate": sample_rate}


from pathlib import Path
from app.report.generator import ReportGenerator
from app.config.settings import settings


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
