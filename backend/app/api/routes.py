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

    job_manager.update_job(job_id, status="processing", current_stage="court_detection")

    def emit_progress(event):
        ws_manager.broadcast_sync(job_id, event)

    # Extract frames from video for real inference
    video_path = job.get("video_path", "")
    frames = _extract_frames(video_path, max_frames=200) if video_path else []

    if frames:
        store.set("video_resolution", {
            "width": int(frames[0].shape[1]),
            "height": int(frames[0].shape[0]),
        })

    stages = [
        ("court_detection", lambda: CourtDetectionStage().run(store, config, corners=[
            (100, 500), (1820, 500), (100, 100), (1820, 100)
        ])),
        ("player_tracking", lambda: PlayerTrackingStage().run(store, config, frames=frames if frames else None)),
        ("shuttle_tracking", lambda: ShuttleTrackingStage().run(store, config, frames=frames if frames else None)),
        ("pose_estimation", lambda: PoseEstimationStage().run(store, config, frames=frames if frames else None)),
        ("hit_frame_localization", lambda: HitFrameLocalizationStage().run(store, config)),
        ("stroke_classification", lambda: StrokeClassificationStage().run(store, config)),
        ("player_attribution", lambda: PlayerAttributionStage().run(store, config)),
        ("rally_segmentation", lambda: RallySegmentationStage().run(store, config)),
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


def _extract_frames(video_path: str, max_frames: int = 200) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


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
async def process_job(job_id: str, background_tasks: BackgroundTasks):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("uploaded", "error"):
        raise HTTPException(400, f"Job is already {job['status']}")

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Video not found")

    background_tasks.add_task(run_pipeline, job_id)
    return {"job_id": job_id, "status": "processing"}


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
