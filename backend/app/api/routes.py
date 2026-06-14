from fastapi import APIRouter, UploadFile, File, HTTPException
from app.storage.jobs import job_manager
from app.config.settings import settings

router = APIRouter(prefix="/api")


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