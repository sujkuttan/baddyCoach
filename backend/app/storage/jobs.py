import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.settings import settings


class JobManager:
    def __init__(self):
        self.jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, video_path: str, filename: str) -> str:
        job_id = str(uuid.uuid4())[:8]
        settings.job_dir(job_id)
        self.jobs[job_id] = {
            "id": job_id, "filename": filename, "video_path": video_path,
            "status": "uploaded", "current_stage": None, "stages_completed": [],
            "created_at": datetime.now().isoformat(), "error": None,
        }
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    def list_jobs(self) -> list[dict]:
        return list(self.jobs.values())


job_manager = JobManager()