from pathlib import Path
from typing import Callable

from app.pipeline.base import ArtifactStore, PipelineStage, StageConfig, StageResult


class PipelineOrchestrator:
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        self._progress_callbacks: list[Callable] = []

    def on_progress(self, callback: Callable) -> None:
        self._progress_callbacks.append(callback)

    def _emit(self, event: dict) -> None:
        for cb in self._progress_callbacks:
            cb(event)

    def run(self, job_dir: Path, config: StageConfig) -> list[StageResult]:
        artifacts = ArtifactStore(job_dir)
        results: list[StageResult] = []

        for stage in self.stages:
            self._emit({"stage": stage.name, "status": "running"})
            result = stage.run(artifacts, config)
            results.append(result)

            if result.status == "error":
                self._emit({"stage": stage.name, "status": "failed", "error": result.error})
                break
            else:
                self._emit({"stage": stage.name, "status": "complete", "metadata": result.metadata})

        return results
