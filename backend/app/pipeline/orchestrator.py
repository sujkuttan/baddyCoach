import time
from pathlib import Path
from typing import Callable

from app.pipeline.base import ArtifactStore, PipelineStage, StageConfig, StageResult


class PipelineOrchestrator:
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        self._progress_callbacks: list[Callable] = []
        self.timings: dict[str, float] = {}

    def on_progress(self, callback: Callable) -> None:
        self._progress_callbacks.append(callback)

    def _emit(self, event: dict) -> None:
        for cb in self._progress_callbacks:
            cb(event)

    def run(self, job_dir: Path, config: StageConfig) -> list[StageResult]:
        artifacts = ArtifactStore(job_dir)
        results: list[StageResult] = []
        self.timings = {}

        for stage in self.stages:
            self._emit({"stage": stage.name, "status": "running"})
            t0 = time.time()
            result = stage.run(artifacts, config)
            elapsed = time.time() - t0
            self.timings[stage.name] = elapsed
            results.append(result)

            if result.status == "error":
                self._emit({"stage": stage.name, "status": "failed", "error": result.error, "duration_s": elapsed})
                break
            else:
                self._emit({"stage": stage.name, "status": "complete", "duration_s": elapsed, "metadata": result.metadata})

        # Store timings in artifacts for the report
        artifacts.set("stage_timings", self.timings)

        return results
