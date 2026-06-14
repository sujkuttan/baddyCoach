from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.orchestrator import PipelineOrchestrator


class MockStage:
    name = "mock_stage"
    input_keys = []
    output_keys = ["mock_output"]

    def __init__(self, result: StageResult):
        self._result = result

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        return self._result


def test_orchestrator_runs_stages(tmp_job_dir):
    stage1 = MockStage(StageResult.success(metadata={"step": 1}))
    stage2 = MockStage(StageResult.success(metadata={"step": 2}))

    orchestrator = PipelineOrchestrator(stages=[stage1, stage2])
    results = orchestrator.run(tmp_job_dir, config=StageConfig())

    assert len(results) == 2
    assert results[0].status == "success"
    assert results[1].status == "success"


def test_orchestrator_stops_on_error(tmp_job_dir):
    stage1 = MockStage(StageResult.success())
    stage2 = MockStage(StageResult.from_error("boom"))
    stage3 = MockStage(StageResult.success())

    orchestrator = PipelineOrchestrator(stages=[stage1, stage2, stage3])
    results = orchestrator.run(tmp_job_dir, config=StageConfig())

    assert len(results) == 2
    assert results[1].status == "error"


def test_orchestrator_collects_progress(tmp_job_dir):
    stage1 = MockStage(StageResult.success(metadata={"frames": 100}))

    orchestrator = PipelineOrchestrator(stages=[stage1])
    progress_events = []
    orchestrator.on_progress(lambda event: progress_events.append(event))

    orchestrator.run(tmp_job_dir, config=StageConfig())

    assert len(progress_events) == 2
    assert progress_events[0]["status"] == "running"
    assert progress_events[1]["status"] == "complete"
