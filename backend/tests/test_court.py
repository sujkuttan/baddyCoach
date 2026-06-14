import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.court import CourtDetectionStage


def test_court_detection_with_known_corners(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = CourtDetectionStage()
    result = stage.run(store, config, corners=[
        (100, 500),   # top-left
        (1820, 500),  # top-right
        (100, 100),   # bottom-left
        (1820, 100),  # bottom-right
    ])

    assert result.status == "success"
    assert "court" in result.artifacts
    court_data = store.get("court")
    assert "homography" in court_data
    assert len(court_data["homography"]) == 3
    assert len(court_data["homography"][0]) == 3


def test_court_detection_requires_corners(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = CourtDetectionStage()
    result = stage.run(store, config)

    assert result.status == "error"
    assert "corners" in result.error.lower()
