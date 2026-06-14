from pathlib import Path
from app.pipeline.base import ArtifactStore, StageResult


def test_artifact_store_set_get(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    store.set("court", {"homography": [[1, 0], [0, 1]]})
    data = store.get("court")
    assert data == {"homography": [[1, 0], [0, 1]]}


def test_artifact_store_persists_to_disk(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    store.set("court", {"homography": [[1, 0], [0, 1]]})
    assert (tmp_job_dir / "court.json").exists()

    store2 = ArtifactStore(tmp_job_dir)
    data = store2.get("court")
    assert data == {"homography": [[1, 0], [0, 1]]}


def test_artifact_store_parquet(tmp_job_dir):
    import pandas as pd
    store = ArtifactStore(tmp_job_dir)
    df = pd.DataFrame({"frame": [1, 2, 3], "x": [10.0, 20.0, 30.0]})
    store.set_parquet("shuttle", df)
    assert (tmp_job_dir / "shuttle.parquet").exists()

    df2 = store.get_parquet("shuttle")
    assert list(df2.columns) == ["frame", "x"]
    assert len(df2) == 3


def test_stage_result_success():
    result = StageResult.success(metadata={"frames": 100})
    assert result.status == "success"
    assert result.error is None
    assert result.metadata == {"frames": 100}


def test_stage_result_error():
    result = StageResult.from_error("model not found")
    assert result.status == "error"
    assert result.error == "model not found"
