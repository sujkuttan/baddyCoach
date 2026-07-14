import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import HitFrameLocalizationStage


def test_hit_detection_finds_trajectory_changes(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    n = 40
    frames = list(range(n))
    x = [100.0 + t * 5.0 for t in range(20)] + [200.0 - (t - 20) * 8.0 for t in range(20, n)]
    y = [200.0 - t * 2.0 for t in range(20)] + [160.0 + (t - 20) * 4.0 for t in range(20, n)]
    shuttle_df = pd.DataFrame({
        "frame": frames,
        "x": x,
        "y": y,
        "confidence": [0.95] * n,
    })
    store.set_parquet("shuttle_raw", shuttle_df)

    pose_df = pd.DataFrame({
        "frame": list(range(20)),
        "player_id": ["player_1"] * 20,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(20)],
    })
    store.set_parquet("pose", pose_df)

    stage = HitFrameLocalizationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert result.metadata["hit_count"] > 0
    # Verify at least one hit near the trajectory reversal (frame 20 ± window)
    hits_df = store.get_parquet("hits")
    assert hits_df is not None and len(hits_df) > 0


def test_hit_stage_prefers_shuttle_raw_over_cleaned(tmp_job_dir, monkeypatch):
    """Cleaned trajectory is flat (no reversal); raw has a clear reversal at frame 20."""
    monkeypatch.setattr("app.pipeline.hits.settings.audio_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.wrist_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_refine_window", 0)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_frame_calibration_offset", 0)

    store = ArtifactStore(tmp_job_dir)
    n = 40
    frames = list(range(n))
    # Raw: V-shaped reversal at 20
    x_raw = [100.0 + t * 5.0 for t in range(20)] + [200.0 - (t - 20) * 8.0 for t in range(20, n)]
    y_raw = [200.0 - t * 2.0 for t in range(20)] + [160.0 + (t - 20) * 4.0 for t in range(20, n)]
    store.set_parquet("shuttle_raw", pd.DataFrame({
        "frame": frames, "x": x_raw, "y": y_raw, "confidence": [0.95] * n,
    }))
    # Cleaned: nearly constant — would hide the hit if preferred
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": frames, "x": [150.0] * n, "y": [180.0] * n, "confidence": [0.95] * n,
    }))
    store.set_parquet("pose", pd.DataFrame({
        "frame": [], "player_id": [], "keypoints": [],
    }))

    result = HitFrameLocalizationStage().run(store, StageConfig())
    assert result.status == "success"
    hits = store.get_parquet("hits")
    assert hits is not None and len(hits) > 0
    assert any(abs(int(f) - 20) <= 5 for f in hits["frame"].tolist())
