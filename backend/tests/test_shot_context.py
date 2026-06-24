import pytest
import pandas as pd

from app.pipeline.analytics.shot_context import ShotContextStage


class FakeStore:
    def __init__(self):
        self._parquet = {}
        self._parquet_data = {}
        self._data = {}
    def get_parquet(self, key):
        return self._parquet.get(key)
    def set_parquet(self, key, df):
        self._parquet[key] = df
        self._parquet_data[key] = df
    def get(self, key, default=None):
        return self._data.get(key, default)
    def set(self, key, value):
        self._data[key] = value
    def path(self, key):
        return None


@pytest.fixture
def sample_rallies():
    return pd.DataFrame({
        "rally_id": [1, 2],
        "start_frame": [0, 10],
        "end_frame": [9, 25],
        "winner_player_id": ["player_1", "player_2"],
        "end_reason": ["winner", "unforced"],
        "shot_count": [4, 5],
    })


@pytest.fixture
def sample_shots():
    return pd.DataFrame({
        "frame": [2, 4, 6, 8, 12, 14, 16, 18, 20],
        "rally_id": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                       "player_1", "player_2", "player_1", "player_2", "player_1"],
        "stroke_type": ["clear", "smash", "drop", "clear",
                        "smash", "drop", "clear", "net_shot", "smash"],
        "stroke_confidence": [0.9]*9,
        "court_x": [1.0]*9, "court_y": [2.0]*9,
    })


def test_shot_context_stage_runs(sample_rallies, sample_shots):
    store = FakeStore()
    store._parquet["rallies"] = sample_rallies
    store._parquet["shots"] = sample_shots
    store._data["court"] = {"valid": True, "court_length": 13.4, "court_width": 6.1}
    from app.pipeline.base import StageConfig
    config = StageConfig(gpu_enabled=False, processing_fps=30)
    stage = ShotContextStage()
    result = stage.run(store, config)
    assert result.status == "success"
    events = store.get_parquet("shot_events")
    assert events is not None
    assert "under_pressure" in events.columns
    assert "shot_outcome" in events.columns
    assert "rally_id" in events.columns


def test_shot_context_missing_data():
    store = FakeStore()
    from app.pipeline.base import StageConfig
    config = StageConfig(gpu_enabled=False, processing_fps=30)
    stage = ShotContextStage()
    result = stage.run(store, config)
    assert result.status == "error"


def test_shot_context_under_pressure(sample_rallies, sample_shots):
    store = FakeStore()
    store._parquet["rallies"] = sample_rallies
    store._parquet["shots"] = sample_shots
    store._data["court"] = {"valid": True, "court_length": 13.4, "court_width": 6.1}
    from app.pipeline.base import StageConfig
    config = StageConfig(gpu_enabled=False, processing_fps=30)
    stage = ShotContextStage()
    stage.run(store, config)
    events = store.get_parquet("shot_events")
    assert events is not None
    assert events["under_pressure"].dtype == bool
