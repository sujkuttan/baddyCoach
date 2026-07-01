import pytest
import numpy as np
import pandas as pd

from app.pipeline.quality import compute_quality
from app.config.settings import settings


class FakeStore:
    def __init__(self):
        self._data = {}
    def get(self, key, default=None):
        return self._data.get(key, default)
    def get_parquet(self, key):
        df = self._data.get(f"{key}_df")
        return df


def test_quality_high_tier():
    n = settings.quality_min_shots_tactical + 5
    store = FakeStore()
    store._data["court"] = {"valid": True}
    store._data["video_metadata"] = {"total_frames": 200, "source_fps": 30, "fps": 10}
    store._data["shots_df"] = pd.DataFrame({
        "frame": list(range(n)),
        "stroke_confidence": [0.9]*n,
        "court_x": [1.0]*n, "court_y": [1.0]*n,
        "is_bst_fallback": [False]*n,
    })
    store._data["pose_df"] = pd.DataFrame({"frame": list(range(150))})
    store._data["shuttle_df"] = pd.DataFrame({"confidence": [0.9]*n})
    store._data["rallies_df"] = pd.DataFrame({"rally_id": [1, 2, 3]})

    q = compute_quality(store)
    assert q["tier"] == "high"
    assert q["quality_score"] > 0.7
    assert q["capability_trust"]["tactical"] is True


def test_quality_low_tier_invalid_court():
    store = FakeStore()
    store._data["court"] = {"valid": False}
    store._data["video_metadata"] = {"total_frames": 0, "source_fps": 0, "fps": 0}
    store._data["shots_df"] = pd.DataFrame({
        "frame": [], "stroke_confidence": [],
    })
    store._data["pose_df"] = pd.DataFrame({"frame": []})
    store._data["shuttle_df"] = pd.DataFrame({"confidence": []})
    store._data["rallies_df"] = pd.DataFrame({"rally_id": []})

    q = compute_quality(store)
    assert q["tier"] == "low"
    assert q["quality_score"] < 0.5
    assert q["capability_trust"]["tactical"] is False


def test_quality_capability_trust_all_true():
    n = settings.quality_min_shots_tactical + 5
    store = FakeStore()
    store._data["court"] = {"valid": True}
    store._data["video_metadata"] = {"total_frames": 200, "source_fps": 30, "fps": 10}
    store._data["shots_df"] = pd.DataFrame({
        "frame": list(range(n)),
        "stroke_confidence": [0.8]*n,
        "court_x": [1.0]*n, "court_y": [1.0]*n,
        "is_bst_fallback": [False]*n,
    })
    store._data["pose_df"] = pd.DataFrame({"frame": list(range(150))})
    store._data["shuttle_df"] = pd.DataFrame({"confidence": [0.9]*n})
    store._data["rallies_df"] = pd.DataFrame({"rally_id": [1, 2, 3]})

    q = compute_quality(store)
    for cap, val in q["capability_trust"].items():
        assert val is True, f"{cap} should be trusted"
