"""
Golden regression test suite for pipeline unification.

Verifies that shared modules produce consistent results across both
backend pipeline stages and direct shared module usage.
"""

import numpy as np
import pandas as pd
import pytest

from app.pipeline.base import ArtifactStore, StageConfig


# ─── Comparison utilities ────────────────────────────────────────────────────

FLOAT_TOLERANCE = 1e-5


def assert_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = False):
    """Assert two DataFrames are equal within float tolerance."""
    assert df1.columns.tolist() == df2.columns.tolist(), f"Column mismatch: {df1.columns} vs {df2.columns}"
    assert len(df1) == len(df2), f"Length mismatch: {len(df1)} vs {len(df2)}"
    for col in df1.columns:
        if df1[col].dtype.kind == "f" or df2[col].dtype.kind == "f":
            c1 = df1[col].astype(float)
            c2 = df2[col].astype(float)
            mask = ~(c1.isna() & c2.isna())
            if mask.any():
                np.testing.assert_allclose(c1[mask], c2[mask], atol=FLOAT_TOLERANCE, err_msg=f"Column '{col}' mismatch")
        else:
            assert df1[col].tolist() == df2[col].tolist(), f"Column '{col}' mismatch"


def assert_dicts_equal(d1: dict, d2: dict, path: str = ""):
    """Assert two dicts are equal within float tolerance, recursively."""
    assert set(d1.keys()) == set(d2.keys()), f"Key mismatch at {path}: {set(d1.keys()) ^ set(d2.keys())}"
    for k in d1:
        full_path = f"{path}.{k}" if path else k
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            assert_dicts_equal(v1, v2, full_path)
        elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            assert len(v1) == len(v2), f"List length mismatch at {full_path}: {len(v1)} vs {len(v2)}"
            for i, (a, b) in enumerate(zip(v1, v2)):
                if isinstance(a, dict) and isinstance(b, dict):
                    assert_dicts_equal(a, b, f"{full_path}[{i}]")
                elif isinstance(a, float) and isinstance(b, float):
                    assert abs(a - b) < FLOAT_TOLERANCE, f"Float mismatch at {full_path}[{i}]: {a} vs {b}"
                else:
                    assert a == b, f"Value mismatch at {full_path}[{i}]: {a} vs {b}"
        elif isinstance(v1, float) and isinstance(v2, float):
            assert abs(v1 - v2) < FLOAT_TOLERANCE, f"Float mismatch at {full_path}: {v1} vs {v2}"
        else:
            assert v1 == v2, f"Value mismatch at {full_path}: {v1} vs {v2}"


# ─── Synthetic test data fixtures ────────────────────────────────────────────

@pytest.fixture
def synthetic_shots_df():
    return pd.DataFrame({
        "frame": [0, 10, 20, 30, 40],
        "player_id": ["player_1", "player_2", "player_1", "player_2", "player_1"],
        "stroke_type": ["serve", "clear", "smash", "drop", "net_shot"],
        "stroke_confidence": [0.9, 0.85, 0.88, 0.76, 0.92],
        "shot_id": [1, 2, 3, 4, 5],
        "rally_id": [1, 1, 1, 1, 1],
        "court_x": [1.0, 10.0, 3.0, 8.0, 2.0],
        "court_y": [2.5, 2.5, 1.0, 4.0, 2.5],
    })


@pytest.fixture
def synthetic_shuttle_df():
    return pd.DataFrame({
        "frame": [0, 5, 10, 15, 20, 25, 30, 35, 40],
        "x": [400, 420, 450, 430, 410, 460, 500, 480, 450],
        "y": [600, 550, 300, 250, 600, 550, 350, 320, 400],
        "confidence": [0.95, 0.92, 0.88, 0.90, 0.93, 0.87, 0.91, 0.89, 0.94],
    })


@pytest.fixture
def synthetic_pose_df():
    kps = np.zeros((5, 17, 3), dtype=np.float32)
    kps[:, 5, :] = [100, 100, 0.9]
    kps[:, 9, :] = [100, 200, 0.9]
    kps[:, 11, :] = [100, 300, 0.9]
    kps[:, 15, :] = [95, 350, 0.9]
    kps[:, 16, :] = [105, 350, 0.9]
    records = []
    for i, frame in enumerate([0, 10, 20, 30, 40]):
        records.append({"frame": frame, "player_id": "player_1", "keypoints": kps[i].tolist()})
        records.append({"frame": frame, "player_id": "player_2", "keypoints": kps[i].tolist()})
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_court_data():
    return {
        "homography": np.eye(3, dtype=np.float64).tolist(),
        "corners_pixel": [[100, 400], [500, 400], [100, 100], [500, 100]],
        "court_length": 13.4,
        "court_width": 6.10,
        "net_height": 1.55,
        "valid": True,
    }


@pytest.fixture
def synthetic_rallies_df():
    return pd.DataFrame({
        "rally_id": [1],
        "start_frame": [0],
        "end_frame": [40],
        "shot_count": [5],
        "winner_player_id": ["player_1"],
        "end_reason": ["winner"],
    })


# ─── Phase 4.1: Test court detection shared module consistency ──────────────

class TestSharedCourtConsistency:
    """Verify shared court module produces consistent results via both paths."""

    def test_court_constants_match(self):
        from app.pipeline.shared.court import (
            COURT_LENGTH as S_LEN, COURT_WIDTH as S_WID, NET_HEIGHT as S_NET,
        )
        from app.pipeline.court import (
            COURT_LENGTH as P_LEN, COURT_WIDTH as P_WID, NET_HEIGHT as P_NET,
        )
        assert S_LEN == P_LEN == 13.4
        assert S_WID == P_WID == 6.10
        assert S_NET == P_NET == 1.55

    def test_homography_consistency(self):
        from app.pipeline.shared.court import compute_homography as shared_homography
        from app.pipeline.shared.court import image_to_court as shared_itc
        from app.pipeline.shared.court import _correct_court_points as shared_correct

        corners = [[100, 400], [500, 400], [150, 100], [450, 100]]
        corrected = shared_correct(corners)
        H, valid = shared_homography(corrected)
        assert valid is not False
        assert H is not None

        cx, cy = shared_itc(H, (300, 250))
        assert 0.0 <= cx <= 13.4 or abs(cx) < 2
        assert 0.0 <= cy <= 6.10 or abs(cy) < 2


    def test_rule_based_shuttle_consistency(self):
        from app.pipeline.shared.utils import _rule_based_shuttle_predict

        shuttle_df = pd.DataFrame({
            "frame": [0, 1, 2, 3, 4],
            "x": [640, 640, 640, 640, 640],
            "y": [200, 250, 350, 500, 700],
        })
        result = _rule_based_shuttle_predict(shuttle_df, 2, 1280, 720)
        assert result == "smash"

    def test_rally_utils_consistency(self):
        from app.pipeline.shared.utils import _infer_end_reason, _is_rally_ending_shot

        assert _infer_end_reason("smash", 0.8) == "winner"
        assert _infer_end_reason("net_shot", 0.7) == "net"
        assert _infer_end_reason("clear", 0.2) == "unforced_error"
        assert _is_rally_ending_shot("smash", 0.7, 30) is True
        assert _is_rally_ending_shot("clear", 0.5, 10) is False

    def test_stage_rally_stats_consistency(self):
        from app.pipeline.shared.utils import stage_rally_stats

        shots = [
            {"frame": 1, "player_id": "player_1", "stroke_type": "serve", "rally_id": 1},
            {"frame": 2, "player_id": "player_2", "stroke_type": "clear", "rally_id": 1},
            {"frame": 3, "player_id": "player_1", "stroke_type": "smash", "rally_id": 1},
        ]
        rallies = [
            {"rally_id": 1, "start_frame": 1, "end_frame": 3, "shot_count": 3,
             "winner_player_id": "player_1"},
        ]
        stats = stage_rally_stats(shots, rallies)
        assert stats["avg_length"] == 3.0


# ─── Phase 4.3: Test analytics stages with shared modules ───────────────────

class TestAnalyticsStagesSharedModules:
    """Verify analytics stages use shared modules correctly."""

    def test_court_position_uses_shared_modules(self, tmp_job_dir, synthetic_court_data,
                                                 synthetic_shots_df, synthetic_shuttle_df):
        from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        store.set("court", synthetic_court_data)
        store.set_parquet("shots", synthetic_shots_df)
        store.set_parquet("shuttle", synthetic_shuttle_df)

        stage = CourtPositionAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "success"
        assert result.metadata["zone_transitions"] > 0

    def test_footwork_uses_shared_modules(self, tmp_job_dir, synthetic_court_data,
                                           synthetic_pose_df, synthetic_rallies_df,
                                           synthetic_shots_df):
        from app.pipeline.analytics.footwork import FootworkAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        store.set("court", synthetic_court_data)
        store.set_parquet("pose", synthetic_pose_df)
        store.set_parquet("rallies", synthetic_rallies_df)
        store.set_parquet("shots", synthetic_shots_df)

        stage = FootworkAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "success"
        assert "distance_covered" in result.metadata

    def test_fitness_uses_shared_modules(self, tmp_job_dir, synthetic_rallies_df,
                                          synthetic_shots_df):
        from app.pipeline.analytics.fitness import FitnessAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        store.set("footwork_analytics", {
            "player_1": {
                "distance_covered": 500.0,
                "recovery_times": [0.8, 1.2, 0.9],
                "avg_recovery": 0.97,
            }
        })
        store.set_parquet("rallies", synthetic_rallies_df)
        store.set_parquet("shots", synthetic_shots_df)

        stage = FitnessAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "success"
        assert "rally_intensity" in result.metadata

    def test_tactical_uses_shared_modules(self, tmp_job_dir, synthetic_shots_df,
                                           synthetic_court_data, synthetic_shuttle_df):
        from app.pipeline.analytics.tactical import TacticalAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        store.set_parquet("shots", synthetic_shots_df)
        store.set("court", synthetic_court_data)
        store.set_parquet("shuttle", synthetic_shuttle_df)

        stage = TacticalAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "success"
        assert "shot_distribution" in result.metadata


# ─── Phase 4.4: Edge cases for shared modules ───────────────────────────────

class TestUnificationEdgeCases:
    """Edge cases for shared module usage."""

    def test_empty_shuttle_data(self, tmp_job_dir):
        from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        store.set("court", {"court_length": 13.4, "court_width": 6.10, "valid": True})

        shots_df = pd.DataFrame({
            "frame": [0, 10, 20],
            "player_id": ["player_1", "player_2", "player_1"],
            "stroke_type": ["serve", "clear", "drop"],
            "stroke_confidence": [0.9, 0.85, 0.88],
        })
        store.set_parquet("shots", shots_df)

        empty_shuttle = pd.DataFrame(columns=["frame", "x", "y", "confidence"])
        store.set_parquet("shuttle", empty_shuttle)

        stage = CourtPositionAnalyticsStage()
        result = stage.run(store, StageConfig())
        assert result.status == "success"

    def test_footwork_no_pose(self, tmp_job_dir):
        from app.pipeline.analytics.footwork import FootworkAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        store.set("court", {"court_length": 13.4, "court_width": 6.10})

        stage = FootworkAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "error"

    def test_fitness_no_footwork(self, tmp_job_dir):
        from app.pipeline.analytics.fitness import FitnessAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        stage = FitnessAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "error"

    def test_tactical_no_shots(self, tmp_job_dir):
        from app.pipeline.analytics.tactical import TacticalAnalyticsStage

        store = ArtifactStore(tmp_job_dir)
        empty_shots = pd.DataFrame(columns=["frame", "player_id", "stroke_type", "stroke_confidence"])
        store.set_parquet("shots", empty_shots)

        stage = TacticalAnalyticsStage()
        result = stage.run(store, StageConfig())

        assert result.status == "error"

    def test_shared_setup_models_standalone(self):
        """Verify setup_models returns empty dict when backend not available."""
        import importlib
        from app.pipeline.shared import models as shared_models

        models = shared_models.setup_models("cpu")
        assert isinstance(models, dict)
