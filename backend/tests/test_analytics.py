import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage


def test_court_zones_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"court_length": 13.4, "court_width": 6.10, "valid": True}
    store.set("court", court_data)

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_2", "player_1"],
        "stroke_type": ["serve", "clear", "drop"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "x": [2.5, 1.0, 4.0],
        "y": [3.0, 10.0, 7.0],
        "confidence": [0.95, 0.92, 0.88],
    })
    store.set_parquet("shuttle", shuttle_df)

    stage = CourtPositionAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "zone_transitions" in result.metadata


def test_court_position_degrades_with_invalid_court_geometry(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    store.set("court", {"court_length": 13.4, "court_width": 6.10, "valid": False})
    store.set_parquet("shots", pd.DataFrame({
        "frame": [0, 10],
        "player_id": ["player_1", "player_2"],
        "court_x": [2.5, 8.0],
        "court_y": [3.0, 4.0],
    }))

    result = CourtPositionAnalyticsStage().run(store, config)

    assert result.status == "success"
    assert "zone_transitions" in result.metadata


from app.pipeline.analytics.footwork import FootworkAnalyticsStage


def test_footwork_metrics_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"court_length": 13.4, "court_width": 6.10}
    store.set("court", court_data)

    pose_df = pd.DataFrame({
        "frame": list(range(30)),
        "player_id": ["player_1"] * 30,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(30)],
    })
    store.set_parquet("pose", pose_df)

    rallies_df = pd.DataFrame({
        "rally_id": [1],
        "start_frame": [0],
        "end_frame": [29],
        "shot_count": [5],
    })
    store.set_parquet("rallies", rallies_df)

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_1", "player_1"],
        "stroke_type": ["serve", "clear", "drop"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    stage = FootworkAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "distance_covered" in result.metadata
    assert "recovery_times" in result.metadata


from app.pipeline.analytics.fitness import FitnessAnalyticsStage


def test_fitness_metrics_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    footwork_data = {
        "player_1": {
            "distance_covered": 500.0,
            "recovery_times": [0.8, 1.2, 0.9, 1.5, 1.1],
            "avg_recovery": 1.1,
        }
    }
    store.set("footwork_analytics", footwork_data)

    rallies_df = pd.DataFrame({
        "rally_id": [1, 2, 3],
        "start_frame": [0, 50, 100],
        "end_frame": [45, 95, 145],
        "shot_count": [5, 6, 4],
    })
    store.set_parquet("rallies", rallies_df)

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20, 55, 65, 105, 115],
        "player_id": ["player_1"] * 7,
        "stroke_type": ["serve", "clear", "drop", "smash", "clear", "net_shot", "drop"],
        "stroke_confidence": [0.9] * 7,
    })
    store.set_parquet("shots", shots_df)

    stage = FitnessAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "rally_intensity" in result.metadata
    assert "fatigue_trend" in result.metadata


from app.pipeline.analytics.tactical import TacticalAnalyticsStage


def test_tactical_analytics_shot_distribution(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": list(range(20)),
        "player_id": ["player_1"] * 20,
        "stroke_type": ["clear"] * 8 + ["smash"] * 5 + ["drop"] * 4 + ["net_shot"] * 3,
        "stroke_confidence": [0.9] * 20,
    })
    store.set_parquet("shots", shots_df)

    court_data = {"court_length": 13.4, "court_width": 6.10}
    store.set("court", court_data)

    shuttle_df = pd.DataFrame({
        "frame": list(range(20)),
        "x": np.random.uniform(0, 6.10, 20),
        "y": np.random.uniform(0, 13.4, 20),
        "confidence": [0.95] * 20,
    })
    store.set_parquet("shuttle", shuttle_df)

    stage = TacticalAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "shot_distribution" in result.metadata


from app.pipeline.analytics.technical import TechnicalAnalyticsStage


def test_technical_analytics_evaluates_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_1", "player_1"],
        "stroke_type": ["smash", "clear", "net_shot"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    pose_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_1", "player_1"],
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(3)],
    })
    store.set_parquet("pose", pose_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "x": [2.5, 1.0, 4.0],
        "y": [3.0, 10.0, 7.0],
        "confidence": [0.95, 0.92, 0.88],
    })
    store.set_parquet("shuttle", shuttle_df)

    court_data = {"court_length": 13.4, "court_width": 6.10}
    store.set("court", court_data)

    bst_clips = {
        "clip_0": {"frames": [0, 1, 2, 3, 4]},
        "clip_1": {"frames": [8, 9, 10, 11, 12]},
        "clip_2": {"frames": [18, 19, 20, 21, 22]},
    }
    store.set("bst_clips", bst_clips)

    stage = TechnicalAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "technical_assessment" in result.metadata
