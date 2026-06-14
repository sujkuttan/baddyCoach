import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage


def test_court_zones_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"court_length": 13.4, "court_width": 5.18}
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


from app.pipeline.analytics.footwork import FootworkAnalyticsStage


def test_footwork_metrics_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"court_length": 13.4, "court_width": 5.18}
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
