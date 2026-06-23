import numpy as np
import pandas as pd
from pathlib import Path

from app.pipeline.base import StageConfig
from app.pipeline import CourtDetectionStage
from app.pipeline import PlayerTrackingStage
from app.pipeline import ShuttleTrackingStage
from app.pipeline import PoseEstimationStage
from app.pipeline import HitFrameLocalizationStage
from app.pipeline import StrokeClassificationStage
from app.pipeline import PlayerAttributionStage
from app.pipeline import RallySegmentationStage
from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage
from app.pipeline.analytics.footwork import FootworkAnalyticsStage
from app.pipeline.analytics.fitness import FitnessAnalyticsStage
from app.pipeline.analytics.tactical import TacticalAnalyticsStage
from app.pipeline.analytics.technical import TechnicalAnalyticsStage
from app.shuttle_coach.engine import analyze_from_pipeline
from app.storage.artifacts import ArtifactStore


def test_full_pipeline_mock(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    corners = [(100, 500), (1820, 500), (100, 100), (1820, 100)]
    result = CourtDetectionStage().run(store, config, corners=corners)
    assert result.status == "success"

    detections = [
        {"frame": 0, "bbox": [100, 350, 200, 500], "confidence": 0.9},
        {"frame": 0, "bbox": [800, 100, 900, 250], "confidence": 0.9},
    ]
    result = PlayerTrackingStage().run(store, config, detections=detections)
    assert result.status == "success"

    # Shuttle trajectory with three clear hit-like direction changes
    shuttle_data = []
    for i in range(50):
        if i < 10:
            x, y = 400, 100 + i * 6
        elif i < 20:
            x, y = 400 - (i - 10) * 14, 160 - (i - 10) * 10
        elif i < 30:
            x, y = 260 + (i - 20) * 8, 60 + (i - 20) * 10
        else:
            x, y = 340 + (i - 30) * 12, 160 + (i - 30) * 6
        shuttle_data.append({"frame": i, "x": x, "y": y, "confidence": 0.9})
    result = ShuttleTrackingStage().run(store, config, shuttle_data=shuttle_data)
    assert result.status == "success"

    pose_data = []
    for frame in range(50):
        for pid in ["player_1", "player_2"]:
            pose_data.append({"frame": frame, "player_id": pid, "keypoints": np.random.rand(17, 3).tolist()})
    result = PoseEstimationStage().run(store, config, pose_data=pose_data)
    assert result.status == "success"

    result = HitFrameLocalizationStage().run(store, config)
    assert result.status == "success"

    result = StrokeClassificationStage().run(store, config)
    assert result.status == "success"

    result = PlayerAttributionStage().run(store, config)
    assert result.status == "success"

    result = RallySegmentationStage().run(store, config)
    assert result.status == "success"

    result = CourtPositionAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = FootworkAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = FitnessAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = TacticalAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = TechnicalAnalyticsStage().run(store, config)
    assert result.status == "success"

    analytics = {
        "fitness_analytics": store.get("fitness_analytics") or {},
        "tactical_analytics": store.get("tactical_analytics") or {},
        "footwork_analytics": store.get("footwork_analytics") or {},
    }

    report = analyze_from_pipeline(analytics, shuttle_metrics={}, player_id="player_1")
    assert "strengths" in report
    assert "evidence" in report