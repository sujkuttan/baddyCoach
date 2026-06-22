import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import PlayerTrackingStage


def test_player_tracking_assigns_near_far(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
    }
    store.set("court", court_data)

    # Mock detection results: two players, one near (y > 300), one far (y < 300)
    detections = [
        {"frame": 0, "bbox": [100, 350, 200, 500], "confidence": 0.9},
        {"frame": 0, "bbox": [800, 100, 900, 250], "confidence": 0.9},
    ]

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2
    sides = [p["side"] for p in players["players"]]
    assert "near" in sides
    assert "far" in sides
