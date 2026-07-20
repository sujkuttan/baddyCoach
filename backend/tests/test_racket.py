import numpy as np


def test_racket_tracker_head_point_extraction():
    from app.models.racket import RacketTracker
    tr = RacketTracker.__new__(RacketTracker)
    # bbox (x1,y1,x2,y2) = (100,200,140,300); head = top-center + margin
    head = tr._head_point((100, 200, 140, 300), margin=0.1)
    # top-center x = (100+140)/2 = 120 ; y nudged up by margin*height = 0.1*100 =10 => 200-10=190
    assert abs(head[0] - 120.0) < 1e-6
    assert abs(head[1] - 190.0) < 1e-6


def test_get_racket_returns_none_when_disabled(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "racket_enabled", False)
    from app.pipeline.shared.models import get_racket
    assert get_racket() is None


def test_racket_detection_stage_stores_artifact(monkeypatch, tmp_path):
    """Backend RacketDetectionStage must populate `racket_detections` so the
    downstream hit/stroke/ownership stages can consume it."""
    import numpy as np
    from app.storage.artifacts import ArtifactStore
    from app.pipeline.base import StageConfig
    from app.pipeline.racket import RacketDetectionStage

    class _StubTracker:
        def detect(self, frames, player_bboxes):
            return [
                {"frame": 0, "player_side": "near", "bbox": (1, 2, 3, 4),
                 "conf": 0.9, "head_point": (2.0, 1.0)},
            ]

    from app.pipeline import shared
    monkeypatch.setattr(shared.models, "get_racket", lambda: _StubTracker())

    store = ArtifactStore(tmp_path)
    store.set("players", {"players": [
        {"id": "p1", "side": "near", "detections": [{"frame": 0, "bbox": [1, 2, 3, 4]}]},
    ]})
    config = StageConfig()

    result = RacketDetectionStage().run(store, config, frames=[np.zeros((10, 10, 3), dtype=np.uint8)])
    assert result.status == "success"
    dets = store.get("racket_detections")
    assert isinstance(dets, list) and len(dets) == 1
    assert dets[0]["player_side"] == "near"
    assert result.metadata["n_racket_detections"] == 1


def test_racket_detection_stage_skips_when_disabled(monkeypatch, tmp_path):
    import numpy as np
    from app.storage.artifacts import ArtifactStore
    from app.pipeline.base import StageConfig
    from app.pipeline.racket import RacketDetectionStage

    from app.pipeline import shared
    monkeypatch.setattr(shared.models, "get_racket", lambda: None)

    store = ArtifactStore(tmp_path)
    config = StageConfig()
    result = RacketDetectionStage().run(
        store, config, frames=[np.zeros((10, 10, 3), dtype=np.uint8)]
    )
    assert result.status == "skipped"
    assert store.get("racket_detections") == []

