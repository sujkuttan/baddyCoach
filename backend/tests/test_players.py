import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import PlayerTrackingStage


def _make_detection(frame, bbox, track_id=None, confidence=0.9):
    return {"frame": frame, "bbox": list(bbox), "confidence": confidence, "track_id": track_id}


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
        _make_detection(0, (100, 350, 200, 500)),
        _make_detection(0, (800, 100, 900, 250)),
    ]

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2
    sides = [p["side"] for p in players["players"]]
    assert "near" in sides
    assert "far" in sides


def test_track_stitching_merges_fragments(tmp_job_dir):
    """8 track-ID fragments for one player should be stitched into 2 players."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
    }
    store.set("court", court_data)
    court_mid_y = 300  # from corner average

    np.random.seed(42)
    near_cy = np.random.uniform(350, 500, 100)
    far_cy = np.random.uniform(100, 250, 100)

    detections = []
    # Near player: cycles through 8 track_ids over 100 frames
    for f in range(100):
        tid = f // 12  # new track_id every 12 frames
        bbox = (150, near_cy[f] - 50, 250, near_cy[f] + 50)
        detections.append(_make_detection(f, bbox, track_id=tid))
    # Far player: consistent track_id
    for f in range(100):
        bbox = (850, far_cy[f] - 50, 950, far_cy[f] + 50)
        detections.append(_make_detection(f, bbox, track_id=999))

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2

    # Each player should have ~100 detections (not 8 fragments + 1)
    for p in players["players"]:
        assert p["detection_count"] >= 95, f"Player {p['id']} has only {p['detection_count']} detections"
        frames = [d["frame"] for d in p["detections"]]
        assert frames == sorted(frames), "Detections must be sorted by frame"

    # Side counts should match
    near_players = [p for p in players["players"] if p["side"] == "near"]
    far_players = [p for p in players["players"] if p["side"] == "far"]
    assert len(near_players) == 1
    assert len(far_players) == 1
    assert near_players[0]["detection_count"] >= 95
    assert far_players[0]["detection_count"] >= 95


def test_track_stitching_handles_midline_crossing(tmp_job_dir):
    """When a near player briefly crosses midline, they stay as near."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
    }
    store.set("court", court_data)

    detections = []
    # Near player: 90 frames above midline, 10 frames below (brief cross)
    for f in range(100):
        cy = 320 if f < 45 or f >= 55 else 280  # dips below 300 midline
        bbox = (150, cy - 50, 250, cy + 50)
        detections.append(_make_detection(f, bbox, track_id=1))
    # Far player: always far
    for f in range(100):
        bbox = (850, 150, 950, 250)
        detections.append(_make_detection(f, bbox, track_id=2))

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2

    near_p = [p for p in players["players"] if p["side"] == "near"][0]
    # The near player should have all 100 detections despite the brief cross
    assert near_p["detection_count"] == 100, (
        f"Near player lost detections: got {near_p['detection_count']}, expected 100"
    )


def test_track_stitching_when_disabled(tmp_job_dir):
    """With track_stitch_enabled=False, track_id grouping is used."""
    from app.config.settings import settings

    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
    }
    store.set("court", court_data)

    detections = []
    # 8 track_id fragments for one player
    for f in range(100):
        tid = f // 12
        bbox = (150, 350, 250, 500)
        detections.append(_make_detection(f, bbox, track_id=tid))
    for f in range(100):
        bbox = (850, 100, 950, 250)
        detections.append(_make_detection(f, bbox, track_id=999))

    old = settings.track_stitch_enabled
    settings.track_stitch_enabled = False
    try:
        stage = PlayerTrackingStage()
        result = stage.run(store, config, detections=detections)

        assert result.status == "success"
        players = store.get("players")
        assert len(players["players"]) == 2  # capped by max_players

        # Track stitching is off, so the fragmented player will only have
        # ~12 detections instead of ~100
        for p in players["players"]:
            assert p["detection_count"] < 30, (
                f"Expected fragmented (<30), got {p['detection_count']}"
            )
    finally:
        settings.track_stitch_enabled = old


def test_no_detections_returns_error(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"valid": True, "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)]}
    store.set("court", court_data)

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=[])

    assert result.status == "error"


def test_no_court_returns_error(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=[_make_detection(0, (100, 350, 200, 500))])

    assert result.status == "error"
