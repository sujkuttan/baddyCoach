import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import PlayerTrackingStage
from app.pipeline.players import stitch_tracks



def _make_detection(frame, bbox, track_id=None, confidence=0.9):
    return {"frame": frame, "bbox": list(bbox), "confidence": confidence, "track_id": track_id}


class _FakeDetection:
    def __init__(self, bbox, confidence, track_id):
        self.bbox = bbox
        self.confidence = confidence
        self.track_id = track_id


class _FakeTracker:
    def track_frames(self, frames):
        return {
            "frames": {
                0: [_FakeDetection([1, 2, 11, 22], 0.9, 7)],
                1: [_FakeDetection([2, 3, 12, 23], 0.8, 7)],
            },
        }


def test_live_yolo_logging_uses_pipeline_logger_contract(monkeypatch):
    monkeypatch.setattr(
        "app.pipeline.shared.models.get_yolov8", lambda: _FakeTracker()
    )

    detections = PlayerTrackingStage()._run_yolov8([
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ])

    assert detections == [
        {"frame": 0, "bbox": [1, 2, 11, 22], "confidence": 0.9, "track_id": 7},
        {"frame": 1, "bbox": [2, 3, 12, 23], "confidence": 0.8, "track_id": 7},
    ]


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


def test_player_tracking_continues_with_invalid_court_geometry(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    store.set("court", {
        "valid": False,
        "corners_pixel": [(100, 500), (1180, 500), (100, 150), (1180, 150)],
    })

    detections = [
        _make_detection(0, (100, 350, 200, 500)),
        _make_detection(0, (800, 100, 900, 250)),
    ]

    result = PlayerTrackingStage().run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2


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
    """When a near player briefly crosses midline, both tracks keep all detections."""
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

    # Both tracks must have all 100 detections (joint assignment guarantees 1:1
    # per frame even when both players land on the same side of midline)
    for p in players["players"]:
        assert p["detection_count"] == 100, (
            f"Player {p['id']} lost detections: got {p['detection_count']}, expected 100"
        )


def test_track_stitching_both_on_same_side(tmp_job_dir):
    """Joint assignment keeps identity when both players are on the same side."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {
        "valid": True,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
    }
    store.set("court", court_data)

    detections = []
    # Both players near the far side for frames 20-40 (e.g., both at net on far side)
    for f in range(50):
        # Player A: left side of court, moves from near to far at frame 20
        cy_a = 400 if f < 20 else 180
        bbox_a = (100, cy_a - 50, 200, cy_a + 50)
        detections.append(_make_detection(f, bbox_a, track_id=10))

        # Player B: right side, stays near throughout
        cy_b = 400
        bbox_b = (700, cy_b - 50, 800, cy_b + 50)
        detections.append(_make_detection(f, bbox_b, track_id=20))

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2

    for p in players["players"]:
        assert p["detection_count"] == 50, (
            f"Player {p['id']} lost detections: got {p['detection_count']}, expected 50"
        )

    # Player A (left, moved to far) should still be the same persistent track
    a_frames = [p for p in players["players"] if any(
        100 <= d["bbox"][0] <= 200 for d in p["detections"]
    )]
    assert len(a_frames) == 1
    # Player A has detections at both near (frame < 20) and far (frame >= 20)
    a_track = a_frames[0]
    near_frames = [d["frame"] for d in a_track["detections"] if d["bbox"][0] >= 100]
    assert len(near_frames) == 50, "Player A lost identity during side change"


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


def test_stitch_tracks_side_follows_court_position_not_x():
    """The near player may start on the right; side must follow court y, so
    player_1 is still the near (lower) player, not the left one."""
    detections = []
    for f in range(10):
        detections.append(_make_detection(f, (800, 350, 900, 500)))  # near (bottom, right)
        detections.append(_make_detection(f, (100, 100, 200, 250)))  # far (top, left)

    result = stitch_tracks(detections, court_mid_y=300)
    players = result["players"]
    assert len(players) == 2

    near = [p for p in players if p["side"] == "near"][0]
    far = [p for p in players if p["side"] == "far"][0]
    assert near["id"] == "player_1"
    assert all(800 <= d["bbox"][0] <= 900 for d in near["detections"])
    assert all(100 <= d["bbox"][0] <= 200 for d in far["detections"])


def test_stitch_tracks_scene_cut_preserves_identity():
    """Across a pause-record scene cut the players swap halves. Without
    hardening the near player's detections would be mixed into both tracks;
    with it, player_1 (near) keeps the same physical player in both segments."""
    detections = []
    # Segment 1 (frames 0-9): near left-bottom, far right-top
    for f in range(10):
        detections.append(_make_detection(f, (100, 350, 200, 500)))  # near, x~150
        detections.append(_make_detection(f, (800, 100, 900, 250)))  # far,  x~850
    # Big frame gap == scene cut (pause-record)
    # Segment 2 (frames 50-59): players swapped x but same court sides
    for f in range(50, 60):
        detections.append(_make_detection(f, (800, 350, 900, 500)))  # near, x~850
        detections.append(_make_detection(f, (100, 100, 200, 250)))  # far,  x~150

    result = stitch_tracks(detections, court_mid_y=300)
    players = result["players"]
    near = [p for p in players if p["side"] == "near"][0]
    far = [p for p in players if p["side"] == "far"][0]

    assert near["detection_count"] == 20
    assert far["detection_count"] == 20

    # player_1 (near) must contain the near player from BOTH segments, i.e.
    # both x-ranges -- proof identity was preserved across the swap.
    near_left = [d for d in near["detections"] if 100 <= d["bbox"][0] <= 200]
    near_right = [d for d in near["detections"] if 800 <= d["bbox"][0] <= 900]
    assert len(near_left) == 10
    assert len(near_right) == 10

    far_right = [d for d in far["detections"] if 800 <= d["bbox"][0] <= 900]
    far_left = [d for d in far["detections"] if 100 <= d["bbox"][0] <= 200]
    assert len(far_right) == 10
    assert len(far_left) == 10

    assert result["scene_cut_count"] >= 1


def test_stitch_tracks_teleport_jump_preserves_identity():
    """A contiguous-frame teleport (both players jump) must reset matching so
    the near player keeps its track. jump_px is set low to exercise the path."""
    detections = []
    for f in range(5):
        detections.append(_make_detection(f, (100, 350, 200, 500)))  # near, x~150
        detections.append(_make_detection(f, (800, 100, 900, 250)))  # far,  x~850
    # Frame 5: both jump (swap x) with no frame gap -> teleport
    detections.append(_make_detection(5, (800, 350, 900, 500)))  # near now right
    detections.append(_make_detection(5, (100, 100, 200, 250)))  # far now left

    result = stitch_tracks(detections, court_mid_y=300, scene_cut_jump_px=50)
    players = result["players"]
    near = [p for p in players if p["side"] == "near"][0]

    near_left = [d for d in near["detections"] if 100 <= d["bbox"][0] <= 200]
    near_right = [d for d in near["detections"] if 800 <= d["bbox"][0] <= 900]
    assert len(near_left) == 5
    assert len(near_right) == 1
    assert result["scene_cut_count"] >= 1


def test_stitch_tracks_contiguous_no_false_cut():
    """Continuous video with no gaps must not report spurious scene cuts."""
    detections = []
    for f in range(100):
        detections.append(_make_detection(f, (100, 350, 200, 500)))
        detections.append(_make_detection(f, (800, 100, 900, 250)))

    result = stitch_tracks(detections, court_mid_y=300)
    assert result["scene_cut_count"] == 0
    players = result["players"]
    assert all(p["detection_count"] == 100 for p in players)
