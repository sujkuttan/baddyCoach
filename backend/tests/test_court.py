import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import CourtDetectionStage, CourtKeypointDetector


def test_court_detection_with_known_corners(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = CourtDetectionStage()
    result = stage.run(store, config, corners=[
        (100, 500),   # top-left
        (1820, 500),  # top-right
        (100, 100),   # bottom-left
        (1820, 100),  # bottom-right
    ])

    assert result.status == "success"
    assert "court" in result.artifacts
    court_data = store.get("court")
    assert "homography" in court_data
    assert len(court_data["homography"]) == 3
    assert len(court_data["homography"][0]) == 3


def test_court_detection_requires_corners(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = CourtDetectionStage()
    result = stage.run(store, config)

    assert result.status == "error"
    assert "corners" in result.error.lower()


def test_court_detection_with_frame(tmp_job_dir):
    from app.config.settings import settings
    import cv2

    video_path = "videos/test_clip_5s.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    stage = CourtDetectionStage()
    result = stage.run(store, config, frame=frame)

    assert result.status == "success"
    court_data = store.get("court")
    assert court_data["homography"] is not None
    assert len(court_data["corners_pixel"]) == 4


def test_court_detection_reaches_color_line_when_model_returns_rectangle(tmp_job_dir, monkeypatch):
    import app.pipeline.court as court_module

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    model_rectangle = [(100, 500), (1180, 500), (100, 150), (1180, 150)]
    color_trapezoid = [(100, 500), (1180, 500), (250, 150), (1030, 150)]

    monkeypatch.setattr(court_module.CourtKeypointDetector, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(court_module.CourtKeypointDetector, "detect_corners", lambda self, frame: model_rectangle)
    monkeypatch.setattr(court_module, "_detect_court_color_line", lambda frame: color_trapezoid)

    store = ArtifactStore(tmp_job_dir)
    result = court_module.CourtDetectionStage().run(store, StageConfig(), frame=frame)

    assert result.status == "success"
    court_data = store.get("court")
    assert court_data["valid"] is True
    assert court_data["corners_pixel"] == [list(c) for c in color_trapezoid]


def test_manual_near_rectangular_corners_accepted(tmp_job_dir):
    """Manual corners are trusted even when their top/bottom width ratio exceeds
    geometry_max_trapezoid_ratio (straight-on phone footage looks near-rectangular).
    The trapezoid-reliability gate must NOT reject deliberate user input."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    # ratio ~0.95 (> 0.92) -> auto gate would reject, manual must accept
    corners = [(80, 500), (1800, 500), (120, 100), (1760, 100)]
    result = CourtDetectionStage().run(store, config, corners=corners, is_manual=True)
    assert result.status == "success"
    court_data = store.get("court")
    assert court_data["valid"] is True
    assert court_data["detection_method"] == "manual"


def test_auto_near_rectangular_corners_rejected(tmp_job_dir):
    """Same near-rectangular corners, but auto-detected -> still rejected by the
    trapezoid-reliability gate (no special-casing for non-manual input)."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    corners = [(80, 500), (1800, 500), (120, 100), (1760, 100)]
    result = CourtDetectionStage().run(store, config, corners=corners, is_manual=False)
    court_data = store.get("court")
    assert court_data["valid"] is False


def test_invalid_auto_falls_back_to_manual_corners(tmp_job_dir):
    """When auto-detected corners are degenerate (near-rectangular), the stage
    falls back to the supplied manual corners instead of the proportional
    fallback, yielding a valid court-space homography."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    bad_auto = [(80, 500), (1800, 500), (120, 100), (1760, 100)]  # near-rectangular -> invalid
    good_manual = [(148, 637), (1184, 641), (466, 77), (831, 76)]  # trapezoid
    result = CourtDetectionStage().run(
        store, config, corners=bad_auto, is_manual=False,
        manual_corners_fallback=good_manual,
    )
    assert result.status == "success"
    court_data = store.get("court")
    assert court_data["valid"] is True
    assert court_data["detection_method"] == "manual_fallback"


def test_valid_auto_ignores_manual_fallback(tmp_job_dir):
    """When auto-detection succeeds, the manual fallback is not used."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()
    good_auto = [(100, 500), (1180, 500), (250, 150), (1030, 150)]  # trapezoid
    good_manual = [(148, 637), (1184, 641), (466, 77), (831, 76)]
    result = CourtDetectionStage().run(
        store, config, corners=good_auto, is_manual=False,
        manual_corners_fallback=good_manual,
    )
    assert result.status == "success"
    court_data = store.get("court")
    assert court_data["valid"] is True
    assert court_data["detection_method"] != "manual_fallback"


def test_court_keypoint_detector_fallback():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = CourtKeypointDetector("/nonexistent/path.pth")
    assert detector.model is None
    corners = detector.detect_with_fallback(frame)
    assert len(corners) == 4
