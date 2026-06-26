import pandas as pd
import numpy as np
import pytest
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import (
    RallySegmentationStage, _is_rally_ending_shot, _infer_end_reason,
    _find_dead_shuttle_window, _winner_from_shuttle_landing,
)


@pytest.fixture
def court_data():
    """Standard court with net at y=390 (midpoint of 720p frame)."""
    return {
        "corners_pixel": [[100, 680], [1180, 680], [100, 100], [1180, 100]],
        "valid": True,
    }


@pytest.fixture
def court_data_with_homography():
    """Same court but with a real homography (used by production code path)."""
    import cv2
    from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, COURT_MODEL

    src = np.array([[100, 680], [1180, 680], [100, 100], [1180, 100]], dtype=np.float64)
    dst = np.array([
        COURT_MODEL["outer_bl"], COURT_MODEL["outer_br"],
        COURT_MODEL["outer_tl"], COURT_MODEL["outer_tr"],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return {
        "corners_pixel": [[100, 680], [1180, 680], [100, 100], [1180, 100]],
        "homography": H.tolist() if H is not None else None,
        "valid": True,
    }


def test_rally_segmentation_groups_shots(tmp_job_dir):
    """Test basic rally segmentation with time gaps."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    # Rallies separated by large gaps (>60 frames)
    shots_df = pd.DataFrame({
        "frame": [0, 5, 10, 15, 20, 100, 105, 110, 115, 200, 205, 210],
        "stroke_type": ["serve", "clear", "drop", "smash", "clear",
                       "serve", "drop", "lift", "clear",
                       "serve", "smash", "drop"],
        "player_id": ["player_1", "player_2", "player_1", "player_2", "player_1",
                      "player_2", "player_1", "player_2", "player_1",
                      "player_1", "player_2", "player_1"],
        "stroke_confidence": [0.9, 0.7, 0.6, 0.8, 0.7,
                             0.9, 0.6, 0.5, 0.7,
                             0.9, 0.8, 0.7],
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=60)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    assert len(rallies_df) == 3
    assert "rally_id" in rallies_df.columns
    assert "start_frame" in rallies_df.columns
    assert "end_frame" in rallies_df.columns
    assert "shot_count" in rallies_df.columns


def test_is_rally_ending_shot_high_conf_smash():
    # High-confidence smash with moderate gap ends rally
    assert _is_rally_ending_shot("smash", 0.7, 30) is True
    assert _is_rally_ending_shot("smash", 0.6, 26) is True
    # High-confidence smash with small gap does NOT end rally
    assert _is_rally_ending_shot("smash", 0.5, 20) is False
    assert _is_rally_ending_shot("smash", 0.7, 10) is False


def test_is_rally_ending_shot_net_shot():
    # Net shot with gap > 45 ends rally (threshold raised from 15 to 45)
    assert _is_rally_ending_shot("net_shot", 0.3, 50) is True
    assert _is_rally_ending_shot("net_shot", 0.9, 46) is True
    # Net shot with small gap does NOT end rally
    assert _is_rally_ending_shot("net_shot", 0.5, 5) is False
    assert _is_rally_ending_shot("net_shot", 0.5, 40) is False


def test_is_rally_ending_shot_large_gap():
    # Large gap always ends rally regardless of stroke type
    assert _is_rally_ending_shot("clear", 0.6, 95) is True
    assert _is_rally_ending_shot("lift", 0.5, 91) is True
    # Below primary threshold does NOT end rally
    assert _is_rally_ending_shot("clear", 0.6, 50) is False
    assert _is_rally_ending_shot("lift", 0.5, 46) is False


def test_is_rally_ending_shot_normal_shot():
    # Normal shots with small gaps do NOT end rallies
    assert _is_rally_ending_shot("clear", 0.6, 15) is False
    assert _is_rally_ending_shot("drop", 0.4, 10) is False
    assert _is_rally_ending_shot("lift", 0.5, 12) is False


def test_rally_segmentation_with_net_shot_ending(tmp_job_dir):
    """Net shot should end a rally with moderate gap."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    # Rally 1: serve -> clear -> drop -> smash -> net_shot (ends rally at frame 18, gap=47 to next)
    # Rally 2: serve -> clear -> drop
    shots_df = pd.DataFrame({
        "frame": [0, 5, 10, 15, 18, 65, 70, 75],
        "stroke_type": ["serve", "clear", "drop", "smash", "net_shot", "serve", "clear", "drop"],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                      "player_1", "player_2", "player_1", "player_2"],
        "stroke_confidence": [0.9, 0.7, 0.6, 0.8, 0.5, 0.9, 0.7, 0.6],
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=60)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    # Rally 1 ends at frame 18 (net_shot + gap=47 > 45), Rally 2 starts at frame 65
    assert len(rallies_df) == 2
    assert rallies_df.iloc[0]["end_frame"] == 18
    assert rallies_df.iloc[1]["start_frame"] == 65


def test_find_dead_shuttle_window_detects_dead_zone():
    n = 40
    frames = list(range(n))
    x = [100.0 + t * 3.0 for t in range(10)] + [130.0] * 25 + [130.0 + (t - 35) * 2.0 for t in range(35, n)]
    y = [200.0] * n
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * n,
    })
    assert _find_dead_shuttle_window(shuttle_df, 0, 39) is True


def test_find_dead_shuttle_window_no_dead_zone():
    n = 40
    frames = list(range(n))
    x = [100.0 + t * 5.0 for t in range(n)]
    y = [200.0] * n
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * n,
    })
    assert _find_dead_shuttle_window(shuttle_df, 0, 39) is False


def test_find_dead_shuttle_window_none_shuttle_df():
    assert _find_dead_shuttle_window(None, 0, 100) is False


def test_find_dead_shuttle_window_lost_track():
    n = 40
    frames = list(range(n))
    x = [100.0 + t * 5.0 for t in range(10)] + [np.nan] * 25 + [200.0 + (t - 35) * 3.0 for t in range(35, n)]
    y = [200.0] * n
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * 10 + [0.05] * 25 + [0.95] * (n - 35),
    })
    assert _find_dead_shuttle_window(shuttle_df, 0, 39) is True


def test_winner_from_shuttle_landing_near_side(court_data):
    n = 80
    frames = list(range(n))
    x = [500.0] * n
    y = (list(range(100, 610, 10)) + [600] * 30)[:n]
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * n,
    })
    winner = _winner_from_shuttle_landing(shuttle_df, 0, 55, court_data)
    # Shuttle ends static at y=600 (bottom half, y > net_y=390) → far side wins
    assert winner == "player_2"


def test_winner_from_shuttle_landing_far_side(court_data):
    n = 80
    frames = list(range(n))
    x = [500.0] * n
    y = (list(reversed(range(100, 610, 10))) + [150] * 30)[:n]
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * n,
    })
    winner = _winner_from_shuttle_landing(shuttle_df, 0, 55, court_data)
    # Shuttle ends static at y=150 (top half, y < net_y=390) → near side wins
    assert winner == "player_1"


def test_winner_from_shuttle_landing_no_shuttle():
    assert _winner_from_shuttle_landing(
        pd.DataFrame(columns=["frame", "x", "y", "confidence"]), 0, 100, None,
    ) is None


def test_winner_from_shuttle_landing_homography_near_side(court_data_with_homography):
    """Regression: homography path uses length axis (index 0), not width axis."""
    n = 80
    frames = list(range(n))
    x = [500.0] * n
    y = (list(range(100, 610, 10)) + [600] * 30)[:n]
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * n,
    })
    winner = _winner_from_shuttle_landing(
        shuttle_df, 0, 55, court_data_with_homography,
    )
    # Shuttle at bottom of frame → near court (X > 6.7) → near player lost → far wins
    assert winner is not None, "Homography path should return a winner"


def test_winner_from_shuttle_landing_homography_far_side(court_data_with_homography):
    """Regression: shuttle at top of frame → far court → winner = near player."""
    n = 80
    frames = list(range(n))
    x = [500.0] * n
    y = (list(reversed(range(100, 610, 10))) + [150] * 30)[:n]
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": x, "y": y,
        "confidence": [0.95] * n,
    })
    winner = _winner_from_shuttle_landing(
        shuttle_df, 0, 55, court_data_with_homography,
    )
    assert winner == "player_1"


def test_find_dead_shuttle_window_too_short():
    """Segment smaller than dead_frames threshold returns False."""
    frames = [0, 1, 2]
    shuttle_df = pd.DataFrame({
        "frame": frames, "x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0],
        "confidence": [0.95] * 3,
    })
    assert _find_dead_shuttle_window(shuttle_df, 0, 2) is False


def test_rally_split_by_dead_shuttle(tmp_job_dir):
    """A dead-shuttle window mid-sequence splits rally without large frame gap."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    # Shots at 0,10,20,30 and 80,90,100,110 (gap between 30 and 80 = 50, < gap_threshold=90)
    shots_df = pd.DataFrame({
        "frame": [0, 10, 20, 30, 80, 90, 100, 110],
        "stroke_type": ["serve", "clear", "drop", "smash",
                        "serve", "clear", "drop", "clear"],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                      "player_1", "player_2", "player_1", "player_2"],
        "stroke_confidence": [0.9, 0.7, 0.6, 0.8, 0.9, 0.7, 0.6, 0.7],
    })
    store.set_parquet("shots", shots_df)

    # Shuttle: moving 0-35, dead (static) 36-65, moving 66-120
    shuttle_frames = list(range(0, 120))
    x = (
        [100.0 + t * 3.0 for t in range(36)]  # 0-35: moving
        + [100 + 35*3] * 30                     # 36-65: static (dead)
        + [205.0 + (t - 66) * 2.0 for t in range(66, 120)]  # 66-119: moving
    )
    y = [200.0] * 120
    shuttle_df = pd.DataFrame({
        "frame": shuttle_frames, "x": x, "y": y,
        "confidence": [0.95] * 120,
    })
    store.set_parquet("shuttle", shuttle_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=90)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    # Should split into 2 rallies: first at 0-30, second at 80-110
    # (dead shuttle between frame 36-65 triggers split, not the 50-frame gap)
    assert len(rallies_df) == 2
    assert rallies_df.iloc[0]["end_frame"] == 30
    assert rallies_df.iloc[1]["start_frame"] == 80


def test_winner_attribution_gate(tmp_job_dir):
    """Shot with untrustworthy attribution tier returns None winner."""
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20, 30, 100, 110, 120],
        "stroke_type": ["serve", "clear", "drop", "smash", "serve", "clear", "drop"],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                      "player_1", "player_2", "player_1"],
        "stroke_confidence": [0.9, 0.7, 0.6, 0.8, 0.9, 0.7, 0.6],
        "attribution_tier": ["final", "final", "final", "rally_fallback",
                            "final", "final", "final"],
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=60)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    # First rally's winner should be None (last shot has rally_fallback tier)
    assert len(rallies_df) >= 1
    assert pd.isna(rallies_df.iloc[0]["winner_player_id"])

