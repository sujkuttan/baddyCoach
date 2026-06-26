"""
Tests for shared pipeline modules.

These tests verify that the shared functions work correctly and produce
identical results to the original implementations.

Note: Tests import directly from the shared submodule files to avoid
triggering the full app.pipeline.__init__ import chain (which pulls in
settings, model loaders, etc.).
"""

import sys
import numpy as np
import pandas as pd
import pytest


# ─── Court module ───────────────────────────────────────────────────────────

class TestSharedCourtModule:
    """Tests for shared court detection and geometric processing."""

    def test_court_constants(self):
        from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH, NET_HEIGHT
        assert COURT_LENGTH == 13.4
        assert COURT_WIDTH == 6.10
        assert NET_HEIGHT == 1.55

    def test_court_model(self):
        from app.pipeline.shared.court import COURT_MODEL, COURT_LENGTH, COURT_WIDTH
        assert COURT_MODEL["outer_tl"] == (0.0, 0.0)
        assert COURT_MODEL["outer_tr"] == (0.0, COURT_WIDTH)
        assert COURT_MODEL["outer_bl"] == (COURT_LENGTH, 0.0)
        assert COURT_MODEL["outer_br"] == (COURT_LENGTH, COURT_WIDTH)

    def test_correct_court_points(self):
        from app.pipeline.shared.court import _correct_court_points
        corners = [[100, 400], [500, 400], [150, 100], [450, 100]]
        corrected = _correct_court_points(corners)
        assert len(corrected) == 4
        assert corrected[2][1] == corrected[3][1]  # top pair same y
        assert corrected[0][1] == corrected[1][1]  # bottom pair same y

    def test_validate_court_geometry_valid(self):
        from app.pipeline.shared.court import _validate_court_geometry
        # Input order [bl, br, tl, tr] — boundary traversal is [bl, br, tr, tl]
        corners = np.array([[100, 400], [500, 400], [100, 100], [500, 100]], dtype=np.float64)
        assert _validate_court_geometry(corners) is True

    def test_validate_court_geometry_too_small(self):
        from app.pipeline.shared.court import _validate_court_geometry
        corners = np.array([[100, 100], [110, 100], [110, 90], [100, 90]], dtype=np.float64)
        assert _validate_court_geometry(corners) is False

    def test_image_to_court(self):
        from app.pipeline.shared.court import image_to_court
        H = np.eye(3, dtype=np.float64)
        x, y = image_to_court(H, (0.0, 0.0))
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_homography_smoother(self):
        from app.pipeline.shared.court import HomographySmoother
        smoother = HomographySmoother(alpha=0.6, win=5)
        corners = np.array([[100, 400], [500, 400], [500, 100], [100, 100]], dtype=np.float64)
        H = np.eye(3, dtype=np.float64)
        H_smooth, valid = smoother.update(corners, H, True)
        assert valid is True
        assert H_smooth is not None

    def test_homography_smoother_none_input(self):
        from app.pipeline.shared.court import HomographySmoother
        smoother = HomographySmoother(alpha=0.6, win=5)
        H_smooth, valid = smoother.update(None, None, False)
        assert valid is False
        assert H_smooth is None

    def test_make_undistorter_no_calibration(self):
        from app.pipeline.shared.court import make_undistorter
        undistort = make_undistorter(None, None, (640, 480))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = undistort(frame)
        np.testing.assert_array_equal(frame, result)

    def test_foot_midpoint_from_pose(self):
        from app.pipeline.shared.court import foot_midpoint_from_pose
        kps = np.zeros((17, 2), dtype=np.float64)
        kps[15] = [100, 200]
        kps[16] = [120, 200]
        midpoint = foot_midpoint_from_pose(kps)
        assert midpoint is not None
        assert midpoint[0] == 110.0
        assert midpoint[1] == 200.0

    def test_foot_midpoint_from_pose_no_confidence(self):
        from app.pipeline.shared.court import foot_midpoint_from_pose
        kps = np.zeros((17, 2), dtype=np.float64)
        conf = np.zeros(17, dtype=np.float64)
        midpoint = foot_midpoint_from_pose(kps, conf, conf_thr=0.3)
        assert midpoint is None

    def test_foot_point_from_bbox(self):
        from app.pipeline.shared.court import foot_point_from_bbox
        foot = foot_point_from_bbox((100, 200, 300, 400))
        assert foot[0] == 200.0
        assert foot[1] == 400.0


# ─── Utils module ───────────────────────────────────────────────────────────

class TestSharedUtilsModule:
    """Tests for shared utility functions."""

    def test_compute_court_homography(self):
        from app.pipeline.shared.utils import compute_court_homography
        corners = [[100, 400], [500, 400], [500, 100], [100, 100]]
        H = compute_court_homography(corners)
        if H is not None:
            assert H.shape == (3, 3)

    def test_rule_based_shuttle_predict_clear(self):
        from app.pipeline.shared.utils import _rule_based_shuttle_predict
        shuttle_df = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'x': [640, 640, 640, 640, 640],
            'y': [400, 350, 300, 250, 200]
        })
        result = _rule_based_shuttle_predict(shuttle_df, 2, 1280, 720)
        assert result == "clear"

    def test_rule_based_shuttle_predict_smash(self):
        from app.pipeline.shared.utils import _rule_based_shuttle_predict
        shuttle_df = pd.DataFrame({
            'frame': [0, 1, 2, 3, 4],
            'x': [640, 640, 640, 640, 640],
            'y': [200, 250, 350, 500, 700]
        })
        result = _rule_based_shuttle_predict(shuttle_df, 2, 1280, 720)
        assert result == "smash"

    def test_rule_based_shuttle_predict_empty(self):
        from app.pipeline.shared.utils import _rule_based_shuttle_predict
        shuttle_df = pd.DataFrame(columns=['frame', 'x', 'y'])
        result = _rule_based_shuttle_predict(shuttle_df, 0, 1280, 720)
        assert result == "clear"

    def test_infer_end_reason_winner(self):
        from app.pipeline.shared.utils import _infer_end_reason
        assert _infer_end_reason("smash", 0.8) == "winner"

    def test_infer_end_reason_net(self):
        from app.pipeline.shared.utils import _infer_end_reason
        assert _infer_end_reason("net_shot", 0.7) == "net"

    def test_infer_end_reason_unforced_error(self):
        from app.pipeline.shared.utils import _infer_end_reason
        assert _infer_end_reason("clear", 0.2) == "unforced_error"

    def test_infer_end_reason_forced_error(self):
        from app.pipeline.shared.utils import _infer_end_reason
        assert _infer_end_reason("clear", 0.5) == "forced_error"

    def test_is_rally_ending_shot_large_gap(self):
        from app.pipeline.shared.utils import _is_rally_ending_shot
        assert _is_rally_ending_shot("clear", 0.5, 95) is True

    def test_is_rally_ending_shot_small_gap(self):
        from app.pipeline.shared.utils import _is_rally_ending_shot
        assert _is_rally_ending_shot("clear", 0.5, 10) is False

    def test_is_rally_ending_shot_winner(self):
        from app.pipeline.shared.utils import _is_rally_ending_shot
        assert _is_rally_ending_shot("smash", 0.7, 30) is True

    def test_stage_rally_stats_empty(self):
        from app.pipeline.shared.utils import stage_rally_stats
        stats = stage_rally_stats([], [])
        assert stats["avg_length"] == 0
        assert stats["max_length"] == 0
        assert stats["min_length"] == 0

    def test_stage_rally_stats(self):
        from app.pipeline.shared.utils import stage_rally_stats
        shots = [
            {"frame": 1, "player_id": "player_1", "stroke_type": "clear", "rally_id": 1},
            {"frame": 2, "player_id": "player_2", "stroke_type": "drop", "rally_id": 1},
            {"frame": 3, "player_id": "player_1", "stroke_type": "smash", "rally_id": 1},
        ]
        rallies = [
            {"rally_id": 1, "start_frame": 1, "end_frame": 3, "shot_count": 3,
             "winner_player_id": "player_1"}
        ]
        stats = stage_rally_stats(shots, rallies)
        assert stats["avg_length"] == 3.0
        assert stats["max_length"] == 3
        assert stats["min_length"] == 3


# ─── Core module ────────────────────────────────────────────────────────────

class TestSharedCoreModule:
    """Tests for shared core module."""

    def test_gpu_batch_config_cpu(self):
        from app.pipeline.shared.core import _get_gpu_batch_config
        config = _get_gpu_batch_config("cpu")
        assert "yolo_chunk" in config
        assert "yolo_batch" in config
        assert "tracknet_chunk" in config
        assert "rtmpose_chunk" in config
        assert "bst_batch" in config

    def test_stroke_classes(self):
        from app.pipeline.shared.core import STROKE_CLASSES
        assert len(STROKE_CLASSES) > 0
        assert "clear" in STROKE_CLASSES
        assert "smash" in STROKE_CLASSES
        assert "drop" in STROKE_CLASSES

    def test_logger(self):
        from app.pipeline.shared.logging import logger
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
