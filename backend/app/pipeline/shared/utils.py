"""
Utility functions shared by both colab and backend pipelines.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, COURT_MODEL,
    _detect_court_color_line, _correct_court_points,
    _validate_court_geometry, compute_homography, image_to_court,
    HomographySmoother, make_undistorter,
    foot_midpoint_from_pose, foot_point_from_bbox,
)


def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """Get video information (width, height, fps)."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps


def frame_generator(video_path: str, sample_interval: int = 3, target_fps: int = 10) -> List[np.ndarray]:
    """Generate frames from video with specified sampling interval."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames


def detect_court_from_frame(frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Detect court corners from a single frame."""
    corners = _detect_court_color_line(frame)
    if corners is None:
        return None
    return _correct_court_points(corners)


def compute_court_homography(corners_pixel: List[List[int]]) -> Optional[np.ndarray]:
    """Compute homography mapping image pixels to court metres (legacy wrapper).

    corners_pixel: list of 4 points [bl, br, tl, tr] in image space
    Returns: 3x3 homography matrix (image -> court metres)
    """
    H, _ = compute_homography(corners_pixel)
    return H


# ─── Stroke classification helpers ───────────────────────────────────────────

def _rule_based_shuttle_predict(shuttle_df, frame, vid_w, vid_h):
    """Classify stroke from shuttle trajectory when BST predicts unknown."""
    if shuttle_df is None or len(shuttle_df) == 0:
        return "clear"
    window = shuttle_df[(shuttle_df['frame'] >= frame - 5) & (shuttle_df['frame'] <= frame + 5)]
    if len(window) < 2:
        return "clear"
    y_vals = window['y'].values / vid_h
    x_vals = window['x'].values / vid_w
    valid = (x_vals != 0) | (y_vals != 0)
    if valid.sum() < 2:
        return "clear"
    y_vals = y_vals[valid]
    dy = np.diff(y_vals)
    dx = x_vals[valid][1:] - x_vals[valid][:-1] if len(x_vals[valid]) > 1 else np.array([0.0])
    speed_vals = np.sqrt(dx**2 + dy**2)
    mean_speed = np.mean(speed_vals)
    max_speed = np.max(speed_vals) if len(speed_vals) > 0 else 0
    mean_dy = float(np.mean(dy))
    end_y = float(y_vals[-1])
    if max_speed > 0.15 and mean_dy > 0.05:
        return "smash"
    elif mean_speed < 0.03:
        return "net_shot"
    elif mean_dy < -0.03 and mean_speed > 0.05:
        return "clear"
    elif mean_speed > 0.08 and abs(mean_dy) < 0.02:
        return "drive"
    elif mean_dy > 0.04 and mean_speed > 0.05 and end_y > 0.5:
        return "lift"
    elif end_y > 0.7 and mean_speed < 0.06:
        return "drop"
    else:
        return "clear"


def _detect_handedness(kps: np.ndarray) -> str:
    """Detect handedness from pose keypoints.

    Uses wrist keypoint confidence (index 2) and relative position:
    the playing hand typically has higher confidence and is raised higher
    during overhead strokes. Left = COCO index 9, Right = COCO index 10.
    """
    left_conf = kps[9, 2] if kps.shape[1] > 2 else 0.5
    right_conf = kps[10, 2] if kps.shape[1] > 2 else 0.5
    # During a smash/clear, the playing wrist is typically above the shoulder
    left_above = kps[9, 1] < kps[5, 1] if kps[9, 1] != 0 and kps[5, 1] != 0 else False
    right_above = kps[10, 1] < kps[6, 1] if kps[10, 1] != 0 and kps[6, 1] != 0 else False
    if left_above and not right_above:
        return "left"
    if right_above and not left_above:
        return "right"
    return "right" if right_conf >= left_conf else "left"


def _compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 formed by vectors p1-p2 and p3-p2, in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def _angle_score(angle: float, ideal_min: float, ideal_max: float,
                 max_deviation: float = 60) -> float:
    """Map joint angle [degrees] to [0, 1] technique score.

    1.0 when angle is within [ideal_min, ideal_max],
    linear drop to 0.0 outside that range.
    """
    if ideal_min <= angle <= ideal_max:
        return 1.0
    deviation = min(abs(angle - ideal_min), abs(angle - ideal_max))
    return max(0.0, 1.0 - deviation / max_deviation)


def _get_playing_arm_kps(kps: np.ndarray, handedness: str) -> dict:
    """Extract playing-side keypoints as {shoulder, elbow, wrist, knee, hip, ankle}."""
    if handedness == "left":
        return {
            "shoulder": kps[5][:2], "elbow": kps[7][:2], "wrist": kps[9][:2],
            "knee": kps[13][:2], "hip": kps[11][:2], "ankle": kps[15][:2],
        }
    return {
        "shoulder": kps[6][:2], "elbow": kps[8][:2], "wrist": kps[10][:2],
        "knee": kps[14][:2], "hip": kps[12][:2], "ankle": kps[16][:2],
    }


def _evaluate_shot(stroke_type: str, kps: np.ndarray) -> float:
    """Evaluate shot technique using biomechanical joint angles."""
    if kps.shape != (17, 3):
        return 0.5

    handedness = _detect_handedness(kps)
    arm = _get_playing_arm_kps(kps, handedness)

    S, E, W = arm["shoulder"], arm["elbow"], arm["wrist"]
    H, K, A = arm["hip"], arm["knee"], arm["ankle"]

    shoulder_angle = _compute_angle(E, S, H)
    elbow_angle = _compute_angle(W, E, S)
    knee_angle = _compute_angle(A, K, H)

    if stroke_type == "smash":
        scores = [
            _angle_score(shoulder_angle, 120, 180, 50),
            _angle_score(elbow_angle, 150, 180, 40),
            1.0 if W[1] < S[1] else max(0.0, (S[1] - W[1]) / 100.0),
        ]
        return float(np.mean(scores))

    elif stroke_type == "clear":
        scores = [
            _angle_score(shoulder_angle, 90, 150, 50),
            _angle_score(elbow_angle, 130, 170, 50),
        ]
        return float(np.mean(scores))

    elif stroke_type == "net_shot":
        scores = [
            _angle_score(knee_angle, 90, 130, 50),
            _angle_score(shoulder_angle, 30, 90, 40),
            _angle_score(elbow_angle, 100, 160, 50),
        ]
        return float(np.mean(scores))

    elif stroke_type == "drive":
        scores = [
            _angle_score(elbow_angle, 120, 160, 50),
            _angle_score(shoulder_angle, 60, 120, 50),
        ]
        return float(np.mean(scores))

    elif stroke_type == "lift":
        scores = [
            _angle_score(shoulder_angle, 100, 160, 50),
            _angle_score(elbow_angle, 140, 175, 40),
        ]
        return float(np.mean(scores))

    elif stroke_type == "drop":
        scores = [
            _angle_score(shoulder_angle, 100, 160, 50),
            _angle_score(elbow_angle, 100, 150, 40),
            1.0 if W[1] < S[1] else max(0.0, (S[1] - W[1]) / 100.0),
        ]
        return float(np.mean(scores))

    elif stroke_type == "block":
        body_height = max(K[1], H[1]) - S[1] + 1e-6
        elbow_proximity = abs(E[0] - S[0]) / body_height
        scores = [
            max(0.0, 1.0 - elbow_proximity / 0.5),
            _angle_score(elbow_angle, 60, 120, 50),
        ]
        return float(np.mean(scores))

    elif stroke_type == "rush":
        scores = [
            _angle_score(knee_angle, 80, 120, 50),
            _angle_score(elbow_angle, 130, 170, 50),
        ]
        return float(np.mean(scores))

    return 0.5


# ─── Rally segmentation helpers ──────────────────────────────────────────────

def _infer_end_reason(stroke_type: str, confidence: float) -> str:
    """Infer rally end reason from the last shot.

    Rules:
    - High-confidence smash/drop/kill -> winner (aggressive finishing shot)
    - Net shot -> net (hitter hit the net)
    - Low-confidence clear/drive/lift -> unforced_error (weak basic shot)
    - Everything else -> forced_error (opponent won, not necessarily an error by hitter)
    """
    if stroke_type in ("smash", "drop", "kill") and confidence >= 0.5:
        return "winner"
    if stroke_type in ("net_shot",):
        return "net"
    if stroke_type in ("clear", "drive", "lift") and confidence < 0.35:
        return "unforced_error"
    return "forced_error"


def _is_rally_ending_shot(stroke_type: str, confidence: float, next_gap: int) -> bool:
    """Determine if a shot likely ended the rally.

    Uses stroke type, confidence, AND the gap to the next shot as signals.
    A shot is considered rally-ending if:
    1. It's followed by a gap > 45 frames (primary signal)
    2. It's a high-confidence winner (smash/drop/kill with conf >= 0.6) AND gap > 25 frames
    3. It's a net shot AND gap > 15 frames (net shots often end rallies quickly)
    """
    if next_gap > 45:
        return True
    if stroke_type in ("smash", "drop", "kill") and confidence >= 0.6 and next_gap > 25:
        return True
    if stroke_type in ("net_shot",) and next_gap > 15:
        return True
    return False


# ─── Rally statistics helper ─────────────────────────────────────────────────

def stage_rally_stats(shots_data: list, rallies_data: list) -> dict:
    """Compute rally-level statistics for coaching."""
    from collections import Counter

    stats = {"avg_length": 0, "max_length": 0, "min_length": 0,
             "first_shot_win_rate": 0, "long_rally_pct": 0}
    if not rallies_data or not shots_data:
        return stats

    lengths = [r["shot_count"] for r in rallies_data]
    stats["avg_length"] = float(np.mean(lengths))
    stats["max_length"] = int(np.max(lengths))
    stats["min_length"] = int(np.min(lengths))

    shots_df = pd.DataFrame(shots_data)
    rallies_df = pd.DataFrame(rallies_data)
    first_shot_wins = 0
    total_rallies = len(rallies_df)

    for _, rally in rallies_df.iterrows():
        rally_shots = shots_df[shots_df["rally_id"] == rally["rally_id"]].sort_values("frame")
        if len(rally_shots) == 0:
            continue
        winner = rally.get("winner_player_id")
        if winner and rally_shots.iloc[0].get("player_id") == winner:
            first_shot_wins += 1

    stats["first_shot_win_rate"] = first_shot_wins / total_rallies if total_rallies > 0 else 0
    long_rallies = sum(1 for l in lengths if l > 10)
    stats["long_rally_pct"] = long_rallies / len(lengths) if lengths else 0

    return stats