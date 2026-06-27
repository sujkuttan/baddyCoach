"""
Utility functions shared by both colab and backend pipelines.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from app.config.settings import settings

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


def _find_dead_shuttle_window(
    shuttle_df: pd.DataFrame | None,
    start_frame: int,
    end_frame: int,
    min_gap_frames: int | None = None,
) -> bool:
    """Check if the shuttle track between two frames has a dead-shuttle window.

    A "dead shuttle" means ≥ rally_dead_frames consecutive frames where:
      - speed < rally_dead_speed_px (shuttle stopped moving), OR
      - confidence < shuttle_min_conf (track lost — below net / out of frame)

    Returns True if such a window exists, meaning the rally likely ended
    between start_frame and end_frame.
    """
    if shuttle_df is None or len(shuttle_df) < 3:
        return False

    segment = shuttle_df[
        (shuttle_df["frame"] >= start_frame) &
        (shuttle_df["frame"] <= end_frame)
    ].copy().sort_values("frame")

    if len(segment) < settings.rally_dead_frames:
        return False

    dead_frames = settings.rally_dead_frames
    min_conf = settings.shuttle_min_conf

    x = segment["x"].values.astype(np.float64)
    y = segment["y"].values.astype(np.float64)
    conf = segment["confidence"].values.astype(np.float64)

    # Per-frame speed: displacement from previous frame (NaN if either is NaN)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx * dx + dy * dy)
    speed = np.concatenate([[np.nan], speed])  # frame 0 has no predecessor

    # Dead if speed < threshold OR confidence collapsed
    dead = (speed < settings.rally_dead_speed_px) | (conf < min_conf)

    # Slide a window looking for dead_frames consecutive True
    count = 0
    for d in dead:
        if d:
            count += 1
            if count >= dead_frames:
                return True
        else:
            count = 0

    return False


def _winner_from_shuttle_landing(
    shuttle_raw: pd.DataFrame,
    rally_start: int,
    rally_end: int,
    court: dict | None = None,
    players: dict | None = None,
) -> str | None:
    """Determine rally winner from where the shuttle landed.

    Reads shuttle_raw (confidence-gated only, no interpolation/smoothing)
    so the landing point is the real last detection, not a carried-forward fill.
    Scans the shuttle track after the last shot for a dead-shuttle window,
    then determines which side of the court the shuttle died on.
    The side the shuttle died on = the side that failed to return →
    the opponent wins.

    Side determination uses two independent cues with voting:
    1. Pixel-midline: landing y vs net y from court corners (always available)
    2. Homography court-half: court_xy[0] vs court_length/2 (when valid)
    If both agree → confident. If they disagree or homography is out of
    bounds → use pixel-midline.

    Returns winner_player_id ("player_1" or "player_2"), or None if
    undetermined.
    """
    search_end_frame = rally_end + settings.rally_winner_search_frames
    segment = shuttle_raw[
        (shuttle_raw["frame"] >= rally_end) &
        (shuttle_raw["frame"] <= search_end_frame)
    ].copy().sort_values("frame")

    if len(segment) < settings.rally_dead_frames:
        return None

    x = segment["x"].values.astype(np.float64)
    y = segment["y"].values.astype(np.float64)
    conf = segment["confidence"].values.astype(np.float64)

    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx * dx + dy * dy)
    speed = np.concatenate([[np.nan], speed])

    dead = (speed < settings.rally_dead_speed_px) | (conf < settings.shuttle_min_conf)

    # Find the first long dead window
    dead_start = None
    count = 0
    for i, d in enumerate(dead):
        if d:
            if count == 0:
                dead_start = i
            count += 1
            if count >= settings.rally_dead_frames:
                break
        else:
            dead_start = None
            count = 0

    if dead_start is None:
        return None

    # Find the last HIGH-CONF point before the track dies (not the last filled NaN)
    # Scan the dead window for the last point with confidence >= min_landing_conf
    landing_idx = None
    search_end = min(dead_start + settings.rally_dead_frames * 2, len(segment))
    for i in range(search_end - 1, dead_start - 1, -1):  # scan backwards
        if not np.isnan(x[i]) and not np.isnan(y[i]) and conf[i] >= settings.rally_winner_min_landing_conf:
            landing_idx = i
            break

    if landing_idx is None:
        # Fall back to last non-NaN point regardless of confidence
        for i in range(search_end - 1, dead_start - 1, -1):
            if not np.isnan(x[i]) and not np.isnan(y[i]):
                landing_idx = i
                break

    if landing_idx is None:
        return None

    lx, ly = float(x[landing_idx]), float(y[landing_idx])

    # ── Two-cue side determination ──────────────────────────────────────
    # Cue 1: Pixel-midline (camera-robust, always available with corners)
    midline_cue = None
    if court and court.get("corners_pixel"):
        corners = court["corners_pixel"]
        bl_y = corners[0][1]
        tl_y = corners[2][1]
        net_y = (tl_y + bl_y) / 2.0
        midline_cue = ly < net_y  # True = above net line = far side
    elif court and court.get("valid", False) and court.get("homography") is not None:
        from .court import image_to_court
        H = np.array(court["homography"], dtype=np.float64)
        court_xy = image_to_court(H, (lx, ly))
        if court_xy is not None:
            midline_cue = court_xy[0] < settings.court_length / 2.0

    # Cue 2: Homography court-half (when court is valid and point is in bounds)
    homography_cue = None
    if court and court.get("homography") is not None and court.get("valid", False):
        from .court import image_to_court
        H = np.array(court["homography"], dtype=np.float64)
        court_xy = image_to_court(H, (lx, ly))
        if court_xy is not None:
            cx, cy = court_xy
            if 0 <= cx <= settings.court_length and 0 <= cy <= settings.court_width:
                homography_cue = cx < settings.court_length / 2.0

    # Vote: prefer agreement, otherwise trust midline
    if midline_cue is not None and homography_cue is not None:
        if midline_cue == homography_cue:
            shuttle_on_far_side = midline_cue
        else:
            # Disagreement → trust midline (camera-robust)
            shuttle_on_far_side = midline_cue
    elif midline_cue is not None:
        shuttle_on_far_side = midline_cue
    elif homography_cue is not None:
        shuttle_on_far_side = homography_cue
    else:
        return None

    # Map side to player_id via players artifact
    if players and "players" in players:
        for p in players["players"]:
            p_side = p.get("side", "")
            p_id = p.get("id", "")
            # Shuttle on far side → far-side player lost → opponent (near player) wins
            if shuttle_on_far_side and p_side == "near":
                return p_id
            elif not shuttle_on_far_side and p_side == "far":
                return p_id

    # Fallback: heuristic mapping (player_1 = far, player_2 = near)
    return "player_1" if shuttle_on_far_side else "player_2"


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


# ─── Rally segmentation helpers ──────────────────────────────────────────────

def _infer_end_reason(stroke_type: str, confidence: float,
                      last_shot_speed: float | None = None) -> str:
    """Infer rally end reason from the last shot.

    Rules:
    - Smash/drop/kill with moderate confidence -> winner (aggressive finishing shot)
    - High-speed smash (>8 m/s) or smash within 2m of net -> winner
    - Net shot -> net (hitter hit the net)
    - Low-confidence clear/drive/lift -> unforced_error (weak basic shot)
    - Everything else -> forced_error (opponent won, not necessarily an error by hitter)
    """
    if stroke_type in ("smash", "drop", "kill"):
        if confidence >= 0.3:
            return "winner"
        if stroke_type == "smash" and last_shot_speed is not None and last_shot_speed > 8.0:
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
    if next_gap > settings.rally_ending_gap_primary:
        return True
    if stroke_type in ("smash", "drop", "kill") and confidence >= settings.rally_ending_high_conf_min and next_gap > settings.rally_ending_gap_high_conf:
        return True
    if stroke_type in ("net_shot",) and next_gap > settings.rally_ending_gap_net:
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