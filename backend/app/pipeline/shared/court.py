"""
Court detection and geometric processing shared by both colab and backend pipelines.
"""

import numpy as np
import cv2
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Court model definition (metres) — origin at top-left, x=along length, y=along width
# Default COURT_WIDTH = 6.10 (doubles). Some videos may show singles court (5.18m wide
# with inner sidelines); stages should use court_width from store if available.
COURT_LENGTH = 13.4
COURT_WIDTH = 6.10
SINGLES_WIDTH = 5.18
NET_HEIGHT = 1.55

COURT_MODEL = {
    "outer_tl": (0.0, 0.0),
    "outer_tr": (0.0, COURT_WIDTH),
    "outer_bl": (COURT_LENGTH, 0.0),
    "outer_br": (COURT_LENGTH, COURT_WIDTH),
    # Singles reference lines (inside doubles court)
    "singles_tl": (0.0, (COURT_WIDTH - SINGLES_WIDTH) / 2),
    "singles_tr": (0.0, (COURT_WIDTH + SINGLES_WIDTH) / 2),
    "singles_bl": (COURT_LENGTH, (COURT_WIDTH - SINGLES_WIDTH) / 2),
    "singles_br": (COURT_LENGTH, (COURT_WIDTH + SINGLES_WIDTH) / 2),
}

COURT_ASPECT_RATIO = COURT_LENGTH / COURT_WIDTH  # ≈ 2.197


def _detect_court_color_line(frame: np.ndarray):
    """Detect court using color segmentation + HoughLinesP. Returns [bl, br, tl, tr] or None."""
    import cv2
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255]))
    floor_mask = cv2.bitwise_or(green_mask, blue_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > w * h * 0.10:
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float64)
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1).flatten()
                return [
                    pts[np.argmax(d)].tolist(),
                    pts[np.argmin(d)].tolist(),
                    pts[np.argmin(s)].tolist(),
                    pts[np.argmax(s)].tolist(),
                ]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=w * 0.2, maxLineGap=10)
    if lines is not None and len(lines) >= 4:
        lines_flat = lines.reshape(-1, 4)
        s = lines_flat[:, 0] + lines_flat[:, 1]
        d = lines_flat[:, 0] - lines_flat[:, 1]
        sorted_by_sum = lines_flat[np.argsort(s)]
        sorted_by_diff = lines_flat[np.argsort(d)]
        corners = [
            sorted_by_diff[-1][:2].tolist(),
            sorted_by_diff[0][2:].tolist(),
            sorted_by_sum[0][:2].tolist(),
            sorted_by_sum[-1][2:].tolist(),
        ]
        return corners
    return None


def _correct_court_points(corners_4):
    """Enforce horizontal court lines by averaging y-coords of left/right pairs.

    corners_4: [bl, br, tl, tr] in image space.
    Returns: corrected [bl, br, tl, tr] with horizontal baselines.
    """
    pts = np.array(corners_4, dtype=np.float64)
    # Average y of top pair (tl, tr) and bottom pair (bl, br)
    tl_y = tr_y = round((pts[2][1] + pts[3][1]) / 2)
    bl_y = br_y = round((pts[0][1] + pts[1][1]) / 2)
    pts[0][1] = bl_y
    pts[1][1] = br_y
    pts[2][1] = tl_y
    pts[3][1] = tr_y
    return [[int(x), int(y)] for x, y in pts]


def compute_homography(image_corners):
    """Compute homography mapping image pixels → court metres.

    Uses geometric validation (aspect ratio + MSE against reference) instead
    of reprojection error, which is always 0 for 4-point homographies.

    image_corners: list of 4 points [bl, br, tl, tr] in image space
    Returns: (H, valid)
    """
    src = np.array(image_corners[:4], dtype=np.float64)
    dst = np.array([
        COURT_MODEL["outer_bl"], COURT_MODEL["outer_br"],
        COURT_MODEL["outer_tl"], COURT_MODEL["outer_tr"],
    ], dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None, False

    # Geometric validation: check aspect ratio of detected court polygon
    valid = _validate_court_geometry(src)

    # Validate that projected corner coordinates are within court bounds
    if valid:
        for corner in src:
            cx, cy = image_to_court(H, corner)
            if cx < -1 or cx > COURT_LENGTH + 1 or cy < -1 or cy > COURT_WIDTH + 1:
                valid = False
                break

    return H, valid


def _validate_court_geometry(corners_px):
    """Validate detected court corners form a reasonable quadrilateral.

    In broadcast views, perspective foreshortening makes the physical 2.587:1
    ratio invisible in pixel space. We check for degenerate cases instead:
    1. Minimum area (not a point or line)
    2. Convex quadrilateral (no crossing edges)
    """
    pts = np.array(corners_px, dtype=np.float64)

    # Traverse boundary in order: bl → br → tr → tl (clockwise)
    bl, br, tl, tr = pts
    boundary = [bl, br, tr, tl]

    area = cv2.contourArea(np.array(boundary, dtype=np.float32).reshape(-1, 1, 2))
    if area < 1000:
        return False

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [cross(boundary[i], boundary[(i + 1) % 4], boundary[(i + 2) % 4]) for i in range(4)]
    if not all(s > 0 for s in signs) and not all(s < 0 for s in signs):
        return False

    return True


def court_geometry_reliable(corners_px, max_trapezoid_ratio=None):
    """Check if the detected court corners form a true trapezoid (telecast/broadcast
    perspective) vs a rectangle (degenerate — camera looking straight down or
    corner detection fallback).

    A valid perspective view has a narrower top edge than bottom edge.
    Ratio = top_width / bottom_width.  When ratio > max_trapezoid_ratio,
    the quadrilateral is too close to rectangular to trust homography-based
    court-space measurements.

    Returns True if reliable (ratio <= threshold), False if degenerate.
    """
    from app.config.settings import settings
    if max_trapezoid_ratio is None:
        max_trapezoid_ratio = settings.geometry_max_trapezoid_ratio
    if corners_px is None or len(corners_px) < 4:
        return False
    pts = np.array(corners_px, dtype=np.float64)
    bl, br, tl, tr = pts[0], pts[1], pts[2], pts[3]
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    if bottom_width < 1.0:
        return False
    ratio = top_width / bottom_width
    return ratio <= max_trapezoid_ratio


def image_to_court(H, uv):
    """Project a single image point (u, v) to court metres (x, y)."""
    pt = np.array([[uv]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def clamp_to_court(x: float, y: float) -> tuple[float, float]:
    """Clamp court-space metres to valid court dimensions."""
    return max(0.0, min(COURT_LENGTH, x)), max(0.0, min(COURT_WIDTH, y))


def clamp_to_unit(x: float, y: float) -> tuple[float, float]:
    """Clamp unit-space ([0,1]) coordinates — used after dividing by court dims."""
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def court_to_unit(cx: float, cy: float,
                  court_length: float = COURT_LENGTH,
                  court_width: float = COURT_WIDTH) -> tuple[float, float]:
    """Convert court metres to unit [0,1], clamped."""
    ux = max(0.0, min(1.0, cx / court_length if court_length > 0 else 0))
    uy = max(0.0, min(1.0, cy / court_width if court_width > 0 else 0))
    return ux, uy


class HomographySmoother:
    """Smooths the FOUR outer court corners (in image space) over time,
    then recomputes H from the smoothed corners. Robust to per-frame flicker."""
    def __init__(self, alpha=0.6, win=5):
        self.alpha = alpha
        self.win = win
        self.buf = deque(maxlen=win)
        self.last_valid_H = None

    def update(self, corners_pixel, H_raw, valid):
        corners = np.array(corners_pixel, dtype=np.float64) if corners_pixel is not None else None

        if corners is None or not valid:
            if self.last_valid_H is not None:
                return self.last_valid_H, False
            return None, False

        self.buf.append(corners)

        if len(self.buf) == 1:
            smoothed_corners = corners
        else:
            med = np.median(np.stack(self.buf), axis=0)
            smoothed_corners = self.alpha * med + (1 - self.alpha) * corners

        dst = np.array([
            COURT_MODEL["outer_bl"], COURT_MODEL["outer_br"],
            COURT_MODEL["outer_tl"], COURT_MODEL["outer_tr"],
        ], dtype=np.float64)
        H_smooth, _ = cv2.findHomography(smoothed_corners, dst, cv2.RANSAC, 5.0)

        if H_smooth is not None:
            self.last_valid_H = H_smooth
            return H_smooth, True
        elif self.last_valid_H is not None:
            return self.last_valid_H, False
        return None, False


def make_undistorter(K, dist, size):
    """Create undistortion function from camera intrinsics.
    If camera not calibrated, return identity (no-op)."""
    if K is None or dist is None:
        return lambda frame: frame

    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha=0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_16SC2)

    def undistort(frame):
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    return undistort


class CourtNormalizer:
    """Unified court coordinate normalizer.

    Converts image-pixel positions (shuttle, player bbox, keypoints) to
    court-space metres via homography.  Optionally mirrors far-side
    coordinates so both players share a common (near-side) reference frame.

    Usage:
        normalizer = CourtNormalizer(H, court_length=13.4, court_width=6.1)
        cx, cy = normalizer.image_to_court((px, py))
        cx, cy = normalizer.normalize_player_position(bbox)
        mx, my = normalizer.mirror_far_side_if_needed((cx, cy), "far")
    """

    def __init__(self, homography_matrix: np.ndarray | None,
                 court_length: float = COURT_LENGTH,
                 court_width: float = COURT_WIDTH):
        self.H = homography_matrix
        self.court_length = court_length
        self.court_width = court_width

    def image_to_court(self, point_px: tuple[float, float]) -> tuple[float, float] | None:
        """Convert a single image-pixel point (u, v) to court metres (x, y).

        Returns None if no valid homography.
        """
        if self.H is None:
            return None
        try:
            return image_to_court(self.H, point_px)
        except Exception:
            return None

    def normalize_player_position(self, player_bbox: list[float]) -> tuple[float, float] | None:
        """Convert a player detection bbox [x1, y1, x2, y2] to court-space
        foot position (metres).  Uses bbox bottom-centre as the foot point."""
        if not player_bbox or len(player_bbox) < 4:
            return None
        foot_px = foot_point_from_bbox(player_bbox)
        return self.image_to_court(foot_px)

    def mirror_far_side_if_needed(self, point_court: tuple[float, float],
                                   player_side: str) -> tuple[float, float]:
        """Mirror court-space x for the far-side player so both players
        share a common near-side reference frame.

        In the un-mirrored court frame, x increases away from the camera
        (near → far).  After mirroring, x increases left-to-right in the
        viewer's perspective for *both* players.

        Returns (mirrored_x, y) — unchanged if player_side == "near".
        """
        x, y = point_court
        if player_side == "far":
            x = self.court_length - x
        return (x, y)

    def foot_from_pose(self, keypoints_xy: np.ndarray,
                       conf: np.ndarray | None = None,
                       conf_thr: float = 0.3) -> tuple[float, float] | None:
        """Court-space foot position from COCO-17 ankle keypoints.

        Returns (x_court, y_court) or None when both ankles are low-confidence.
        """
        foot_px = foot_midpoint_from_pose(keypoints_xy, conf, conf_thr)
        if foot_px is None:
            return None
        return self.image_to_court(foot_px)

    def to_unit(self, cx: float, cy: float) -> tuple[float, float]:
        """Court metres → unit [0, 1], clamped."""
        return court_to_unit(cx, cy, self.court_length, self.court_width)

    def clamp(self, cx: float, cy: float) -> tuple[float, float]:
        """Clamp to valid court dimensions."""
        return clamp_to_court(cx, cy)


def foot_midpoint_from_pose(keypoints_xy, conf=None, conf_thr=0.3):
    """COCO-17 ankles are indices 15 (left) and 16 (right).
    Returns (u, v) midpoint of ankles, or None if both low-confidence."""
    L_ANKLE, R_ANKLE = 15, 16
    pts = []
    for i in (L_ANKLE, R_ANKLE):
        if conf is None or conf[i] >= conf_thr:
            pts.append(keypoints_xy[i])
    if not pts:
        return None
    pts = np.array(pts, dtype=np.float64)
    return tuple(pts.mean(axis=0))


def foot_point_from_bbox(bbox_xyxy):
    """Fallback when pose is unavailable: bottom-center of the player box."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, float(y2))