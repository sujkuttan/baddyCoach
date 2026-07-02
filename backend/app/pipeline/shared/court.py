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
    """Legacy wrapper for the Hough-line trapezoid court detector."""
    return detect_court_hough_lines(frame)


def detect_court_hough_lines(frame: np.ndarray, use_cuda: bool = True):
    """Detect the outer court as a true trapezoid from Hough line intersections."""
    if frame is None or frame.size == 0:
        return None

    line_mask = _court_line_mask(frame, use_cuda=use_cuda)
    h, w = frame.shape[:2]
    min_len = max(60, int(min(h, w) * 0.16))
    lines = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_len,
        maxLineGap=max(20, int(min(h, w) * 0.04)),
    )
    if lines is None:
        return None

    candidates = [_line_candidate(line[0]) for line in lines]
    candidates = [c for c in candidates if c is not None]
    if len(candidates) < 4:
        return None

    horizontal = [c for c in candidates if c["kind"] == "horizontal"]
    side_left = [c for c in candidates if c["kind"] == "side_left"]
    side_right = [c for c in candidates if c["kind"] == "side_right"]
    if len(horizontal) < 2 or not side_left or not side_right:
        return None

    top_line = _fit_boundary_line(_select_horizontal_boundary(horizontal, top=True))
    bottom_line = _fit_boundary_line(_select_horizontal_boundary(horizontal, top=False))
    left_line = _fit_boundary_line(_select_side_boundary(side_left, frame.shape, left=True))
    right_line = _fit_boundary_line(_select_side_boundary(side_right, frame.shape, left=False))
    if any(line is None for line in [top_line, bottom_line, left_line, right_line]):
        return None

    tl = _intersect_lines(top_line, left_line)
    tr = _intersect_lines(top_line, right_line)
    bl = _intersect_lines(bottom_line, left_line)
    br = _intersect_lines(bottom_line, right_line)
    if any(pt is None for pt in [bl, br, tl, tr]):
        return None

    corners = [bl, br, tl, tr]
    if not _corners_within_frame(corners, w, h):
        return None
    corners = [[int(round(x)), int(round(y))] for x, y in corners]
    if not _validate_court_geometry(corners):
        return None
    H, valid = compute_homography(corners)
    if H is None or not valid:
        return None
    return corners


def _court_line_mask(frame: np.ndarray, use_cuda: bool = True) -> np.ndarray:
    mask = None
    if use_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                rgb = torch.as_tensor(frame[:, :, ::-1].copy(), device="cuda", dtype=torch.float32) / 255.0
                maxc, _ = rgb.max(dim=2)
                minc, _ = rgb.min(dim=2)
                sat = (maxc - minc) / torch.clamp(maxc, min=1e-6)
                white = (maxc > 0.55) & (sat < 0.35)
                mask = white.detach().cpu().numpy().astype(np.uint8) * 255
        except Exception:
            pass

    if mask is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 95, 255]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.inRange(gray, 130, 255)
        mask = cv2.bitwise_and(white, bright)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(mask, 40, 120)
    return cv2.dilate(edges, kernel, iterations=1)


def _line_candidate(line):
    x1, y1, x2, y2 = [float(v) for v in line]
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1.0:
        return None
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    if angle > 90:
        angle = 180 - angle
    if angle <= 18:
        kind = "horizontal"
    elif 35 <= angle <= 82 and abs(dx) > 8:
        kind = "side_right" if (dy / dx) > 0 else "side_left"
    else:
        return None
    return {
        "points": np.array([[x1, y1], [x2, y2]], dtype=np.float64),
        "mid": np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64),
        "length": length,
        "kind": kind,
    }


def _select_horizontal_boundary(candidates, top: bool):
    candidates = sorted(candidates, key=lambda c: c["mid"][1])
    n = max(1, min(4, len(candidates) // 2))
    return candidates[:n] if top else candidates[-n:]


def _select_side_boundary(candidates, frame_shape, left: bool):
    h, w = frame_shape[:2]
    y_ref = h * 0.55

    def x_at_ref(candidate):
        pts = candidate["points"]
        line = _fit_line_from_points(pts)
        if line is None or abs(line[0]) < 1e-6:
            return candidate["mid"][0]
        vx, vy, x0, y0 = line
        return x0 + (y_ref - y0) * (vx / vy) if abs(vy) > 1e-6 else candidate["mid"][0]

    candidates = sorted(candidates, key=x_at_ref)
    n = max(1, min(4, len(candidates) // 2))
    return candidates[:n] if left else candidates[-n:]


def _fit_boundary_line(candidates):
    if not candidates:
        return None
    points = np.vstack([c["points"] for c in candidates])
    return _fit_line_from_points(points)


def _fit_line_from_points(points):
    if points is None or len(points) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)


def _intersect_lines(line_a, line_b):
    vx1, vy1, x1, y1 = line_a
    vx2, vy2, x2, y2 = line_b
    a = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    det = np.linalg.det(a)
    if abs(det) < 1e-6:
        return None
    t, _ = np.linalg.solve(a, b)
    return [x1 + t * vx1, y1 + t * vy1]


def _corners_within_frame(corners, width: int, height: int, margin_ratio: float = 0.08) -> bool:
    margin = max(width, height) * margin_ratio
    for x, y in corners:
        if x < -margin or x > width + margin or y < -margin or y > height + margin:
            return False
    return True


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


def _validate_court_geometry(corners_px, max_trapezoid_ratio=None):
    """Validate detected court corners form a reasonable quadrilateral.

    Checks:
    1. Minimum area (not a point or line)
    2. Convex quadrilateral (no crossing edges)
    3. True trapezoid (top edge narrower than bottom edge), rejecting
       rectangles which produce degenerate homographies.
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

    return bool(court_geometry_reliable(corners_px, max_trapezoid_ratio))


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
    return bool(ratio <= max_trapezoid_ratio)


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
