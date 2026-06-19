import numpy as np
import cv2
from collections import deque
from pathlib import Path

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


# PRD §2.2: Canonical court model (metres)
COURT_MODEL = {
    "outer_tl": (0.0, 0.0),
    "outer_tr": (0.0, 5.18),
    "outer_bl": (13.4, 0.0),
    "outer_br": (13.4, 5.18),
}


class CourtDetectionStage:
    name = "court_detection"
    input_keys = []
    output_keys = ["court"]

    # Standard badminton court dimensions in meters (singles)
    COURT_LENGTH = 13.4
    COURT_WIDTH = 5.18
    NET_HEIGHT = 1.55

    def run(self, artifacts: ArtifactStore, config: StageConfig, corners: list[tuple[int, int]] | None = None) -> StageResult:
        if corners is None or len(corners) != 4:
            return StageResult.from_error("Court corners are required (4 points). Provide via manual calibration.")

        # PRD §2.5: Compute homography with reprojection error
        H, reproj_err, n_used = compute_homography(corners)

        # PRD §2.6: Apply temporal smoothing
        smoother = HomographySmoother(alpha=0.6, win=5)
        H_smooth, valid = smoother.update(corners, H, reproj_err)

        court_data = {
            "homography": (H_smooth if H_smooth is not None else H).tolist() if H_smooth is not None or H is not None else None,
            "corners_pixel": [list(c) for c in corners],
            "court_length": self.COURT_LENGTH,
            "court_width": self.COURT_WIDTH,
            "net_height": self.NET_HEIGHT,
            "reproj_err_m": reproj_err,
            "valid": valid,
        }

        artifacts.set("court", court_data)

        return StageResult.success(
            artifacts={"court": artifacts.path("court")},
            metadata={"homography_computed": True, "reproj_err_m": reproj_err, "valid": valid}
        )


# ─── PRD §2.5: Per-frame homography with reprojection error ─────────────────

def compute_homography(image_corners, min_points=4):
    """Compute homography mapping image pixels → court metres with reprojection error.

    image_corners: list of 4 points [bl, br, tl, tr] in image space
    Returns: (H, reproj_err_m, n_used) or (None, inf, 0)
    """
    if len(image_corners) < min_points:
        return None, float("inf"), 0

    src = np.array(image_corners[:4], dtype=np.float64)
    dst = np.array([
        COURT_MODEL["outer_bl"], COURT_MODEL["outer_br"],
        COURT_MODEL["outer_tl"], COURT_MODEL["outer_tr"],
    ], dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None, float("inf"), 0

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_m = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return H, err_m, int(mask.sum()) if mask is not None else len(src)


def image_to_court(H, uv):
    """Project a single image point (u, v) to court metres (x, y)."""
    pt = np.array([[uv]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ─── PRD §2.6: Temporal smoothing (handheld-critical) ──────────────────────

ERR_GATE_M = 0.20  # reject frames with reproj error > 20cm


class HomographySmoother:
    """Smooths the FOUR outer court corners (in image space) over time,
    then recomputes H from the smoothed corners. Robust to per-frame flicker."""
    def __init__(self, alpha=0.6, win=5):
        self.alpha = alpha
        self.win = win
        self.buf = deque(maxlen=win)
        self.last_valid_H = None

    def update(self, corners_pixel, H_raw, reproj_err):
        corners = np.array(corners_pixel, dtype=np.float64) if corners_pixel else None

        if corners is None or reproj_err > ERR_GATE_M:
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


# ─── PRD §2.3: Undistortion (optional, requires camera calibration) ────────

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


# ─── PRD §2.7: Player foot point ───────────────────────────────────────────

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
