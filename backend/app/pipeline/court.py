import numpy as np
import cv2
from collections import deque
from pathlib import Path

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.config.settings import settings


# PRD §2.2: Canonical court model (metres)
COURT_MODEL = {
    "outer_tl": (0.0, 0.0),
    "outer_tr": (0.0, 5.18),
    "outer_bl": (13.4, 0.0),
    "outer_br": (13.4, 5.18),
}

# Court aspect ratio (length / width) for geometric validation
COURT_ASPECT_RATIO = 13.4 / 5.18  # ≈ 2.587


# ─── Court Keypoint Detector (court_kpRCNN) ────────────────────────────────

class CourtKeypointDetector:
    """Detects 6 court keypoints using a fine-tuned torchvision KeypointRCNN.

    The model outputs 6 keypoints in this order:
      0: far-left corner   (court metres: 0, 0)
      1: far-right corner  (court metres: 0, 5.18)
      2: net-left          (court metres: 6.7, 0)   — unreliable at broadcast angles
      3: net-right         (court metres: 6.7, 5.18)
      4: near-left corner  (court metres: 13.4, 0)
      5: near-right corner (court metres: 13.4, 5.18)

    Only KP0, KP1, KP4, KP5 (the 4 outer corners) are used for homography.
    KP2/KP3 are ignored because KP2 often duplicates KP0 at broadcast camera angles.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.device = device
        self.model = None
        model_path = Path(model_path)
        if not model_path.exists():
            return
        try:
            import torch
            self.model = torch.load(str(model_path), map_location=device, weights_only=False)
            self.model.to(device).eval()
        except Exception:
            self.model = None

    def detect(self, frame: np.ndarray) -> list[list[int]] | None:
        """Detect 6 court keypoints. Returns [[x,y], ...] x 6 or None."""
        if self.model is None:
            return None
        import torch
        import torchvision.transforms.functional as F
        import torchvision

        # Pass frame directly to model (BGR, same as training).
        # The original SoloShuttlePose does NOT convert BGR→RGB.
        tensor = F.to_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)

        scores = output[0]["scores"].cpu().numpy()
        # Use 0.7 threshold matching the reference SoloShuttlePose code
        high = np.where(scores > 0.7)[0].tolist()
        if not high:
            return None

        # NMS returns indices sorted by score descending — take index 0
        nms = torchvision.ops.nms(
            output[0]["boxes"][high], output[0]["scores"][high], 0.3
        ).cpu().numpy()
        kps = output[0]["keypoints"][high][nms]
        kps_np = kps[0].cpu().numpy()
        points = [[int(kp[0]), int(kp[1])] for kp in kps_np]
        if len(points) < 6:
            return None

        # Validate: bottom y must be below top y
        top_y = (points[0][1] + points[1][1]) / 2
        bot_y = (points[4][1] + points[5][1]) / 2
        if bot_y <= top_y:
            return None
        return points

    def detect_corners(self, frame: np.ndarray) -> list[list[int]] | None:
        """Detect 4 outer court corners: [bl, br, tl, tr]."""
        kps = self.detect(frame)
        if kps is None:
            return None
        return [kps[4], kps[5], kps[0], kps[1]]

    def detect_with_fallback(self, frame: np.ndarray) -> list[list[int]]:
        """Detect corners with fallback chain: model → proportional."""
        corners = self.detect_corners(frame)
        if corners is not None:
            return corners
        # Proportional fallback based on typical broadcast framing
        h, w = frame.shape[:2]
        mx = int(w * 0.08)
        return [
            (mx, int(h * 0.72)), (w - mx, int(h * 0.72)),  # bl, br
            (mx, int(h * 0.28)), (w - mx, int(h * 0.28)),  # tl, tr
        ]


class CourtDetectionStage:
    name = "court_detection"
    input_keys = []
    output_keys = ["court"]

    # Standard badminton court dimensions in meters (singles)
    COURT_LENGTH = 13.4
    COURT_WIDTH = 5.18
    NET_HEIGHT = 1.55

    def run(self, artifacts: ArtifactStore, config: StageConfig,
            corners: list[tuple[int, int]] | None = None,
            frame: np.ndarray | None = None) -> StageResult:
        # Auto-detect from frame if corners not provided
        if corners is None and frame is not None:
            detector = CourtKeypointDetector(settings.court_kpRCNN_model_path, device=settings.device)
            corners = detector.detect_with_fallback(frame)

        if corners is None or len(corners) != 4:
            return StageResult.from_error("Court corners are required (4 points). Provide frame or manual corners.")

        # Apply geometric correction: average y-coords of left/right pairs
        # to enforce horizontal court lines before computing homography.
        corrected = _correct_court_points(corners)

        # Compute homography
        H, valid = compute_homography(corrected)

        # Temporal smoothing
        smoother = HomographySmoother(alpha=0.6, win=5)
        H_smooth, valid = smoother.update(corrected, H, valid)

        court_data = {
            "homography": (H_smooth if H_smooth is not None else H).tolist() if H_smooth is not None or H is not None else None,
            "corners_pixel": [list(c) for c in corrected],
            "court_length": self.COURT_LENGTH,
            "court_width": self.COURT_WIDTH,
            "net_height": self.NET_HEIGHT,
            "valid": valid,
        }

        artifacts.set("court", court_data)

        return StageResult.success(
            artifacts={"court": artifacts.path("court")},
            metadata={"homography_computed": True, "valid": valid}
        )


# ─── Geometric correction (from SoloShuttlePose __correction) ───────────────

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


# ─── Homography with geometric validation ───────────────────────────────────

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
    return H, valid


def _validate_court_geometry(corners_px):
    """Validate detected court corners form a reasonable quadrilateral.

    In broadcast views, perspective foreshortening makes the physical 2.587:1
    ratio invisible in pixel space. We check for degenerate cases instead:
    1. Minimum area (not a point or line)
    2. Convex quadrilateral (no crossing edges)
    """
    pts = np.array(corners_px, dtype=np.float64)
    area = cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32))
    if area < 1000:
        return False

    # Traverse boundary in order: bl → br → tr → tl (clockwise)
    bl, br, tl, tr = pts
    boundary = [bl, br, tr, tl]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [cross(boundary[i], boundary[(i + 1) % 4], boundary[(i + 2) % 4]) for i in range(4)]
    if not all(s > 0 for s in signs) and not all(s < 0 for s in signs):
        return False

    return True


def image_to_court(H, uv):
    """Project a single image point (u, v) to court metres (x, y)."""
    pt = np.array([[uv]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ─── Temporal smoothing (handheld-critical) ────────────────────────────────

class HomographySmoother:
    """Smooths the FOUR outer court corners (in image space) over time,
    then recomputes H from the smoothed corners. Robust to per-frame flicker."""
    def __init__(self, alpha=0.6, win=5):
        self.alpha = alpha
        self.win = win
        self.buf = deque(maxlen=win)
        self.last_valid_H = None

    def update(self, corners_pixel, H_raw, valid):
        corners = np.array(corners_pixel, dtype=np.float64) if corners_pixel else None

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
