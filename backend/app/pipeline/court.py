import numpy as np
import cv2
from collections import deque
from pathlib import Path

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings
from app.pipeline.shared.court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT,
    _detect_court_color_line, _correct_court_points,
    _validate_court_geometry, compute_homography, image_to_court,
    HomographySmoother,
)


def _manual_corners_sane(corners):
    """Basic sanity for user-provided (manual) corners.

    Unlike auto-detected corners, manual corners are deliberate input and must
    NOT be rejected by the trapezoid-reliability gate (a near-rectangular
    perspective from straight-on phone footage would otherwise be discarded).
    We only require a non-degenerate, convex quadrilateral with sufficient area.
    """
    if corners is None or len(corners) != 4:
        return False
    pts = np.array(corners, dtype=np.float64)
    bl, br, tl, tr = pts
    boundary = [bl, br, tr, tl]
    area = cv2.contourArea(np.array(boundary, dtype=np.float32).reshape(-1, 1, 2))
    if area < 1000:
        return False

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [_cross(boundary[i], boundary[(i + 1) % 4], boundary[(i + 2) % 4]) for i in range(4)]
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


# ─── Court Keypoint Detector (court_kpRCNN) ────────────────────────────────

class CourtKeypointDetector:
    """Detects 6 court keypoints using a fine-tuned torchvision KeypointRCNN.

    The model outputs 6 keypoints in this order:
      0: far-left corner   (court metres: 0, 0)
      1: far-right corner  (court metres: 0, 6.10)
      2: net-left          (court metres: 6.7, 0)   — unreliable at broadcast angles
      3: net-right         (court metres: 6.7, 6.10)
      4: near-left corner  (court metres: 13.4, 0)
      5: near-right corner (court metres: 13.4, 6.10)

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

        # Per-keypoint validation: near corners must be at bottom, far corners at top
        h = frame.shape[0]
        mid_y = h / 2
        # KP4 (near-left) and KP5 (near-right) must be in bottom half
        if points[4][1] < mid_y or points[5][1] < mid_y:
            return None
        # KP0 (far-left) and KP1 (far-right) must be in top half
        if points[0][1] > mid_y or points[1][1] > mid_y:
            return None

        return points

    def detect_corners(self, frame: np.ndarray) -> list[list[int]] | None:
        """Detect 4 outer court corners: [bl, br, tl, tr]."""
        kps = self.detect(frame)
        if kps is None:
            return None
        return [kps[4], kps[5], kps[0], kps[1]]

    def detect_with_fallback(self, frame: np.ndarray) -> list[list[int]]:
        """Detect corners with fallback chain: model → color+line → proportional."""
        corners = self.detect_corners(frame)
        if _corners_are_valid(corners):
            return corners

        corners = _detect_court_color_line(frame)
        if _corners_are_valid(corners):
            return corners

        h, w = frame.shape[:2]
        mx = int(w * settings.court_corner_margin_x)
        return [
            (mx, int(h * settings.court_corner_bottom_y)), (w - mx, int(h * settings.court_corner_bottom_y)),  # bl, br
            (mx, int(h * settings.court_corner_top_y)), (w - mx, int(h * settings.court_corner_top_y)),  # tl, tr
        ]


class CourtDetectionStage:
    name = "court_detection"
    input_keys = []
    output_keys = ["court"]

    def run(self, artifacts: ArtifactStore, config: StageConfig,
            corners: list[tuple[int, int]] | None = None,
            frame: np.ndarray | None = None,
            is_manual: bool = False,
            manual_corners_fallback: list[tuple[int, int]] | None = None) -> StageResult:
        H, valid = None, False
        corrected = None
        detection_method = "auto"

        if corners is None and frame is not None:
            detector = CourtKeypointDetector(settings.court_kpRCNN_model_path, device=settings.device)
            corners = detector.detect_with_fallback(frame)

        if corners is not None and len(corners) == 4:
            corrected = _correct_court_points(corners)
            H, valid = compute_homography(corrected)
            # Manual corners are deliberate user input: trust them even if they
            # fail the trapezoid-reliability gate (that gate exists to catch
            # auto-detection hallucinations, not user clicks). Keep only the
            # basic non-degenerate / convex sanity check.
            if is_manual and not valid and H is not None and _manual_corners_sane(corrected):
                valid = True

        # When auto-detection produced an invalid/degenerate court, prefer
        # user-supplied manual corners (if available) over the proportional
        # fallback. Manual corners bypass the trapezoid-reliability gate, so
        # they yield a usable (valid) homography for court-space features.
        if not valid and not is_manual and manual_corners_fallback is not None \
                and len(manual_corners_fallback) == 4:
            fb_corrected = _correct_court_points(manual_corners_fallback)
            fb_H, fb_valid = compute_homography(fb_corrected)
            if fb_H is not None and _manual_corners_sane(fb_corrected):
                corrected = fb_corrected
                H, valid = fb_H, True
                is_manual = True
                detection_method = "manual_fallback"
                logger.info("Auto court detection invalid; using manual corners fallback")

        # Fallback to proportional corners only when NOT manually set.
        # User-provided manual corners are preserved even if homography fails,
        # so the caller can still use them for downstream validation/override.
        if not valid and frame is not None and not is_manual:
            h, w = frame.shape[:2]
            mx = int(w * settings.court_corner_margin_x)
            corners = [
                (mx, int(h * settings.court_corner_bottom_y)), (w - mx, int(h * settings.court_corner_bottom_y)),
                (mx, int(h * settings.court_corner_top_y)), (w - mx, int(h * settings.court_corner_top_y)),
            ]
            corrected = _correct_court_points(corners)
            H, valid = compute_homography(corrected)
            valid = False

        if corrected is None or len(corrected) != 4:
            return StageResult.from_error("Court corners are required (4 points). Provide frame or manual corners.")

        # Temporal smoothing
        smoother = HomographySmoother(alpha=0.6, win=5)
        H_smooth, valid_smooth = smoother.update(corrected, H, valid)

        court_data = {
            "homography": (H_smooth if H_smooth is not None else H).tolist() if H_smooth is not None or H is not None else None,
            "corners_pixel": [list(c) for c in corrected],
            "court_length": COURT_LENGTH,
            "court_width": COURT_WIDTH,
            "net_height": NET_HEIGHT,
            "valid": valid_smooth,
            "detection_method": detection_method if detection_method == "manual_fallback" else ("manual" if is_manual else detection_method),
            "has_manual_corners": is_manual,
        }

        artifacts.set("court", court_data)

        return StageResult.success(
            artifacts={"court": artifacts.path("court")},
            metadata={"homography_computed": True, "valid": valid_smooth, "manual": is_manual}
        )


def _corners_are_valid(corners: list[list[int]] | list[tuple[int, int]] | None) -> bool:
    if corners is None or len(corners) != 4:
        return False
    corrected = _correct_court_points(corners)
    _, valid = compute_homography(corrected)
    return bool(valid)
