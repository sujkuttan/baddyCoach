"""
Canonical BST preprocessing functions shared by backend and colab pipelines.
"""

import numpy as np
import pandas as pd

from app.pipeline.shared.court import image_to_court, COURT_LENGTH, COURT_WIDTH

BONE_PAIRS = [
    (0,1),(0,2),(1,2),(1,3),(2,4),
    (3,5),(4,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def normalize_joints(coords: np.ndarray, det_bbox: tuple | None = None,
                     bbox_margin: float = 0.0,
                     conf: np.ndarray | None = None,
                     min_confidence: float = 0.35) -> np.ndarray:
    """Normalize joints using bbox diagonal with center_align.

    Matches the official BST preprocessing:
    - Origin = top-left of the player bounding box
    - Scale = diagonal distance of the bounding box
    - center_align=True shifts origin to bbox center

    Args:
        coords: (17, 2) keypoints in pixel coords
        det_bbox: optional (x1, y1, x2, y2) detection bbox.
                  If None, uses keypoint bbox (coords min/max of valid keypoints).
        bbox_margin: fraction to expand bbox on all sides (e.g., 0.15 = 15%).
                     Applied after deriving bbox_min/bbox_max from either source.
                     Compensates for keypoint bboxes being tighter than detection bboxes.
        conf: optional (17,) keypoint confidence scores (0-1). Used only when
              det_bbox is None to mask low-confidence and zero-coordinate keypoints
              from the keypoint-bbox computation, preventing spurious outliers from
              compressing the real skeleton.

    Returns:
        (17, 2) normalized joints, range roughly [-0.X, 0.X]
    """
    coords = np.asarray(coords, dtype=np.float64)
    invalid_mask = ~np.isfinite(coords).all(axis=1) | np.all(coords == 0.0, axis=1)
    if conf is not None:
        invalid_mask |= np.asarray(conf) < min_confidence

    if det_bbox is not None:
        bbox_min = np.array([det_bbox[0], det_bbox[1]], dtype=np.float64)
        bbox_max = np.array([det_bbox[2], det_bbox[3]], dtype=np.float64)
    else:
        mask = ~invalid_mask
        if mask.any():
            bbox_min = coords[mask].min(axis=0)
            bbox_max = coords[mask].max(axis=0)
        else:
            bbox_min = np.zeros(2, dtype=np.float64)
            bbox_max = np.ones(2, dtype=np.float64)

    if bbox_margin > 0:
        margin = (bbox_max - bbox_min) * bbox_margin
        bbox_min -= margin
        bbox_max += margin

    diag = np.linalg.norm(bbox_max - bbox_min)
    if diag < 1e-6:
        diag = 1.0

    normalized = (coords - bbox_min) / diag
    center = (bbox_min + bbox_max) / 2.0
    normalized -= (center - bbox_min) / diag

    normalized[invalid_mask] = 0.0

    return normalized.astype(np.float32)


def normalize_joints_batched(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True,
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance (batched).

    Args:
        arr: (N, 17, 2) keypoints in pixel coords
        bbox: (N, 4) bounding boxes (x1, y1, x2, y2)
        center_align: whether to center-align (default True)

    Returns:
        (N, 17, 2) normalized joints
    """
    diag = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    diag = np.where(diag == 0, 1, diag)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / diag, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / diag, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / diag
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


def normalize_joints_court(
    coords: np.ndarray,
    homography: np.ndarray,
    court_length: float = COURT_LENGTH,
    court_width: float = COURT_WIDTH,
) -> np.ndarray:
    """Normalize joints in court-space via homography.

    Preserves absolute position while providing scale-invariant
    joint positions matched to the fixed court dimensions.

    Args:
        coords: (17, 2) keypoints in pixel coords.
        homography: 3x3 homography matrix (pixel → court).
        court_length: court length in meters (default 13.4).
        court_width: court width in meters (default 6.1).

    Returns:
        (17, 2) normalized joints, range [-0.5, 0.5].
    """
    court_coords = np.array([
        image_to_court(homography, (float(x), float(y)))
        for x, y in coords
    ])
    normalized = court_coords / np.array([court_length, court_width])
    return (normalized - 0.5).astype(np.float32)


def normalize_joints_hip_centered(
    coords: np.ndarray,
    vid_w: float = 1.0,
    vid_h: float = 1.0,
    conf: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize joints by centering on hip midpoint and scaling by torso length.

    From Ryan-z-Feng-ccsf/badminton-coach: hip midpoint is more stable than
    bbox center (doesn't shift when arms raise), and torso length provides
    a person-specific scale that's invariant to camera distance.

    COCO-17 indices used:
    - shoulders: 5 (left), 6 (right)
    - hips: 11 (left), 12 (right)

    Args:
        coords: (17, 2) keypoints in pixel coords.
        vid_w: video width in pixels (for aspect ratio correction).
        vid_h: video height in pixels (for aspect ratio correction).
        conf: optional (17,) keypoint confidence; low-conf joints are
              not used for hip/shoulder center computation.

    Returns:
        (17, 2) normalized joints, range roughly [-1, 1].
    """
    arr = coords.copy().astype(np.float64)

    # Default mask: use all non-zero coords
    mask = np.ones(len(arr), dtype=bool)
    if conf is not None:
        mask &= conf > 0.1
    mask &= ~np.all(arr == 0.0, axis=1)

    # Hip center
    if mask[11] and mask[12]:
        hip_center = (arr[11] + arr[12]) * 0.5
    elif mask[11]:
        hip_center = arr[11]
    elif mask[12]:
        hip_center = arr[12]
    else:
        hip_center = np.array([0.0, 0.0])

    # Shoulder center
    if mask[5] and mask[6]:
        shoulder_center = (arr[5] + arr[6]) * 0.5
    elif mask[5]:
        shoulder_center = arr[5]
    elif mask[6]:
        shoulder_center = arr[6]
    else:
        shoulder_center = hip_center + np.array([0.0, -100.0])

    torso_length = float(np.linalg.norm(shoulder_center - hip_center))
    if torso_length < 1e-6:
        torso_length = 1.0

    normalized = (arr - hip_center) / torso_length

    # Aspect ratio correction (x is wider in landscape)
    if vid_h > 0:
        normalized[:, 0] *= (vid_w / vid_h)

    normalized[~mask] = 0.0

    return normalized.astype(np.float32)


def create_bones(joints: np.ndarray, velocity_mag: np.ndarray | None = None,
                 amp_factor: float = 0.0) -> np.ndarray:
    """Create bone vectors from joint positions, optionally velocity-weighted.

    When velocity_mag is provided and amp_factor > 0, bone vectors are amplified
    by the motion of their endpoint joints — fast-moving limbs produce larger
    bone values, giving the model temporal motion signal even from single-frame
    features.

    Args:
        joints: (17, 2) single-frame joints (normalized).
        velocity_mag: (17,) per-joint velocity magnitude, or None.
        amp_factor: amplification factor for velocity weighting (0 = disabled).

    Returns:
        (19, 2) bone vectors
    """
    use_velocity = velocity_mag is not None and amp_factor > 0
    bones = []
    for s, e in BONE_PAIRS:
        sj, ej = joints[s], joints[e]
        if np.any(sj != 0) and np.any(ej != 0):
            bone = ej - sj
            if use_velocity:
                avg_motion = (velocity_mag[s] + velocity_mag[e]) * 0.5
                bone = bone * (1.0 + avg_motion * amp_factor)
            bones.append(bone)
        else:
            bones.append(np.zeros(2, dtype=np.float32))
    return np.array(bones, dtype=np.float32)


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    return arr / np.array([v_width, v_height])
