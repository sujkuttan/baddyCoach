"""
Canonical BST preprocessing functions shared by backend and colab pipelines.
"""

import numpy as np
import pandas as pd

BONE_PAIRS = [
    (0,1),(0,2),(1,2),(1,3),(2,4),
    (3,5),(4,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def normalize_joints(coords: np.ndarray, det_bbox: tuple | None = None) -> np.ndarray:
    """Normalize joints using bbox diagonal with center_align.

    Matches the official BST preprocessing:
    - Origin = top-left of the player bounding box
    - Scale = diagonal distance of the bounding box
    - center_align=True shifts origin to bbox center

    Args:
        coords: (17, 2) keypoints in pixel coords
        det_bbox: optional (x1, y1, x2, y2) detection bbox for stable normalization.
                  If None, falls back to keypoint bbox (less stable).

    Returns:
        (17, 2) normalized joints, range roughly [-0.X, 0.X]
    """
    if det_bbox is not None:
        bbox_min = np.array([det_bbox[0], det_bbox[1]], dtype=np.float64)
        bbox_max = np.array([det_bbox[2], det_bbox[3]], dtype=np.float64)
    else:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)

    diag = np.linalg.norm(bbox_max - bbox_min)
    if diag < 1e-6:
        diag = 1.0

    normalized = (coords - bbox_min) / diag
    center = (bbox_min + bbox_max) / 2.0
    normalized -= (center - bbox_min) / diag
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


def create_bones(joints: np.ndarray) -> np.ndarray:
    """Create bone vectors from joint positions.

    Args:
        joints: (17, 2) single-frame joints

    Returns:
        (19, 2) bone vectors
    """
    bones = []
    for s, e in BONE_PAIRS:
        sj, ej = joints[s], joints[e]
        bones.append(ej - sj if np.any(sj != 0) and np.any(ej != 0) else np.zeros(2, dtype=np.float32))
    return np.array(bones, dtype=np.float32)


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    return arr / np.array([v_width, v_height])
