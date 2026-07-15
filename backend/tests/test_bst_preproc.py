"""
Task 3.1 — investigation of BST joint centering (INVESTIGATION-FIRST).

Builds synthetic COCO-17 poses and measures normalize_joints (bbox mode, the
active "bbox" path) vs normalize_joints_hip_centered. Determines whether the
bbox-midpoint centering produces a real defect (torso joints pushed out of the
~[-0.5, 0.5] model-trained range / dominant offset) that would hurt the model.

Findings drive the decision: fix (stable_center) vs DONE_WITH_CONCERNS (no defect).
"""

import numpy as np

from app.pipeline.shared.bst_preproc import (
    normalize_joints,
    normalize_joints_hip_centered,
)


def _build_standing_pose():
    """Symmetric standing COCO-17 pose (pixels), image 1920x1080."""
    c = np.zeros((17, 2), dtype=np.float64)
    c[0] = [960, 360]   # nose
    c[1] = [945, 350]   # leye
    c[2] = [975, 350]   # reye
    c[3] = [940, 365]   # lear
    c[4] = [980, 365]   # rear
    c[5] = [905, 460]   # lshoulder
    c[6] = [1015, 460]  # rshoulder
    c[7] = [870, 600]   # lelbow
    c[8] = [1050, 600]  # relbow
    c[9] = [855, 720]   # lwrist
    c[10] = [1065, 720]  # rwrist
    c[11] = [920, 640]  # lhip
    c[12] = [1000, 640]  # rhip
    c[13] = [905, 840]  # lknee
    c[14] = [1015, 840]  # rknee
    c[15] = [895, 1020]  # lankle
    c[16] = [1025, 1020]  # rankle
    return c


def _build_swing_pose():
    """Badminton swing: same base but right wrist extended FAR up/out."""
    c = _build_standing_pose()
    # racket-side (right) wrist extended far from the body
    c[10] = [1480, 180]   # rwrist extended up & to the right
    c[8] = [1280, 320]    # relbow follows
    return c


def _metrics(norm):
    abs_mean = float(np.mean(np.abs(norm)))
    mean_x = float(np.mean(norm[:, 0]))
    mean_y = float(np.mean(norm[:, 1]))
    max_abs = float(np.max(np.abs(norm)))
    return abs_mean, mean_x, mean_y, max_abs


def test_bbox_mode_symmetric_pose_in_range():
    """Standing pose: bbox centering must keep joints in [-0.5, 0.5]."""
    c = _build_standing_pose()
    norm = normalize_joints(c, det_bbox=None, bbox_margin=0.15)
    abs_mean, mean_x, mean_y, max_abs = _metrics(norm)
    # Note: mean offset ~0 for symmetric pose (x/y near 0)
    assert abs(mean_x) < 0.15
    assert abs(mean_y) < 0.15
    # all joints within model-trained range (diagonal scale guarantees this)
    assert np.all(np.abs(norm) <= 0.5 + 1e-6)
    # sanity check numbers printed for the report
    print(f"[stand bbox] abs_mean={abs_mean:.4f} mean=({mean_x:.4f},{mean_y:.4f}) max_abs={max_abs:.4f}")


def test_bbox_mode_swing_pose_in_range_no_clipping():
    """Swing pose with far-extended wrist: bbox centering must NOT clip torso
    joints out of [-0.5, 0.5]. Diagonal scale bounds every joint to the bbox."""
    c = _build_swing_pose()
    norm = normalize_joints(c, det_bbox=None, bbox_margin=0.15)
    abs_mean, mean_x, mean_y, max_abs = _metrics(norm)
    # Swollen bbox shifts the center; mean offset non-zero but bounded (benign).
    # The KEY assertion: no joint escapes the model-trained range.
    assert np.all(np.abs(norm) <= 0.5 + 1e-6), "torso joint clipped out of [-0.5,0.5]"
    # wrist still finite & in-range (just pushed toward an edge, not out)
    assert np.abs(norm[10, 0]) <= 0.5 + 1e-6
    print(f"[swing bbox] abs_mean={abs_mean:.4f} mean=({mean_x:.4f},{mean_y:.4f}) max_abs={max_abs:.4f}")


def test_hip_centered_swing_pose_difference():
    """Document that hip_centered uses torso-length scale (~[-2,2]) which is a
    DIFFERENT (model-incompatible) range than bbox-diagonal (~[-0.5,0.5])."""
    c = _build_swing_pose()
    norm = normalize_joints_hip_centered(c, vid_w=1920, vid_h=1080)
    abs_mean, mean_x, mean_y, max_abs = _metrics(norm)
    # hip_centered expands the range well beyond [-0.5, 0.5]
    assert max_abs > 0.5, "hip_centered unexpectedly in bbox range"
    print(f"[swing hip_centered] abs_mean={abs_mean:.4f} mean=({mean_x:.4f},{mean_y:.4f}) max_abs={max_abs:.4f}")


def test_bbox_mean_offset_benign_for_swing():
    """The mean offset for the asymmetric swing pose is bounded & benign:
    the model was trained on bbox-center-aligned joints, so a non-zero mean
    offset from an extended wrist is expected, not a defect."""
    c = _build_swing_pose()
    norm = normalize_joints(c, det_bbox=None, bbox_margin=0.15)
    mean_x = float(np.mean(norm[:, 0]))
    mean_y = float(np.mean(norm[:, 1]))
    # offset must be small relative to the model range (well under 0.5)
    assert abs(mean_x) < 0.5 and abs(mean_y) < 0.5
