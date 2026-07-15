"""BSTInputValidator — pre-inference validation for BST clip tensors.

Checks all 10 conditions that must match the official BST training pipeline
before every inference call:

  1. p0 = far player, p1 = near player
  2. Sequence length matches training seq_len
  3. Joint order matches training (COCO-17)
  4. Bone edges match JnB_bone configuration
  5. Shuttle tensor normalized by video width/height (range [0, 1])
  6. Joints are bbox-relative (not court-relative) unless court-norm configured
  7. Joints are center-aligned (mean ≈ 0)
  8. Player positions are court-normalized (range [0, 1])
  9. Missing joints are imputed consistently (no silent zero-fill)
  10. Hit frame is aligned as training expects (frame 0 = contact point)

Uses configured strictness level:
  - "warn" (default): log warnings, continue inference
  - "error": raise ValidationError on critical failures
  - "off": skip all validation
"""

import logging
import numpy as np

from dataclasses import dataclass, field

from app.pipeline.shared.bst_preproc import BONE_PAIRS

logger = logging.getLogger("bst_validator")


# ── Source code locations for each check ─────────────────────────────
# Every warning/error message includes the file:line where the
# problematic data originates, so fixes target the right code path.

_SRC = {
    # Player ordering
    "player_order_set": "backend/app/pipeline/strokes.py:175  player_order = [far_pid, near_pid]",
    "player_order_side_resolve": "backend/app/pipeline/strokes.py:162-174  frame_players from player_detections",

    # Sequence length
    "seq_len_param": "backend/app/pipeline/strokes.py:35  seq_len parameter passed to _build_clip",
    "seq_len_zeros": "backend/app/pipeline/strokes.py:57-59  np.zeros((seq_len, ...)) allocation",

    # Joint order
    "pose_read": "backend/app/pipeline/strokes.py:90-101  _get_keypoints_for_frame reads pose_df",
    "pose_apply": "backend/app/pipeline/strokes.py:380-383  normalize_joints(masked_coords, det_bbox=None, bbox_margin=settings.bst_bbox_margin, conf=...)",

    # Bone edges
    "bone_pairs_def": "backend/app/pipeline/shared/bst_preproc.py:10-16  BONE_PAIRS list",
    "bone_create": "backend/app/pipeline/strokes.py:205-215  create_bones + concat to JnB",

    # Shuttle normalization
    "shuttle_norm_resolution": "backend/app/pipeline/strokes.py:147-149  shuttle[t] = sx / vid_w, sy / vid_h",
    "shuttle_norm_court": "backend/app/pipeline/strokes.py:143-146  shuttle[t] = image_to_court + /court_dims",
    "shuttle_norm_setting": "backend/app/config/settings.py:117  bst_shuttle_norm = 'resolution'|'court'",

    # Joint normalization
    "joint_norm_bbox": "backend/app/pipeline/strokes.py:380-383  normalize_joints(masked_coords, det_bbox=None [keypoint-derived bbox], bbox_margin=settings.bst_bbox_margin, conf=...)",
    "joint_norm_court": "backend/app/pipeline/strokes.py:367-368  normalize_joints_court(coords, homography)",
    "joint_norm_setting": "backend/app/config/settings.py:132  bst_joint_norm = 'bbox'|'court'",
    "normalize_joints_fn": "backend/app/pipeline/shared/bst_preproc.py:19-77  normalize_joints() bbox diag + center_align + conf mask",
    "normalize_joints_batched_fn": "backend/app/pipeline/shared/bst_preproc.py:80-109  normalize_joints_batched()",

    # Center align
    "center_align_code": "backend/app/pipeline/shared/bst_preproc.py:71-73  center_align subtraction (per-frame normalize_joints); batched at 103-107",

    # Player positions
    "pos_court": "backend/app/pipeline/strokes.py:193-198  pos[t] = image_to_court(feet) / court_dims",
    "pos_pixel": "backend/app/pipeline/strokes.py:199-201  pos[t] = feet / vid_w, vid_h fallback",

    # Missing joints
    "missing_pose": "backend/app/pipeline/strokes.py:202-203  kps None → debug_clip_stats n_missing_pose += 1",
    "missing_bbox": "backend/app/pipeline/strokes.py:365-366  interpolated_bbox coverage (diagnostic; no longer affects normalization)",
    "interp_bbox": "backend/app/pipeline/strokes.py:83-109  _interpolate_bboxes() linear fill",

    # Hit frame alignment
    "clip_window_start": "backend/app/pipeline/strokes.py:27  _build_clip — frame 0 = hit_frame",
    "clip_window_end": "backend/app/pipeline/strokes.py:306-351  floor/ceiling from StrokeClassificationStage.run()",
    "clip_boundary_setting": "backend/app/config/settings.py:208  bst_clip_boundary = 'midpoint' (hit centered: wind-up + follow-through) | 'hit_start' (frame 0 = hit)",
}


# ── COCO-17 keypoint names for readable diagnostic messages ──────────
COCO_17_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]

# Anatomical plausibility checks for COCO-17 joint ordering.
# Each entry: (higher_y_joint, lower_y_joint, description, pose_invariant).
# In image coords (y ↑ down), the first joint should be above the second.
#
# ``pose_invariant`` marks chains whose vertical order holds across ALL athletic
# poses: eyes above nose, shoulder above hip, knee above ankle. A genuine
# keypoint-ORDER swap would corrupt these too, so they are the reliable signal
# for a COCO-17 mismatch.
#
# ``pose_invariant=False`` marks chains that LEGITIMATELY invert during badminton
# motion — raised-racket arms (shoulder>elbow>wrist) and deep lunges
# (hip>knee). These are NEVER taken as evidence of a joint-order mismatch; they
# are expected for the sport and only produce an informational note.
_ANATOMY_CHECKS = [
    (1, 0, "L_eye should be above nose", True),
    (2, 0, "R_eye should be above nose", True),
    (5, 11, "L_shoulder should be above L_hip", True),
    (6, 12, "R_shoulder should be above R_hip", True),
    (13, 15, "L_knee should be above L_ankle", True),
    (14, 16, "R_knee should be above R_ankle", True),
    (5, 7, "L_shoulder should be above L_elbow", False),
    (7, 9, "L_elbow should be above L_wrist", False),
    (6, 8, "R_shoulder should be above R_elbow", False),
    (8, 10, "R_elbow should be above R_wrist", False),
    (11, 13, "L_hip should be above L_knee", False),
    (12, 14, "R_hip should be above R_knee", False),
]

# Fraction of a clip's valid frames for a (player, joint-pair) above which an
# anatomical violation is treated as a SYSTEMATIC keypoint-order defect rather
# than transient pose-estimation noise. Below this, a few flipped joints per
# pair (e.g. a cocked racket or occlusion) is expected and not flagged as a
# joint-order mismatch.
_ANATOMY_SYSTEMATIC_FRAC = 0.3


def _loc(check_name: str) -> str:
    """Return a ` [file:line]` suffix for the given check name."""
    loc = _SRC.get(check_name, "?")
    return f" [{loc}]"


@dataclass
class ValidationResult:
    """Result of a validation run."""
    n_checks: int = 0
    n_passed: int = 0
    n_warnings: int = 0
    n_errors: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all checks passed (no errors and no warnings)."""
        return self.n_errors == 0 and self.n_warnings == 0

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.n_checks += other.n_checks
        self.n_passed += other.n_passed
        self.n_warnings += other.n_warnings
        self.n_errors += other.n_errors
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        return self


class ValidationError(Exception):
    """Raised when a critical validation check fails in strict mode."""


class BSTInputValidator:
    """Validates BST clip tensors before model inference.

    Args:
        seq_len: Expected sequence length (from model checkpoint).
        n_classes: Expected number of output classes (for context).
        shuttle_norm: "resolution" or "court" — which normalization is active.
        joint_norm: "bbox" or "court" — which normalization is active.
        level: "warn" (default), "error", or "off".
        center_align: Whether center_align is applied (default True).
        clip_boundary: "hit_start" or "midpoint" — affects hit-frame alignment check.
    """

    def __init__(
        self,
        seq_len: int = 100,
        n_classes: int = 25,
        shuttle_norm: str = "resolution",
        joint_norm: str = "bbox",
        level: str = "warn",
        center_align: bool = True,
        clip_boundary: str = "hit_start",
    ):
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.shuttle_norm = shuttle_norm
        self.joint_norm = joint_norm
        self.level = level
        self.center_align = center_align
        self.clip_boundary = clip_boundary

    # ── Public API ───────────────────────────────────────────────────

    def validate_clip(self, clip: dict) -> ValidationResult:
        """Validate a single clip dict before batching."""
        result = ValidationResult()

        if self.level == "off":
            return result

        JnB = clip.get("JnB")
        shuttle = clip.get("shuttle")
        pos = clip.get("pos")
        vid_w = clip.get("vid_w", 0)
        vid_h = clip.get("vid_h", 0)

        if JnB is not None:
            result.merge(self._check_seq_len(JnB))
            result.merge(self._check_player_order(pos))
            result.merge(self._check_joint_order(JnB))
            result.merge(self._check_bone_edges(JnB))
            result.merge(self._check_joint_norm(JnB))
            result.merge(self._check_center_align(JnB))
            result.merge(self._check_missing_joints(JnB))

        if shuttle is not None:
            result.merge(self._check_shuttle_norm(shuttle, vid_w, vid_h))

        if pos is not None:
            result.merge(self._check_player_pos(pos))

        result.merge(self._check_hit_frame_alignment(shuttle))

        self._log_and_maybe_raise(result)
        return result

    def validate_batch(self, JnB: np.ndarray, shuttle: np.ndarray,
                       pos: np.ndarray) -> ValidationResult:
        """Validate batched tensors (B, T, ...) just before model call."""
        result = ValidationResult()

        if self.level == "off":
            return result

        result.merge(self._check_batch_seq_len(JnB))
        result.merge(self._check_batch_dtype(JnB, shuttle, pos))
        result.merge(self._check_batch_nan(JnB, shuttle, pos))

        self._log_and_maybe_raise(result)
        return result

    # ── Check 1: p0 = far player, p1 = near player ──────────────────

    def _check_player_order(self, pos: np.ndarray | None) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        if pos is None or pos.ndim < 3 or pos.shape[1] != 2:
            r.warnings.append(
                "Cannot check player order: pos tensor missing or wrong shape"
                + _loc("player_order_set")
            )
            r.n_warnings += 1
            return r

        # Use court-x (depth axis) to determine far/near order.
        # In the court model (court.py), x=0 is the far end (top of image)
        # and x=COURT_LENGTH (13.4) is the near end (bottom of image).
        # After normalization: far player has smaller x, near player has larger x.
        mid = min(pos.shape[0] // 2, 10)
        far_x = pos[:mid, 0, 0].mean()
        near_x = pos[:mid, 1, 0].mean()

        if abs(far_x - near_x) < 0.05:
            r.warnings.append(
                f"Player order: far-player mean x ({far_x:.3f}) ≈ "
                f"near-player mean x ({near_x:.3f}). Players may be "
                "stacked along the depth axis or one side is missing."
                + _loc("player_order_side_resolve")
            )
            r.n_warnings += 1
            return r

        if far_x > near_x:
            r.warnings.append(
                f"Player order may be REVERSED: p0 (far) mean x={far_x:.3f} "
                f"> p1 (near) mean x={near_x:.3f}. "
                "Expected far_x < near_x (far player closer to x=0 / far end)."
                + _loc("player_order_set")
            )
            r.n_warnings += 1
            return r

        r.n_passed += 1
        return r

    # ── Check 2: Sequence length matches training seq_len ────────────

    def _check_seq_len(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1
        if JnB.shape[0] != self.seq_len:
            r.errors.append(
                f"Sequence length mismatch: clip has {JnB.shape[0]} frames, "
                f"model expects {self.seq_len}. "
                "This will cause a tensor shape error at model call."
                + _loc("seq_len_param")
            )
            r.n_errors += 1
        else:
            r.n_passed += 1
        return r

    # ── Check 3: Joint order matches COCO-17 ─────────────────────────

    def _check_joint_order(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1
        if JnB.shape[-1] < 34:
            r.warnings.append(
                f"Cannot check joint order: expected ≥34 joint values "
                f"(17 keypoints × 2), got {JnB.shape[-1]}."
                + _loc("pose_apply")
            )
            r.n_warnings += 1
            return r

        joints_flat = JnB[:, :, :34]
        joints_xy = joints_flat.reshape(*joints_flat.shape[:2], 17, 2)

        n_violations = 0
        n_systematic = 0           # systematic POSE-INVARIANT (true order-defect) pairs
        n_systematic_variant = 0   # systematic pose-variant (athletic) pairs
        for player_idx in range(2):
            player_joints = joints_xy[:, player_idx]
            non_zero_mask = (np.abs(player_joints).sum(axis=-1) > 1e-6)
            for high_j, low_j, desc, invariant in _ANATOMY_CHECKS:
                both_valid = non_zero_mask[:, high_j] & non_zero_mask[:, low_j]
                n_both = int(both_valid.sum())
                if n_both == 0:
                    continue
                high_y = player_joints[both_valid, high_j, 1]
                low_y = player_joints[both_valid, low_j, 1]
                violated = high_y > low_y
                n_v = int(violated.sum())
                if n_v == 0:
                    continue
                n_violations += 1
                frac = n_v / n_both
                if frac > _ANATOMY_SYSTEMATIC_FRAC:
                    if invariant:
                        # Violations affect a large fraction of the clip's frames
                        # for a POSE-INVARIANT pair → a real keypoint-order defect
                        # (e.g. the pose model emitting a different ordering than
                        # COCO-17).
                        n_systematic += 1
                        r.warnings.append(
                            f"Player {player_idx}: {desc} violated in {n_v}/{n_both} "
                            f"frames ({frac:.0%}) — systematic. Joint order may not "
                            "match COCO-17." + _loc("pose_read")
                        )
                        r.n_warnings += 1
                    else:
                        # Pose-variant pair (raised-arm / lunge) inverted at high
                        # rate — expected for badminton, NOT an order defect.
                        n_systematic_variant += 1
                        r.warnings.append(
                            f"Player {player_idx}: {desc} violated in {n_v}/{n_both} "
                            f"frames ({frac:.0%}). Expected for raised-arm / lunge "
                            "athletic poses; not treated as a keypoint-order defect."
                            + _loc("pose_read")
                        )
                        r.n_warnings += 1
                elif n_violations <= 3:
                    # A handful of frames per pair is normal pose-estimation
                    # noise (cocked racket, occlusion) and does NOT indicate a
                    # joint-order mismatch. Report it as benign only.
                    r.warnings.append(
                        f"Player {player_idx}: {desc} violated in {n_v}/{n_both} "
                        "frames. Likely transient pose-estimation noise "
                        "(not a keypoint-order defect)." + _loc("pose_read")
                    )
                    r.n_warnings += 1

        if n_violations == 0:
            r.n_passed += 1
        elif n_systematic >= 2:
            # Multiple systematically-violated POSE-INVARIANT pairs strongly
            # indicate the pose model emits a different keypoint definition
            # than COCO-17.
            r.warnings.append(
                f"Joint order: {n_systematic} systematic anatomical violations "
                "across joint pairs. Pose model may output a different "
                "keypoint definition than COCO-17." + _loc("pose_read")
            )
            r.n_warnings += 1
        return r

    # ── Check 4: Bone edges match JnB_bone configuration ─────────────

    def _check_bone_edges(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        n_expected_bones = len(BONE_PAIRS)
        n_expected_joints = 17
        n_expected_total = (n_expected_joints + n_expected_bones) * 2

        if JnB.shape[-1] != n_expected_total:
            r.errors.append(
                f"JnB feature dimension mismatch: got {JnB.shape[-1]}, "
                f"expected {n_expected_total} "
                f"(= ({n_expected_joints} joints + {n_expected_bones} bones) × 2). "
                "BONE_PAIRS config differs from what the model was trained on."
                + _loc("bone_pairs_def")
            )
            r.n_errors += 1
            return r

        bones_flat = JnB[:, :, 34:]
        non_zero_frac = (np.abs(bones_flat) > 1e-6).mean()
        if non_zero_frac < 0.01:
            r.warnings.append(
                f"Bone vectors are nearly all zero (non-zero fraction: "
                f"{non_zero_frac:.4f}). "
                "Skeleton may be collapsed or all keypoints missing."
                + _loc("bone_create")
            )
            r.n_warnings += 1
        else:
            r.n_passed += 1
        return r

    # ── Check 5: Shuttle tensor normalized by video width/height ─────

    def _check_shuttle_norm(self, shuttle: np.ndarray,
                            vid_w: float, vid_h: float) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        non_zero = shuttle[np.any(shuttle != 0, axis=1)]
        if len(non_zero) == 0:
            r.warnings.append(
                "Shuttle tensor is all-zero — no valid detections in clip."
                + _loc("shuttle_norm_resolution")
            )
            r.n_warnings += 1
            return r

        x_vals = non_zero[:, 0]
        y_vals = non_zero[:, 1]

        x_out = (x_vals < -0.02) | (x_vals > 1.02)
        y_out = (y_vals < -0.02) | (y_vals > 1.02)

        n_out_x = int(x_out.sum())
        n_out_y = int(y_out.sum())

        if n_out_x > 0 or n_out_y > 0:
            mode = self.shuttle_norm
            loc_key = "shuttle_norm_court" if mode == "court" else "shuttle_norm_resolution"
            msg_parts = []
            if n_out_x > 0:
                msg_parts.append(f"x range [{x_vals.min():.3f}, {x_vals.max():.3f}] (expected [0,1])")
            if n_out_y > 0:
                msg_parts.append(f"y range [{y_vals.min():.3f}, {y_vals.max():.3f}] (expected [0,1])")
            r.warnings.append(
                f"Shuttle values outside [0, 1]: {'; '.join(msg_parts)}. "
                f"Active mode: '{mode}' ("
                + _loc("shuttle_norm_setting")
                + "). Check that normalization at "
                + _loc(loc_key)
                + " matches the configured mode."
            )
            r.n_warnings += 1
        else:
            r.n_passed += 1
        return r

    # ── Check 6: Joints are bbox-relative / correct range ────────────

    def _check_joint_norm(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        joints_flat = JnB[:, :, :34]
        non_zero = joints_flat[np.any(joints_flat != 0, axis=-1)]

        if len(non_zero) == 0:
            r.warnings.append(
                "Joint tensor is all-zero — no valid pose data."
                + _loc("pose_apply")
            )
            r.n_warnings += 1
            return r

        if self.joint_norm == "bbox":
            lower, upper = -0.6, 0.6
            label = "bbox-diagonal + center_align"
            fn_loc = _loc("normalize_joints_fn")
        elif self.joint_norm == "court":
            lower, upper = -0.6, 0.6
            label = "court-space"
            fn_loc = _loc("joint_norm_court")
        else:
            r.warnings.append(f"Unknown joint_norm '{self.joint_norm}'")
            r.n_warnings += 1
            return r

        vmin = float(non_zero.min())
        vmax = float(non_zero.max())

        if vmin < lower or vmax > upper:
            mode_loc = _loc("joint_norm_setting")
            r.warnings.append(
                f"Joint values exceed expected range [{lower}, {upper}] "
                f"for '{label}' normalization: "
                f"actual [{vmin:.3f}, {vmax:.3f}]. "
                f"Setting says {self.joint_norm}" + mode_loc
                + f". Normalization function at" + fn_loc
                + (". Note: colab overrides to 'court' at colab/pipeline.py:1192"
                   if self.joint_norm == "bbox" else "")
            )
            r.n_warnings += 1
        else:
            r.n_passed += 1
        return r

    # ── Check 7: Joints are center-aligned ───────────────────────────

    def _check_center_align(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        if not self.center_align:
            r.n_passed += 1
            return r

        joints_flat = JnB[:, :, :34]
        nonzero_mask = joints_flat != 0
        nonzero_count = int(nonzero_mask.sum())

        if nonzero_count == 0:
            r.warnings.append(
                "Cannot check center alignment: all joints are zero."
                + _loc("pose_apply")
            )
            r.n_warnings += 1
            return r

        # Per-coordinate mean: divide the sum of ALL coordinate values by the
        # number of NON-ZERO coordinate values — NOT by the number of
        # (frame, player) pairs with any valid joint. The pair-wise divisor
        # inflates a genuine ~-0.08 per-coordinate mean into ~-1.3 and falsely
        # reports a centering failure on properly aligned clips that merely
        # have sparse (partially-missing) keypoints.
        total_sum = float(joints_flat.sum())
        mean_val = total_sum / nonzero_count

        if abs(mean_val) > 0.15:
            r.warnings.append(
                f"Joint mean is {mean_val:.3f} (expected ≈ 0 for center-aligned data). "
                "Center alignment may not have been applied."
                + _loc("center_align_code")
                + " (invoked from "
                + _loc("joint_norm_bbox")
                + ")"
            )
            r.n_warnings += 1
        else:
            r.n_passed += 1
        return r

    # ── Check 8: Player positions are in [0, 1] ──────────────────────

    def _check_player_pos(self, pos: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        non_zero = pos[np.any(pos != 0, axis=-1)]
        if len(non_zero) == 0:
            r.warnings.append(
                "Player position tensor is all-zero."
                + _loc("pos_court")
            )
            r.n_warnings += 1
            return r

        vmin = float(non_zero.min())
        vmax = float(non_zero.max())

        if vmin < -0.02 or vmax > 1.02:
            has_homography = vmin >= 0 and vmax <= 1  # tight check — if court-normalized
            loc_key = "pos_court" if has_homography else "pos_pixel"
            r.warnings.append(
                f"Player positions outside expected [0, 1] range: "
                f"actual [{vmin:.3f}, {vmax:.3f}]. "
                "Check normalization at "
                + _loc(loc_key)
            )
            r.n_warnings += 1
        else:
            r.n_passed += 1
        return r

    # ── Check 9: Missing joints (imputation consistency) ─────────────

    def _check_missing_joints(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        joints_flat = JnB[:, :, :34]
        joints_xy = joints_flat.reshape(*joints_flat.shape[:2], 17, 2)

        for p_idx in range(2):
            player = joints_xy[:, p_idx]
            zero_joints = (np.abs(player).sum(axis=-1) < 1e-6)
            all_zero_frames = zero_joints.all(axis=1)
            pct_missing = float(all_zero_frames.mean()) * 100

            if pct_missing > 50:
                r.warnings.append(
                    f"Player {p_idx}: {pct_missing:.0f}% frames have all-zero joints. "
                    "Pose data is largely missing for this player."
                    + _loc("missing_pose")
                    + " (bbox interpolation at "
                    + _loc("interp_bbox")
                    + " may not suffice)"
                )
                r.n_warnings += 1
            elif pct_missing > 10:
                r.warnings.append(
                    f"Player {p_idx}: {pct_missing:.0f}% frames have all-zero joints. "
                    "Partial pose data loss may degrade BST accuracy."
                    + _loc("missing_pose")
                )
                r.n_warnings += 1

            partial_mask = zero_joints.any(axis=1) & (~all_zero_frames)
            pct_partial = float(partial_mask.mean()) * 100
            if pct_partial > 20:
                r.warnings.append(
                    f"Player {p_idx}: {pct_partial:.0f}% frames have partially "
                    "missing joints (some keypoints present, others zero). "
                    "Missing joints are zeroed consistently, but reduced pose "
                    "coverage may degrade BST accuracy."
                    + _loc("missing_bbox")
                )
                r.n_warnings += 1

        if r.n_warnings == 0:
            r.n_passed += 1
        return r

    # ── Check 10: Hit frame / clip alignment ─────────────────────────

    def _check_hit_frame_alignment(self, shuttle: np.ndarray | None) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        if shuttle is None or len(shuttle) == 0:
            r.warnings.append(
                "Cannot check hit frame alignment: no shuttle data."
                + _loc("shuttle_norm_resolution")
            )
            r.n_warnings += 1
            return r

        non_zero_mask = np.any(shuttle != 0, axis=1)
        valid_pts = shuttle[non_zero_mask]
        if len(valid_pts) < 3:
            r.warnings.append(
                f"Too few valid shuttle detections ({len(valid_pts)}) "
                "to check clip alignment."
                + _loc("shuttle_norm_resolution")
            )
            r.n_warnings += 1
            return r

        first_valid_idx = int(np.argmax(non_zero_mask)) if non_zero_mask.any() else 0
        first_valid = shuttle[first_valid_idx]
        y_range = valid_pts[:, 1].max() - valid_pts[:, 1].min() + 1e-6
        y_frac = (first_valid[1] - valid_pts[:, 1].min()) / y_range

        if self.clip_boundary == "midpoint":
            # Midpoint mode: frame 0 is the approach/preparation phase.
            # The shuttle should be IN FLIGHT (not at the contact point),
            # approaching from the previous hit. It should be somewhere
            # in mid-trajectory, not at an extreme (which would mean it's
            # sitting on the ground or at the peak of a clear).
            at_extreme = y_frac < 0.10 or y_frac > 0.90
            at_contact = 0.35 < y_frac < 0.65  # suspiciously centered = contact at frame 0
            if at_extreme:
                r.warnings.append(
                    f"Midpoint clip: frame-0 shuttle y is at trajectory extreme "
                    f"(y_frac={y_frac:.2f}, expected ~0.1-0.9 in-flight position). "
                    "Clip may start too early or too late."
                    + _loc("clip_boundary_setting")
                )
                r.n_warnings += 1
            elif at_contact:
                r.warnings.append(
                    f"Midpoint clip: frame-0 shuttle y is centered in trajectory "
                    f"(y_frac={y_frac:.2f}), suggesting the contact frame is at "
                    "position 0. Clipping may be using hit_start convention "
                    "despite midpoint setting."
                    + _loc("clip_boundary_setting")
                )
                r.n_warnings += 1
            else:
                r.n_passed += 1

            # Also check that the trajectory midpoint (where we expect contact)
            # falls in the middle third of the clip duration.
            mid = len(shuttle) // 2
            mid_y = shuttle[mid, 1] if np.any(shuttle[mid] != 0) else None
            if mid_y is not None and y_range > 0.01:
                mid_y_frac = (mid_y - valid_pts[:, 1].min()) / y_range
                if mid_y_frac < 0.15 or mid_y_frac > 0.85:
                    r.warnings.append(
                        f"Midpoint clip: mid-frame shuttle y is at trajectory "
                        f"extreme (y_frac={mid_y_frac:.2f}), expected near mid-range "
                        "for the contact point."
                        + _loc("clip_boundary_setting")
                    )
                    r.n_warnings += 1
        else:
            # Hit-start mode: frame 0 = the stroke contact point.
            # The shuttle should be near the middle of its trajectory
            # (being struck), not at an extreme.
            if y_frac < 0.05 or y_frac > 0.95:
                r.warnings.append(
                    f"Hit-start clip: frame-0 shuttle y is at extreme of trajectory "
                    f"(y_frac={y_frac:.2f}, expected ~0.5 for contact). "
                    "Clip may not be starting at the contact frame."
                    + _loc("clip_window_start")
                    + " (window limits at "
                    + _loc("clip_window_end")
                    + ")"
                )
                r.n_warnings += 1
            else:
                r.n_passed += 1
        return r

    # ── Batch-level checks ───────────────────────────────────────────

    def _check_batch_seq_len(self, JnB: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1
        if JnB.shape[1] != self.seq_len:
            r.errors.append(
                f"Batch sequence length mismatch: got {JnB.shape[1]}, "
                f"model expects {self.seq_len}."
                + _loc("seq_len_param")
            )
            r.n_errors += 1
        else:
            r.n_passed += 1
        return r

    def _check_batch_dtype(self, JnB: np.ndarray, shuttle: np.ndarray,
                           pos: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        expected = np.float32
        for name, arr in [("JnB", JnB), ("shuttle", shuttle), ("pos", pos)]:
            if arr.dtype != expected:
                r.warnings.append(
                    f"Batch {name} dtype is {arr.dtype}, expected {expected}. "
                    "May cause silent precision loss at model call."
                    + _loc("seq_len_zeros")
                )
                r.n_warnings += 1

        if r.n_warnings == 0:
            r.n_passed += 1
        return r

    def _check_batch_nan(self, JnB: np.ndarray, shuttle: np.ndarray,
                         pos: np.ndarray) -> ValidationResult:
        r = ValidationResult()
        r.n_checks += 1

        n_nan = 0
        for name, arr in [("JnB", JnB), ("shuttle", shuttle), ("pos", pos)]:
            n = int(np.isnan(arr).sum())
            if n > 0:
                loc_key = {
                    "JnB": "pose_apply",
                    "shuttle": "shuttle_norm_resolution",
                    "pos": "pos_court",
                }[name]
                r.warnings.append(
                    f"Batch {name} contains {n} NaN values. "
                    "NaN will propagate silently through the model."
                    + _loc(loc_key)
                )
                r.n_warnings += 1
                n_nan += n

        if n_nan == 0:
            r.n_passed += 1
        return r

    # ── Internal helpers ─────────────────────────────────────────────

    def _log_and_maybe_raise(self, result: ValidationResult):
        if result.n_checks == 0:
            return

        for msg in result.errors:
            logger.error("BST VALIDATION FAIL: %s", msg)
        for msg in result.warnings:
            logger.warning("BST VALIDATION: %s", msg)

        if result.n_errors > 0 and self.level == "error":
            raise ValidationError(
                f"BST validation failed with {result.n_errors} error(s) "
                f"and {result.n_warnings} warning(s). "
                f"First error: {result.errors[0]}"
            )

    def __repr__(self) -> str:
        return (
            f"BSTInputValidator(seq_len={self.seq_len}, "
            f"shuttle_norm='{self.shuttle_norm}', "
            f"joint_norm='{self.joint_norm}', "
            f"level='{self.level}', "
            f"center_align={self.center_align})"
        )
