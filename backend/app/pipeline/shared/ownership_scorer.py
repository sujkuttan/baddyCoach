import numpy as np
import pandas as pd

from app.pipeline.shared.court import (
    image_to_court, foot_midpoint_from_pose, foot_point_from_bbox,
    COURT_LENGTH, COURT_WIDTH,
)


# ── Utility ────────────────────────────────────────────────────────
_R_SHOULDER = 6
_R_ELBOW = 8
_R_WRIST = 10
_R_HIP = 12


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float | None:
    """Angle (radians) between two vectors.  Returns None on degenerate input."""
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return float(np.arccos(np.clip(dot / (n1 * n2), -1.0, 1.0)))





# ── Sub-score functions ────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, returns [-1, 1]."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(dot / (na * nb), -1.0, 1.0))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def trajectory_ownership_score(shuttle_before: np.ndarray | None,
                                shuttle_now: np.ndarray | None,
                                shuttle_after: np.ndarray | None,
                                player_court: np.ndarray | None) -> float:
    """Score how likely *this* player hit based on shuttle trajectory.

    Uses court-space coordinates exclusively.  Before contact the shuttle
    should be travelling **toward** the hitter; after contact it should be
    travelling **away** from the hitter.

    Parameters
    ----------
    shuttle_before : np.ndarray | None  — court-space xy at t-3
    shuttle_now   : np.ndarray | None  — court-space xy at t
    shuttle_after : np.ndarray | None  — court-space xy at t+3
    player_court  : np.ndarray | None  — court-space foot xy of this player at t

    Returns
    -------
    float in [0, 1] — higher = more likely this player hit it.
    """
    if any(s is None for s in (shuttle_before, shuttle_now, shuttle_after, player_court)):
        return 0.5

    v_in = shuttle_now - shuttle_before          # incoming velocity
    v_out = shuttle_after - shuttle_now          # outgoing velocity

    to_player = player_court - shuttle_now       # vector from shuttle → player
    away_from_player = shuttle_after - player_court  # vector from player → outgoing

    incoming_towards = _cosine_similarity(v_in, to_player)
    outgoing_away = _cosine_similarity(v_out, away_from_player)

    raw = 0.5 * incoming_towards + 0.5 * outgoing_away
    return _clamp01((raw + 1.0) / 2.0)


def court_side_feasibility_score(shuttle_court: tuple[float, float] | None,
                                  net_y: float = COURT_LENGTH / 2,
                                  margin: float = 0.75,
                                  wrong_side_score: float = 0.20) -> tuple[float, float]:
    """Score based on which side of the court the shuttle is on.

    A player is more likely to have hit the shuttle when the contact point is
    on or near their side of the court.  The ``net_y`` parameter is the court-
    length coordinate of the net (default 6.7 m = COURT_LENGTH / 2).

    Within ``margin`` of the net both sides are possible; beyond that margin
    the wrong-side player receives ``wrong_side_score``.

    Returns (near_score, far_score).
    """
    if shuttle_court is None:
        return wrong_side_score, wrong_side_score

    shuttle_court_y = shuttle_court[0]  # length-axis coordinate

    near_score = 1.0 if shuttle_court_y < net_y + margin else wrong_side_score
    far_score = 1.0 if shuttle_court_y > net_y - margin else wrong_side_score

    return near_score, far_score


def _wrist_px_and_conf(kps: np.ndarray | None) -> tuple[np.ndarray | None, float]:
    """Get racket wrist (COCO 10, right wrist) pixel position and confidence.

    Returns (pixel_xy, confidence).  If keypoints are missing returns (None, 0.0).
    """
    if kps is None or kps.shape != (17, 3):
        return None, 0.0
    return kps[10, :2].copy(), float(kps[10, 2])


def normalized_proximity_score(shuttle_px: np.ndarray | None,
                                shuttle_court: np.ndarray | None,
                                near_kps: np.ndarray | None,
                                far_kps: np.ndarray | None,
                                near_bbox_h: float | None,
                                far_bbox_h: float | None,
                                H_arr: np.ndarray | None,
                                sigma_norm: float = 0.15,
                                sigma_meters: float = 0.75,
                                min_pose_conf: float = 0.25,
                                unknown_score: float = 0.50) -> tuple[float, float]:
    """Proximity score with two methods (spec 13.1, 13.2).

    **Preferred (13.1):** Court-coordinate proximity — project wrist to court-
    space via homography and measure Euclidean distance to shuttle court-space
    position: ``score = exp(-distance_m / sigma_meters)``.

    **Fallback (13.2):** Body-scale-normalised pixel distance:
    ``normalised_distance = pixel_distance(shuttle_px, wrist_px) / bbox_h``
    ``score = exp(-normalised_distance / sigma_norm)``.

    If wrist keypoint confidence < ``min_pose_conf`` returns ``unknown_score``.

    Returns (near_score, far_score).
    """
    if shuttle_px is None:
        return unknown_score, unknown_score

    def _prox(kps, bbox_h):
        if kps is None or bbox_h is None or bbox_h < 1:
            return unknown_score
        wrist_px, conf = _wrist_px_and_conf(kps)
        if wrist_px is None or conf < min_pose_conf:
            return unknown_score

        # Preferred: court-coordinate proximity (13.1)
        if shuttle_court is not None and H_arr is not None:
            try:
                wrist_court = image_to_court(H_arr, (float(wrist_px[0]), float(wrist_px[1])))
                dist_m = float(np.linalg.norm(np.array(wrist_court) - shuttle_court))
                score = float(np.exp(-dist_m / sigma_meters))
                if np.isfinite(score):
                    return score
            except Exception:
                pass

        # Fallback: body-scale-normalised pixel distance (13.2)
        dist = float(np.linalg.norm(wrist_px - shuttle_px))
        norm_dist = dist / bbox_h
        score = float(np.exp(-norm_dist / sigma_norm))
        return score if np.isfinite(score) else unknown_score

    ns = _prox(near_kps, near_bbox_h)
    fs = _prox(far_kps, far_bbox_h)

    ns = ns if np.isfinite(ns) else unknown_score
    fs = fs if np.isfinite(fs) else unknown_score

    return ns, fs


def racket_motion_score(near_kps_list: list[np.ndarray],
                         far_kps_list: list[np.ndarray],
                         hit_idx: int,
                         wrist_weight: float = 0.50,
                         elbow_weight: float = 0.30,
                         shoulder_weight: float = 0.20,
                         min_confidence: float = 0.35,
                         unknown_score: float = 0.50,
                         vel_norm: float = 50.0) -> tuple[float, float]:
    """Score based on racket-arm motion around the hit frame (spec §10.3).

    Uses wrist velocity, elbow angular velocity, and shoulder angular
    velocity at the hit frame via central difference.  Low-confidence
    pose returns neutral evidence (unknown_score).

    Returns (near_score, far_score).
    """
    def _mean_conf(kps_list, hi):
        kps = kps_list[hi] if kps_list and hi < len(kps_list) else None
        if kps is None or kps.shape != (17, 3):
            return 0.0
        return float(np.mean([kps[_R_WRIST, 2], kps[_R_ELBOW, 2], kps[_R_SHOULDER, 2]]))

    def _velocity_at(joint, hi, kps_list):
        if hi < 1 or hi >= len(kps_list) - 1:
            return None
        prev, curr, nxt = kps_list[hi-1], kps_list[hi], kps_list[hi+1]
        if any(k is None for k in (prev, curr, nxt)):
            return None
        return float(np.linalg.norm(nxt[joint, :2] - prev[joint, :2]) / 2.0)

    def _elbow_angle_rad(kps):
        if kps is None or kps.shape != (17, 3):
            return None
        s, e, w = kps[_R_SHOULDER, :2], kps[_R_ELBOW, :2], kps[_R_WRIST, :2]
        return _angle_between(s - e, w - e)

    def _shoulder_angle_rad(kps):
        if kps is None or kps.shape != (17, 3):
            return None
        h, s, e = kps[_R_HIP, :2], kps[_R_SHOULDER, :2], kps[_R_ELBOW, :2]
        return _angle_between(h - s, e - s)

    def _ang_vel_at(angle_fn, hi, kps_list):
        if hi < 1 or hi >= len(kps_list) - 1:
            return None
        prev = angle_fn(kps_list[hi-1])
        nxt = angle_fn(kps_list[hi+1])
        if prev is None or nxt is None:
            return None
        return float(abs(nxt - prev) / 2.0)

    def _normalize(v):
        return 0.0 if v is None else min(1.0, v / vel_norm)

    def _motion_score(kps_list):
        if not kps_list or len(kps_list) < 3:
            return unknown_score
        if _mean_conf(kps_list, hit_idx) < min_confidence:
            return unknown_score

        wrist_vel = _velocity_at(_R_WRIST, hit_idx, kps_list)
        elbow_ang_vel = _ang_vel_at(_elbow_angle_rad, hit_idx, kps_list)
        shoulder_ang_vel = _ang_vel_at(_shoulder_angle_rad, hit_idx, kps_list)

        if wrist_vel is None and elbow_ang_vel is None and shoulder_ang_vel is None:
            return unknown_score

        raw = (wrist_weight * _normalize(wrist_vel) +
               elbow_weight * _normalize(elbow_ang_vel) +
               shoulder_weight * _normalize(shoulder_ang_vel))
        total_w = wrist_weight + elbow_weight + shoulder_weight
        return min(1.0, raw / total_w)

    near_score = _motion_score(near_kps_list)
    far_score = _motion_score(far_kps_list)

    total = near_score + far_score
    if total > 0:
        near_score /= total
        far_score /= total

    return float(near_score), float(far_score)


def pose_contact_feasibility_score(shuttle_px: np.ndarray | None,
                                    near_kps: np.ndarray | None,
                                    far_kps: np.ndarray | None,
                                    strong_reach_ratio: float = 0.75,
                                    medium_reach_ratio: float = 1.25,
                                    weak_reach_ratio: float = 1.75,
                                    min_confidence: float = 0.35,
                                    unknown_score: float = 0.50) -> tuple[float, float]:
    """Score based on biomechanical plausibility of the contact (spec §10.4).

    Uses the ratio of (wrist-to-shuttle distance) / (arm length).
    - < 0.75 : arm naturally reaches shuttle → 1.0
    - < 1.25 : plausible reach → 0.7
    - < 1.75 : extended reach → 0.4
    - ≥ 1.75 : unreachable → 0.1

    Low-confidence pose returns ``unknown_score`` (neutral evidence).

    Returns (near_score, far_score).
    """
    if shuttle_px is None:
        return unknown_score, unknown_score

    def _mean_upper_body_conf(kps):
        if kps is None or kps.shape != (17, 3):
            return 0.0
        return float(np.mean([kps[_R_SHOULDER, 2], kps[_R_ELBOW, 2], kps[_R_WRIST, 2]]))

    def _reach_score(kps):
        if _mean_upper_body_conf(kps) < min_confidence:
            return unknown_score

        shoulder = kps[_R_SHOULDER, :2]
        wrist = kps[_R_WRIST, :2]
        arm_length_px = float(max(np.linalg.norm(wrist - shoulder), 1.0))
        shuttle_to_wrist = float(np.linalg.norm(shuttle_px - wrist))
        reach_ratio = shuttle_to_wrist / arm_length_px

        if reach_ratio < strong_reach_ratio:
            return 1.0
        elif reach_ratio < medium_reach_ratio:
            return 0.7
        elif reach_ratio < weak_reach_ratio:
            return 0.4
        else:
            return 0.1

    ns = _reach_score(near_kps)
    fs = _reach_score(far_kps)

    total = ns + fs
    if total > 0:
        ns /= total
        fs /= total
    return float(ns), float(fs)


def initial_turn_prior_score(prev_owner: str | None,
                              near_id: str = "player_1",
                              far_id: str = "player_2",
                              alternate_score: float = 0.95,
                              same_player_score: float = 0.05,
                              first_hit_score: float = 0.50) -> tuple[float, float]:
    """Turn-taking prior: badminton alternates.

    Returns (near_score, far_score) based on who hit the previous shot.
    """
    if prev_owner is None:
        return first_hit_score, first_hit_score

    if prev_owner == near_id:
        # Near player just hit — far should hit next
        return same_player_score, alternate_score
    else:
        # Far player just hit — near should hit next
        return alternate_score, same_player_score


def bst_attribution_score(shot: dict | None,
                          alpha_threshold: float = 0.15,
                          conf_min: float = 0.3,
                          unknown_score: float = 0.50) -> tuple[float, float]:
    """Score based on BST AimPlayer attention and class_id prefix.

    Uses aimplayer_alpha (>0.5 → far, <0.5 → near) as a continuous
    signal and shuttleset_class_id prefix (Top_ → far, Bottom_ → near)
    as a discrete override.

    Returns (near_score, far_score).
    """
    if shot is None:
        return unknown_score, unknown_score

    alpha = shot.get("aimplayer_alpha", 0.5)
    class_id = shot.get("shuttleset_class_id", 0)
    conf = shot.get("stroke_confidence", 0)

    # Signal A: continuous AimPlayer alpha
    if abs(alpha - 0.5) <= alpha_threshold:
        alpha_near, alpha_far = unknown_score, unknown_score
    else:
        scale = (abs(alpha - 0.5) - alpha_threshold) / max(0.5 - alpha_threshold, 1e-6)
        raw_near = 1.0 - alpha
        raw_far = alpha
        alpha_near = unknown_score + scale * (raw_near - unknown_score)
        alpha_far = unknown_score + scale * (raw_far - unknown_score)

    near_score, far_score = alpha_near, alpha_far

    # Signal B: discrete class_id prefix (overrides alpha when available)
    if class_id > 0 and conf >= conf_min:
        try:
            from app.models.bst import get_shuttleset_class_info, SHUTTLESET_CLASSES
            if class_id <= len(SHUTTLESET_CLASSES) - 1:
                _, side = get_shuttleset_class_info(class_id)
                if side == "top":
                    near_score, far_score = 0.2, 0.8
                elif side == "bottom":
                    near_score, far_score = 0.8, 0.2
        except Exception:
            pass

    total = near_score + far_score
    if total > 0:
        near_score /= total
        far_score /= total

    return float(near_score), float(far_score)


# ── Viterbi rally-level assignment (spec §17) ────────────────────

class ViterbiConfig:
    """Transition probabilities for rally-level Viterbi owner assignment."""
    def __init__(self,
                 p_alternate: float = 0.95,
                 p_same: float = 0.05,
                 epsilon: float = 1e-6):
        self.p_alternate = p_alternate
        self.p_same = p_same
        self.epsilon = epsilon

    @classmethod
    def from_settings(cls):
        from app.config.settings import settings
        return cls(
            p_alternate=settings.viterbi_p_alternate,
            p_same=settings.viterbi_p_same,
            epsilon=settings.viterbi_epsilon,
        )


def assign_hit_owners_viterbi(candidates: list,
                               emissions: list[dict[str, float]],
                               config: ViterbiConfig) -> list[str]:
    """Assign owner sequence across a rally via Viterbi decoding.

    Parameters
    ----------
    candidates : list
        List of candidate shot records (used only for length).
    emissions : list[dict[str, float]]
        Per-candidate dict with ``"near"`` and ``"far"`` keys (probability-like,
        higher = more likely).  These are typically ``near_score`` / ``far_score``
        from ``OwnershipScorer.score()``.
    config : ViterbiConfig
        Transition probabilities (p_alternate, p_same) and log epsilon.

    Returns
    -------
    list[str]
        ``"near"`` or ``"far"`` for each candidate.
    """
    states = ["near", "far"]
    eps = config.epsilon
    log_alt = np.log(config.p_alternate + eps)
    log_same = np.log(config.p_same + eps)

    dp = []
    backptr = []

    for i in range(len(candidates)):
        dp.append({s: -1e18 for s in states})
        backptr.append({s: None for s in states})

        for s in states:
            log_emit = np.log(emissions[i][s] + eps)

            if i == 0:
                dp[i][s] = log_emit
                continue

            best_score = -1e18
            best_prev = None
            for ps in states:
                trans = log_alt if s != ps else log_same
                score = dp[i - 1][ps] + trans + log_emit
                if score > best_score:
                    best_score = score
                    best_prev = ps

            dp[i][s] = best_score
            backptr[i][s] = best_prev

    return _backtrack_viterbi(dp, backptr)


def _backtrack_viterbi(dp: list[dict[str, float]],
                        backptr: list[dict[str, str | None]]) -> list[str]:
    """Backtrack through Viterbi DP table to extract optimal state sequence."""
    last = len(dp) - 1
    final_state = max(dp[last], key=dp[last].get)
    owners = [final_state]
    for i in range(last, 0, -1):
        owners.append(backptr[i][owners[-1]])
    owners.reverse()
    return owners


# ── Scorer class ───────────────────────────────────────────────────

class OwnershipScorer:
    """Compute per-candidate near/far ownership scores from 6 signals.

    Usage:
        scorer = OwnershipScorer.from_settings()
        near_score, far_score, debug = scorer.score(shuttle_df, pose_df,
                                                      players_data, court_data,
                                                      candidate_frame, prev_owner)
    """

    def __init__(self,
                 trajectory_weight: float = 0.35,
                 court_side_weight: float = 0.20,
                 proximity_weight: float = 0.15,
                 motion_weight: float = 0.15,
                 pose_feasibility_weight: float = 0.10,
                 turn_prior_weight: float = 0.05,
                 bst_weight: float = 0.06,
                 bst_alpha_threshold: float = 0.15,
                 bst_conf_min: float = 0.3,
                 window_frames: int = 3,
                 net_margin: float = 0.75,
                 prox_sigma_norm: float = 0.15,
                 prox_sigma_meters: float = 0.75,
                 prox_min_pose_conf: float = 0.25,
                 min_pose_conf: float = 0.35,
                 unknown_score: float = 0.50,
                 strong_reach_ratio: float = 0.75,
                 medium_reach_ratio: float = 1.25,
                 weak_reach_ratio: float = 1.75,
                 alternate_score: float = 0.95,
                 same_player_score: float = 0.05,
                 first_hit_score: float = 0.50,
                 traj_min_shuttle_conf: float = 0.30,
                 traj_interp_penalty: float = 0.80,
                 court_net_y: float = 6.7,
                 court_wrong_side_score: float = 0.20,
                 motion_wrist_weight: float = 0.50,
                 motion_elbow_weight: float = 0.30,
                 motion_shoulder_weight: float = 0.20,
                 calib_near_mean: float = 0.62,
                 calib_near_std: float = 0.14,
                 calib_far_mean: float = 0.48,
                 calib_far_std: float = 0.18):
        self.trajectory_weight = trajectory_weight
        self.court_side_weight = court_side_weight
        self.proximity_weight = proximity_weight
        self.motion_weight = motion_weight
        self.pose_feasibility_weight = pose_feasibility_weight
        self.turn_prior_weight = turn_prior_weight
        self.bst_weight = bst_weight
        self.bst_alpha_threshold = bst_alpha_threshold
        self.bst_conf_min = bst_conf_min

        self.window_frames = window_frames
        self.net_margin = net_margin
        self.prox_sigma_norm = prox_sigma_norm
        self.prox_sigma_meters = prox_sigma_meters
        self.prox_min_pose_conf = prox_min_pose_conf
        self.min_pose_conf = min_pose_conf
        self.unknown_score = unknown_score
        self.strong_reach_ratio = strong_reach_ratio
        self.medium_reach_ratio = medium_reach_ratio
        self.weak_reach_ratio = weak_reach_ratio
        self.alternate_score = alternate_score
        self.same_player_score = same_player_score
        self.first_hit_score = first_hit_score
        self.traj_min_shuttle_conf = traj_min_shuttle_conf
        self.traj_interp_penalty = traj_interp_penalty
        self.court_net_y = court_net_y
        self.court_wrong_side_score = court_wrong_side_score
        self.motion_wrist_weight = motion_wrist_weight
        self.motion_elbow_weight = motion_elbow_weight
        self.motion_shoulder_weight = motion_shoulder_weight
        self.calib_near_mean = calib_near_mean
        self.calib_near_std = calib_near_std
        self.calib_far_mean = calib_far_mean
        self.calib_far_std = calib_far_std

    @classmethod
    def from_settings(cls):
        from app.config.settings import settings
        return cls(
            trajectory_weight=getattr(settings, 'ownership_trajectory_weight', 0.35),
            court_side_weight=getattr(settings, 'ownership_court_side_weight', 0.20),
            proximity_weight=getattr(settings, 'ownership_proximity_weight', 0.15),
            motion_weight=getattr(settings, 'ownership_motion_weight', 0.15),
            pose_feasibility_weight=getattr(settings, 'ownership_pose_feasibility_weight', 0.10),
            turn_prior_weight=getattr(settings, 'ownership_turn_prior_weight', 0.05),
            bst_weight=getattr(settings, 'ownership_bst_weight', 0.06),
            bst_alpha_threshold=getattr(settings, 'ownership_bst_alpha_threshold', 0.15),
            bst_conf_min=getattr(settings, 'ownership_bst_conf_min', 0.3),
            window_frames=getattr(settings, 'ownership_window_frames', 3),
            net_margin=getattr(settings, 'ownership_net_margin', 0.75),
            prox_sigma_norm=getattr(settings, 'ownership_prox_sigma_norm', 0.15),
            prox_sigma_meters=getattr(settings, 'ownership_prox_sigma_meters', 0.75),
            prox_min_pose_conf=getattr(settings, 'ownership_prox_min_pose_conf', 0.25),
            min_pose_conf=getattr(settings, 'ownership_min_pose_conf', 0.35),
            unknown_score=getattr(settings, 'ownership_unknown_score', 0.50),
            strong_reach_ratio=getattr(settings, 'ownership_strong_reach', 0.75),
            medium_reach_ratio=getattr(settings, 'ownership_medium_reach', 1.25),
            weak_reach_ratio=getattr(settings, 'ownership_weak_reach', 1.75),
            alternate_score=getattr(settings, 'ownership_alternate_score', 0.95),
            same_player_score=getattr(settings, 'ownership_same_player_score', 0.05),
            first_hit_score=getattr(settings, 'ownership_first_hit_score', 0.50),
            traj_min_shuttle_conf=getattr(settings, 'ownership_traj_min_shuttle_conf', 0.30),
            traj_interp_penalty=getattr(settings, 'ownership_traj_interp_penalty', 0.80),
            court_net_y=getattr(settings, 'ownership_court_net_y', 6.7),
            court_wrong_side_score=getattr(settings, 'ownership_court_wrong_side_score', 0.20),
            motion_wrist_weight=getattr(settings, 'ownership_motion_wrist_weight', 0.50),
            motion_elbow_weight=getattr(settings, 'ownership_motion_elbow_weight', 0.30),
            motion_shoulder_weight=getattr(settings, 'ownership_motion_shoulder_weight', 0.20),
            calib_near_mean=getattr(settings, 'calib_near_mean', 0.62),
            calib_near_std=getattr(settings, 'calib_near_std', 0.14),
            calib_far_mean=getattr(settings, 'calib_far_mean', 0.48),
            calib_far_std=getattr(settings, 'calib_far_std', 0.18),
        )

    def score(self,
              shuttle_df: pd.DataFrame,
              pose_df: pd.DataFrame | None,
              players_data: dict,
              court_data: dict,
              frame: int,
              near_id: str = "player_1",
              far_id: str = "player_2",
              prev_owner: str | None = None,
              shot: dict | None = None,
              ) -> dict:
        """Compute near/far ownership scores for a single candidate hit frame.

        Parameters
        ----------
        shuttle_df : pd.DataFrame
            Cleaned shuttle data with 'x', 'y', 'frame' columns.
        pose_df : pd.DataFrame | None
            Pose data with 'frame', 'player_id', 'keypoints' columns.
        players_data : dict
            'players' list with 'id', 'side', 'detections'.
        court_data : dict
            Must contain 'homography' (3×3 matrix).
        frame : int
            Candidate hit frame index.
        near_id, far_id : str
            Player IDs for near and far.
        prev_owner : str | None
            Player ID who hit the previous shot (for turn prior).

        Returns
        -------
        dict with keys:
            near_score, far_score: weighted final scores
            trajectory_near, trajectory_far
            court_side_near, court_side_far
            proximity_near, proximity_far
            motion_near, motion_far
            pose_near, pose_far
            turn_near, turn_far
        """
        result = {
            "near_score": self.unknown_score,
            "far_score": self.unknown_score,
        }

        # ── Lookup helper ──────────────────────────────────────
        def _shuttle_at(f: int) -> np.ndarray | None:
            rows = shuttle_df[shuttle_df["frame"] == f]
            if len(rows) == 0:
                return None
            r = rows.iloc[0]
            xv, yv = r["x"], r["y"]
            if pd.isna(xv) or pd.isna(yv):
                return None
            return np.array([float(xv), float(yv)])

        def _pose_at(f: int, pid: str) -> np.ndarray | None:
            if pose_df is None:
                return None
            rows = pose_df[(pose_df["frame"] == f) & (pose_df["player_id"] == pid)]
            if len(rows) == 0:
                return None
            raw = rows.iloc[0]["keypoints"]
            kps = np.array(raw.tolist()) if hasattr(raw, 'tolist') else np.array(raw)
            if kps.shape != (17, 3):
                return None
            return kps

        def _player_id(side: str) -> str:
            for p in players_data.get("players", []):
                if p.get("side") == side:
                    return p["id"]
            return near_id if side == "near" else far_id

        def _player_bbox_h(pid: str, f: int) -> float | None:
            for p in players_data.get("players", []):
                if p["id"] == pid:
                    for d in p.get("detections", []):
                        if d["frame"] == f:
                            return float(d["bbox"][3] - d["bbox"][1])
                    # Fallback: look ±5 frames
                    for d in p.get("detections", []):
                        if abs(d["frame"] - f) <= 5:
                            return float(d["bbox"][3] - d["bbox"][1])
            return None

        def _foot_px(pid: str, f: int) -> np.ndarray | None:
            # Try pose first, then bbox
            kps = _pose_at(f, pid)
            if kps is not None:
                foot = foot_midpoint_from_pose(kps[:, :2], kps[:, 2])
                if foot is not None:
                    return np.array(foot)
            for p in players_data.get("players", []):
                if p["id"] == pid:
                    for d in p.get("detections", []):
                        if d["frame"] == f:
                            return np.array([(d["bbox"][0] + d["bbox"][2]) / 2.0,
                                             float(d["bbox"][3])])
            return None

        # ── Shuttle position (pixel + court-space) ────────────
        w = self.window_frames
        shuttle_before_px = _shuttle_at(frame - w)
        shuttle_now_px = _shuttle_at(frame)
        shuttle_after_px = _shuttle_at(frame + w)
        if shuttle_after_px is None:
            shuttle_after_px = shuttle_now_px

        # Court-space shuttle positions for trajectory scoring
        H_mat = court_data.get("homography")
        H_arr = np.array(H_mat) if H_mat is not None else None
        def _to_court(px):
            if H_arr is None or px is None:
                return None
            try:
                return np.array(image_to_court(H_arr, px))
            except Exception:
                return None
        shuttle_before_court = _to_court(shuttle_before_px)
        shuttle_now_court = _to_court(shuttle_now_px)
        shuttle_after_court = _to_court(shuttle_after_px)

        # Court-space shuttle position (at hit frame, for court-side score)
        shuttle_court = shuttle_now_court

        # ── Player positions (pixel + court-space) ────────────
        def _foot_court(pid: str, f: int) -> np.ndarray | None:
            foot_px = _foot_px(pid, f)
            if foot_px is None or H_arr is None:
                return None
            try:
                return np.array(image_to_court(H_arr, foot_px))
            except Exception:
                return None

        near_id = _player_id("near")
        far_id = _player_id("far")
        near_foot_px = _foot_px(near_id, frame)
        far_foot_px = _foot_px(far_id, frame)
        near_foot_court = _foot_court(near_id, frame)
        far_foot_court = _foot_court(far_id, frame)

        near_kps = _pose_at(frame, near_id)
        far_kps = _pose_at(frame, far_id)

        near_bbox_h = _player_bbox_h(near_id, frame)
        far_bbox_h = _player_bbox_h(far_id, frame)

        # ── Motion: collect keypoint sequences ─────────────────
        def _kps_window(pid: str, center: int, half: int = 4) -> list[np.ndarray]:
            seq = []
            for f in range(center - half, center + half + 1):
                seq.append(_pose_at(f, pid))
            return seq

        near_kps_seq = _kps_window(near_id, frame)
        far_kps_seq = _kps_window(far_id, frame)

        hit_idx = half = 4  # index of the hit frame within the window

        # ── Compute sub-scores ─────────────────────────────────
        # Trajectory: court-space, per-player (document Section 10.1.1)
        traj_n = trajectory_ownership_score(
            shuttle_before_court, shuttle_now_court, shuttle_after_court,
            near_foot_court,
        )
        traj_f = trajectory_ownership_score(
            shuttle_before_court, shuttle_now_court, shuttle_after_court,
            far_foot_court,
        )
        # Normalise trajectory scores to sum to 1 so they behave as a
        # relative comparison (consistent with the other sub-scores).
        traj_total = traj_n + traj_f
        if traj_total > 0:
            traj_n /= traj_total
            traj_f /= traj_total

        # Apply interpolated-frame penalty when shuttle confidence is low
        def _shuttle_conf_at(f):
            rows = shuttle_df[shuttle_df["frame"] == f]
            return float(rows.iloc[0]["confidence"]) if len(rows) > 0 else 0.0
        if self.traj_interp_penalty < 1.0:
            w = self.window_frames
            confs = [_shuttle_conf_at(f) for f in (frame - w, frame, frame + w)]
            if any(c < self.traj_min_shuttle_conf for c in confs):
                traj_n *= self.traj_interp_penalty
                traj_f *= self.traj_interp_penalty

        court_n, court_f = court_side_feasibility_score(
            shuttle_court,
            net_y=self.court_net_y,
            margin=self.net_margin,
            wrong_side_score=self.court_wrong_side_score,
        )

        prox_n, prox_f = normalized_proximity_score(
            shuttle_now_px, shuttle_now_court,
            near_kps, far_kps, near_bbox_h, far_bbox_h,
            H_arr,
            sigma_norm=self.prox_sigma_norm,
            sigma_meters=self.prox_sigma_meters,
            min_pose_conf=self.prox_min_pose_conf,
            unknown_score=self.unknown_score,
        )

        mot_n, mot_f = racket_motion_score(
            near_kps_seq, far_kps_seq, hit_idx,
            wrist_weight=self.motion_wrist_weight,
            elbow_weight=self.motion_elbow_weight,
            shoulder_weight=self.motion_shoulder_weight,
            min_confidence=self.min_pose_conf,
            unknown_score=self.unknown_score,
        )

        pose_n, pose_f = pose_contact_feasibility_score(
            shuttle_now_px, near_kps, far_kps,
            strong_reach_ratio=self.strong_reach_ratio,
            medium_reach_ratio=self.medium_reach_ratio,
            weak_reach_ratio=self.weak_reach_ratio,
            min_confidence=self.min_pose_conf,
            unknown_score=self.unknown_score,
        )

        turn_n, turn_f = initial_turn_prior_score(
            prev_owner, near_id=near_id, far_id=far_id,
            alternate_score=self.alternate_score,
            same_player_score=self.same_player_score,
            first_hit_score=self.first_hit_score,
        )

        bst_n, bst_f = bst_attribution_score(
            shot,
            alpha_threshold=self.bst_alpha_threshold,
            conf_min=self.bst_conf_min,
            unknown_score=self.unknown_score,
        )

        # ── Weighted combination (normalised to sum 1.0) ───────
        w_traj = self.trajectory_weight
        w_court = self.court_side_weight
        w_prox = self.proximity_weight
        w_mot = self.motion_weight
        w_pose = self.pose_feasibility_weight
        w_turn = self.turn_prior_weight
        w_bst = self.bst_weight
        total_w = w_traj + w_court + w_prox + w_mot + w_pose + w_turn + w_bst
        if total_w > 0:
            w_traj /= total_w
            w_court /= total_w
            w_prox /= total_w
            w_mot /= total_w
            w_pose /= total_w
            w_turn /= total_w
            w_bst /= total_w
        else:
            w_traj = w_court = w_prox = w_mot = w_pose = w_turn = w_bst = 1.0 / 7.0

        near_score = (
            w_traj * traj_n +
            w_court * court_n +
            w_prox * prox_n +
            w_mot * mot_n +
            w_pose * pose_n +
            w_turn * turn_n +
            w_bst * bst_n
        )
        far_score = (
            w_traj * traj_f +
            w_court * court_f +
            w_prox * prox_f +
            w_mot * mot_f +
            w_pose * pose_f +
            w_turn * turn_f +
            w_bst * bst_f
        )

        # Side-specific z-score calibration (spec §18)
        near_z = (near_score - self.calib_near_mean) / max(self.calib_near_std, 1e-6)
        far_z = (far_score - self.calib_far_mean) / max(self.calib_far_std, 1e-6)
        near_prob = float(1.0 / (1.0 + np.exp(-near_z)))
        far_prob = float(1.0 / (1.0 + np.exp(-far_z)))
        total_cal = near_prob + far_prob
        if total_cal > 0:
            near_score = near_prob / total_cal
            far_score = far_prob / total_cal

        result.update({
            "near_score": round(near_score, 4),
            "far_score": round(far_score, 4),
            "trajectory_near": round(traj_n, 4),
            "trajectory_far": round(traj_f, 4),
            "court_side_near": round(court_n, 4),
            "court_side_far": round(court_f, 4),
            "proximity_near": round(prox_n, 4),
            "proximity_far": round(prox_f, 4),
            "motion_near": round(mot_n, 4),
            "motion_far": round(mot_f, 4),
            "pose_near": round(pose_n, 4),
            "pose_far": round(pose_f, 4),
            "turn_near": round(turn_n, 4),
            "turn_far": round(turn_f, 4),
            "bst_near": round(bst_n, 4),
            "bst_far": round(bst_f, 4),
        })
        return result
