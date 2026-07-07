import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings

# COCO-17 keypoint indices
_R_WRIST = 10
_L_WRIST = 9
_R_ELBOW = 8
_L_ELBOW = 7


# ── Global-hit-candidate detector (shuttle-centric, no player dependency) ──

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (radians) between two 2D vectors."""
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(dot / norm, -1.0, 1.0)))


def _menger_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Menger curvature for three consecutive points = 4*area / (|AB|*|BC|*|CA|)."""
    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ca = np.linalg.norm(c - a)
    if ab < 1e-6 or bc < 1e-6 or ca < 1e-6:
        return 0.0
    area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
    return 4.0 * area / (ab * bc * ca)


def _normalize_by_p95(values: np.ndarray) -> np.ndarray:
    """95th-percentile normalisation, robust to extreme frames."""
    m = float(np.percentile(values[np.isfinite(values)], 95)) if np.any(np.isfinite(values)) else 1.0
    return values / (m + 1e-6) if m > 0 else values


def non_max_suppression(candidates: list[dict], min_gap: int) -> list[dict]:
    """Greedy non-maximum suppression: keep highest-score candidate within each
    min_gap-sized window, preserving the strongest detections."""
    if len(candidates) < 2:
        return candidates
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    suppressed = set()
    for i, c in enumerate(sorted_cands):
        if i in suppressed:
            continue
        kept.append(c)
        for j in range(i + 1, len(sorted_cands)):
            if abs(c["frame"] - sorted_cands[j]["frame"]) < min_gap:
                suppressed.add(j)
    return sorted(kept, key=lambda c: c["frame"])


class GlobalHitCandidateDetector:
    """Detect hit candidates from shuttle trajectory alone.

    Five-signal fusion: direction change, speed delta, curvature,
    visibility transition, TrackNet confidence dip.  The dip signal
    pinpoints shuttle occlusion (racket covering shuttle at contact),
    which the trajectory-based signals miss by 1-8 frames.

    No player-dependent signals (no proximity gate, no swing acceleration).
    Candidate scoring is purely shuttle-centric.
    """

    def __init__(self, window: int = 3, direction_weight: float = 0.30,
                 speed_weight: float = 0.30, curvature_weight: float = 0.20,
                 visibility_weight: float = 0.05, dip_weight: float = 0.15,
                 threshold: float = 0.62, min_gap_frames: int = 6):
        self.window = window
        self.direction_weight = direction_weight
        self.speed_weight = speed_weight
        self.curvature_weight = curvature_weight
        self.visibility_weight = visibility_weight
        self.dip_weight = dip_weight
        self.threshold = threshold
        self.min_gap_frames = min_gap_frames

    @classmethod
    def from_settings(cls) -> "GlobalHitCandidateDetector":
        return cls(
            window=getattr(settings, 'hit_window_frames', 3),
            direction_weight=getattr(settings, 'hit_direction_weight', 0.30),
            speed_weight=getattr(settings, 'hit_speed_weight', 0.30),
            curvature_weight=getattr(settings, 'hit_curvature_weight', 0.20),
            visibility_weight=getattr(settings, 'hit_visibility_weight', 0.05),
            dip_weight=getattr(settings, 'hit_dip_weight', 0.15),
            threshold=getattr(settings, 'hit_candidate_threshold', 0.62),
            min_gap_frames=getattr(settings, 'hit_min_gap_frames', 6),
        )

    def detect(self, shuttle_track: pd.DataFrame) -> list[dict]:
        """Detect hit candidates from cleaned shuttle trajectory.

        Parameters
        ----------
        shuttle_track : pd.DataFrame
            Must contain columns ``x``, ``y`` (pixel coords, NaN for missing
            frames).  May contain ``was_interpolated``; interpolated regions
            are down-weighted via the visibility score.

        Returns
        -------
        candidates : list[dict]
            Each dict has keys ``frame``, ``score``, ``direction_change``,
            ``speed_delta``, ``curvature`` (raw sub-scores before
            weighting) and ``visibility_transition``.
        """
        x = shuttle_track["x"].values.astype(np.float64)
        y = shuttle_track["y"].values.astype(np.float64)
        n = len(x)
        w = self.window

        # Fill NaNs with forward-fill so diff/angle computations don't
        # collapse.  Track original NaN locations for the visibility score.
        orig_nan = np.isnan(x)
        x_filled = pd.Series(x).ffill().bfill().values
        y_filled = pd.Series(y).ffill().bfill().values

        # Interpolated flag for visibility transition scoring
        is_interpolated = orig_nan.copy()
        if "was_interpolated" in shuttle_track.columns:
            is_interpolated = is_interpolated | shuttle_track["was_interpolated"].values.astype(bool)

        # Velocity before and after each frame (vector from t-w to t and t to t+w)
        v_before = np.zeros((n, 2), dtype=np.float64)
        v_after = np.zeros((n, 2), dtype=np.float64)

        for t in range(w, n - w):
            v_before[t] = [x_filled[t] - x_filled[t - w],
                           y_filled[t] - y_filled[t - w]]
            v_after[t] = [x_filled[t + w] - x_filled[t],
                          y_filled[t + w] - y_filled[t]]

        speed_before = np.linalg.norm(v_before, axis=1)
        speed_after = np.linalg.norm(v_after, axis=1)

        # 1. Direction change: angle between v_before and v_after
        direction_signal = np.zeros(n, dtype=np.float64)
        for t in range(w, n - w):
            direction_signal[t] = _angle_between(v_before[t], v_after[t])

        # 2. Speed delta: absolute change in speed
        speed_delta_signal = np.abs(speed_after - speed_before)

        # 3. Menger curvature
        curvature_signal = np.zeros(n, dtype=np.float64)
        for t in range(1, n - 1):
            if any(orig_nan[t - 1:t + 2]):
                continue
            curvature_signal[t] = _menger_curvature(
                np.array([x_filled[t - 1], y_filled[t - 1]]),
                np.array([x_filled[t], y_filled[t]]),
                np.array([x_filled[t + 1], y_filled[t + 1]]),
            )

        # 4. Visibility transition: shuttle appears/disappears (occlusion)
        visibility_signal = np.zeros(n, dtype=np.float64)
        vis_changes = np.diff((~orig_nan).astype(int))
        change_frames = np.where(np.abs(vis_changes) > 0)[0] + 1
        visibility_signal[change_frames] = 1.0

        # 5. TrackNet confidence dip: short-lived occlusion at contact frame.
        #    When the racket covers the shuttle, TrackNet confidence drops
        #    for 1-3 frames before recovering.  This dip precedes the
        #    trajectory inflection point (which lags by 1-8 frames).
        dip_signal = np.zeros(n, dtype=np.float64)
        if "confidence" in shuttle_track.columns:
            confidence = shuttle_track["confidence"].values.astype(np.float64)
            conf_norm = confidence / (np.nanmax(confidence) + 1e-6)
            for t in range(2, n - 2):
                has_real_data = np.sum(~orig_nan[t - 2:t + 3]) >= 4
                is_dip = conf_norm[t] < 0.3 and \
                         np.nanmean(conf_norm[t - 2:t]) > 0.4 and \
                         np.nanmean(conf_norm[t + 1:t + 3]) > 0.4
                if has_real_data and is_dip:
                    dip_signal[t] = 1.0

        # Normalize each signal
        direction_norm = _normalize_by_p95(direction_signal)
        speed_delta_norm = _normalize_by_p95(speed_delta_signal)
        curvature_norm = _normalize_by_p95(curvature_signal)

        # Combined event score
        combined = (
            self.direction_weight * direction_norm +
            self.speed_weight * speed_delta_norm +
            self.curvature_weight * curvature_norm +
            self.visibility_weight * visibility_signal +
            self.dip_weight * dip_signal
        )

        # Build candidates
        candidates = []
        for t in range(w, n - w):
            if not np.isfinite(combined[t]):
                continue
            if combined[t] >= self.threshold:
                candidates.append({
                    "frame": t,
                    "score": float(combined[t]),
                    "direction_change": float(direction_signal[t]),
                    "speed_delta": float(speed_delta_signal[t]),
                    "curvature": float(curvature_signal[t]),
                    "visibility_transition": int(visibility_signal[t]),
                    "dip_score": float(dip_signal[t]),
                })

        # Suppress scene-cut induced teleport false positives:
        # when shuttle displacement exceeds 10× the median, suppress nearby frames.
        med_disp = float(np.median(
            np.sqrt(np.diff(x_filled) ** 2 + np.diff(y_filled) ** 2)
        ))
        if np.isfinite(med_disp) and med_disp > 1.0:
            disp = np.sqrt(
                np.diff(x_filled, prepend=x_filled[0]) ** 2 +
                np.diff(y_filled, prepend=y_filled[0]) ** 2
            )
            cut_frames = np.where(disp > 10 * med_disp)[0]
            candidates = [
                c for c in candidates
                if not any(abs(c["frame"] - cf) <= 2 for cf in cut_frames)
            ]

        # Non-maximum suppression
        return non_max_suppression(candidates, self.min_gap_frames)


def _direction_reversal_angle(
    pos_before: np.ndarray, pos_after: np.ndarray,
) -> float:
    """Angle (radians) between before and after velocity vectors.
    Near π (180°) = strong direction reversal = likely contact frame.
    """
    n_before = np.linalg.norm(pos_before)
    n_after = np.linalg.norm(pos_after)
    if n_before < 1e-6 or n_after < 1e-6:
        return 0.0
    dot = float(np.dot(pos_before, pos_after))
    cos_angle = np.clip(dot / (n_before * n_after), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _find_nearest_wrist_frame(
    candidate_frame: int,
    pose_df: pd.DataFrame,
    shuttle_df: pd.DataFrame,
    search_window: int = 8,
    min_shuttle_conf: float = 0.20,
) -> int:
    """Refine a hit candidate using shuttle direction reversal + wrist proximity.

    **Backward-only search** — the pipeline systematically lags the true
    contact frame (95/99 enriched labels have pipeline-after-label).  The
    refinement only searches frames [candidate - search_window, candidate].

    Primary signal: shuttle direction reversal angle (near 180° = contact).
    Tiebreaker: wrist-to-shuttle distance across all detected players.

    Args:
        candidate_frame: Initial hit frame from shuttle trajectory.
        pose_df: DataFrame with 'frame', 'player_id', 'keypoints' columns.
        shuttle_df: DataFrame with 'frame', 'x', 'y', 'confidence' columns.
        search_window: frames to search BACKWARD from candidate_frame.
        min_shuttle_conf: Minimum shuttle detection confidence.

    Returns:
        Refined frame, or candidate_frame if refinement fails.
    """
    # Backward-only search range.  Pre-extract extra trajectory ahead for
    # v_after computation at the search boundary.
    lo = max(0, candidate_frame - search_window)
    hi = candidate_frame + 1  # backward-only — never search forward

    traj = []
    for f in range(lo - search_window, hi + search_window):
        srows = shuttle_df[shuttle_df["frame"] == f]
        if len(srows) > 0:
            sx = float(srows.iloc[0].get("x", np.nan))
            sy = float(srows.iloc[0].get("y", np.nan))
            if np.isfinite(sx) and np.isfinite(sy):
                traj.append((f, sx, sy))
    if len(traj) < 4:
        return candidate_frame
    traj_frames, traj_x, traj_y = zip(*traj)
    traj_x = np.array(traj_x, dtype=np.float64)
    traj_y = np.array(traj_y, dtype=np.float64)
    traj_frames = np.array(traj_frames, dtype=np.int64)

    def _pos_at(f: int) -> np.ndarray | None:
        """Interpolate shuttle position at frame f from trajectory."""
        if f < traj_frames[0] or f > traj_frames[-1]:
            return None
        x = np.interp(f, traj_frames, traj_x)
        y = np.interp(f, traj_frames, traj_y)
        return np.array([x, y])

    # Compute direction reversal score for each candidate frame
    direction_frames: dict[int, float] = {}
    for f in range(lo, hi):
        pos_before = _pos_at(f - search_window)
        pos_now = _pos_at(f)
        pos_after = _pos_at(f + search_window)
        if pos_before is None or pos_now is None or pos_after is None:
            continue
        v_in = pos_now - pos_before
        v_out = pos_after - pos_now
        angle = _direction_reversal_angle(v_in, v_out)
        direction_frames[f] = angle  # near π = strong reversal

    if not direction_frames:
        return candidate_frame

    best_frame = candidate_frame
    best_score: float | None = None

    for f in range(lo, hi):
        # Shuttle must be detected at this frame
        srows = shuttle_df[
            (shuttle_df["frame"] == f) &
            (shuttle_df.get("confidence", pd.Series([1.0]) * len(shuttle_df)) >= min_shuttle_conf)
        ]
        if len(srows) == 0:
            srows = shuttle_df[
                (shuttle_df["frame"] == f) &
                (shuttle_df["x"].notna()) & (shuttle_df["y"].notna())
            ]
        if len(srows) == 0:
            continue
        sx, sy = float(srows.iloc[0]["x"]), float(srows.iloc[0]["y"])

        prows = pose_df[pose_df["frame"] == f]
        if len(prows) == 0:
            continue

        # Compute wrist proximity score: min distance across all wrists
        min_wrist_dist: float | None = None
        for _, prow in prows.iterrows():
            raw = prow["keypoints"]
            kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
            if kps.ndim != 2 or kps.shape[0] < 11 or kps.shape[1] < 2:
                continue
            for wrist_idx in (9, 10):
                wx, wy = float(kps[wrist_idx, 0]), float(kps[wrist_idx, 1])
                wconf = float(kps[wrist_idx, 2]) if kps.shape[1] >= 3 else 1.0
                if wconf < 0.3:
                    continue
                dist = float(np.sqrt((wx - sx) ** 2 + (wy - sy) ** 2))
                if min_wrist_dist is None or dist < min_wrist_dist:
                    min_wrist_dist = dist
        if min_wrist_dist is None:
            continue

        # Combined score: direction reversal (weighted higher) + wrist proximity
        rev_angle = direction_frames.get(f, 0.0)
        # Normalize reversal angle: π → 1.0, 0 → 0.0
        rev_score = min(rev_angle / np.pi, 1.0)
        # Normalize wrist distance: 0px → 1.0, large → 0 (exp decay)
        wrist_score = float(np.exp(-min_wrist_dist / 100.0))

        score = 0.7 * rev_score + 0.3 * wrist_score
        if best_score is None or score > best_score:
            best_score = score
            best_frame = f

    return best_frame


# ── Wrist-speed hit detector (pose-only fallback) ──────────────────────
# Adapted from Haimantika/badminton-coach: detect strikes from racket-wrist
# speed peaks.  Runs when shuttle-based detection finds too few candidates.

def _wrist_speed_series(wrist_xy: np.ndarray, window: int = 3) -> np.ndarray:
    """Compute smoothed wrist speed (normalized px/frame)."""
    if len(wrist_xy) < window + 2:
        return np.zeros(len(wrist_xy))
    cum = np.cumsum(np.insert(wrist_xy, 0, 0, axis=0), axis=0)
    smoothed = (cum[window:] - cum[:-window]) / window
    # Pad to original length
    pad = np.full((window // 2, 2), np.nan)
    smoothed = np.vstack([pad, smoothed, pad[:max(0, len(wrist_xy) - len(smoothed) - len(pad))]]) if len(smoothed) < len(wrist_xy) else smoothed
    smoothed = smoothed[:len(wrist_xy)]
    diffs = np.diff(smoothed, axis=0)
    speed = np.sqrt(np.sum(diffs ** 2, axis=1))
    speed = np.nan_to_num(speed, nan=0.0)
    return np.concatenate([[0.0], speed])


def _detect_peaks_1d(values: np.ndarray, min_height: float, min_distance: int) -> list[int]:
    """Simple 1D peak detection. Returns indices of peaks."""
    if len(values) < 3:
        return []
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1] and values[i] >= min_height:
            peaks.append(i)
    # Enforce min_distance
    if not peaks:
        return []
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
    return filtered


def _detect_wrist_peaks(pose_df: pd.DataFrame, fps: float,
                         min_speed: float, min_interval_s: float,
                         min_kp_conf: float = 0.30) -> list[dict]:
    """Detect hit candidates from wrist-speed peaks in pose data.

    Returns list of dicts with 'frame' and 'score' keys, sorted by frame.
    Adapted from Haimantika/badminton-coach's detect_strike_frames.
    """
    if pose_df is None or len(pose_df) < 5:
        return []

    candidates = []
    min_dist = max(1, int(round(min_interval_s * fps)))

    for player_id in pose_df["player_id"].unique():
        player = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
        frames = player["frame"].values
        if len(frames) < 3:
            continue

        # Extract both wrists — check which has higher peak speed (racket hand)
        r_wrist = np.full((len(player), 2), np.nan)
        l_wrist = np.full((len(player), 2), np.nan)
        for i, (_, row) in enumerate(player.iterrows()):
            raw = row["keypoints"]
            kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
            if kps.shape != (17, 3):
                continue
            if kps[_R_WRIST, 2] >= min_kp_conf:
                r_wrist[i] = kps[_R_WRIST, :2]
            if kps[_L_WRIST, 2] >= min_kp_conf:
                l_wrist[i] = kps[_L_WRIST, :2]

        # Use the wrist with higher max speed (racket hand detection)
        r_speed = _wrist_speed_series(r_wrist)
        l_speed = _wrist_speed_series(l_wrist)
        r_max = float(np.max(r_speed))
        l_max = float(np.max(l_speed))

        if max(r_max, l_max) < min_speed:
            continue

        wrist_speed = r_speed if r_max >= l_max else l_speed
        peaks = _detect_peaks_1d(wrist_speed, min_height=min_speed, min_distance=min_dist)

        for p in peaks:
            if p < len(frames):
                candidates.append({
                    "frame": int(frames[p]),
                    "score": float(wrist_speed[p]),
                    "source": "wrist_peak",
                })

    # Sort by frame, merge close duplicates (NMS)
    candidates.sort(key=lambda c: c["frame"])
    return candidates


class HitFrameLocalizationStage:
    name = "hit_frame_localization"
    input_keys = ["shuttle", "pose"]
    output_keys = ["hits"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shuttle_df = artifacts.get_parquet("shuttle")
        if shuttle_df is None or len(shuttle_df) == 0:
            shuttle_df = artifacts.get_parquet("shuttle_raw")
            if shuttle_df is None or len(shuttle_df) == 0:
                return StageResult.from_error("Shuttle tracking data required")
            shuttle_df["x"] = shuttle_df["x"].ffill().bfill()
            shuttle_df["y"] = shuttle_df["y"].ffill().bfill()

        pose_df = artifacts.get_parquet("pose")

        detector = GlobalHitCandidateDetector.from_settings()
        candidates = detector.detect(shuttle_df)

        # Phase 2: pose-based contact refinement — backward-only search.
        # Pipeline lags the true contact frame (95/99 enriched labels are
        # pipeline-after-label), so we only search backward from each candidate.
        refine_window = getattr(settings, "hit_refine_window", 8)
        refined_count = 0
        if pose_df is not None and len(pose_df) > 0 and refine_window > 0:
            for c in candidates:
                orig_frame = c["frame"]
                refined = _find_nearest_wrist_frame(
                    orig_frame, pose_df, shuttle_df,
                    search_window=refine_window,
                    min_shuttle_conf=getattr(settings, "shuttle_min_conf", 0.30),
                )
                if refined != orig_frame:
                    c["frame"] = refined
                    c["_refined_offset"] = refined - orig_frame
                    refined_count += 1

        if refined_count > 0:
            # Re-run NMS after refinement (shifted frames may now collide)
            candidates = non_max_suppression(candidates, detector.min_gap_frames)
            logger.info("Refined hit frames via wrist-to-shuttle proximity",
                        refined_count=refined_count, total=len(candidates), window=refine_window)

        # Phase 3: wrist-speed fallback — catch hits missed by shuttle detector
        # Adapted from Haimantika/badminton-coach: purely kinematic strike detection
        wrist_hits = []
        if getattr(settings, "wrist_hit_enabled", True) and pose_df is not None and len(pose_df) > 0:
            fps = float(config.processing_fps or getattr(settings, "fps", 30.0))
            wrist_hits = _detect_wrist_peaks(
                pose_df, fps,
                min_speed=getattr(settings, "wrist_hit_min_speed", 0.15),
                min_interval_s=getattr(settings, "wrist_hit_min_interval_s", 0.3),
                min_kp_conf=getattr(settings, "wrist_hit_min_conf", 0.30),
            )

        if wrist_hits:
            # Merge wrist peaks with shuttle candidates
            wrist_weight = getattr(settings, "wrist_hit_score_weight", 0.40)
            candidate_frames = {c["frame"] for c in candidates}
            for w in wrist_hits:
                w["score"] *= wrist_weight
                if w["frame"] not in candidate_frames:
                    candidates.append(w)
            # Re-run NMS on merged list
            candidates = non_max_suppression(candidates, detector.min_gap_frames)
            logger.info("Wrist-speed fallback added hits",
                        wrist_count=len(wrist_hits), merged_total=len(candidates))

        # Write debug hit scores if debug_level >= 2
        if config.debug_level >= 2:
            n = len(shuttle_df)
            debug_hit_df = pd.DataFrame({
                "frame": shuttle_df["frame"].values if "frame" in shuttle_df.columns else range(n),
                "combined": np.zeros(n, dtype=np.float64),
                "is_peak": False,
                "_placeholder": np.zeros(n),
            })
            debug_frames = [c["frame"] for c in candidates]
            debug_scores = [c["score"] for c in candidates]
            debug_hit_df.loc[debug_frames, "combined"] = debug_scores
            debug_hit_df.loc[debug_frames, "is_peak"] = True
            artifacts.set_parquet("debug_hit_scores", debug_hit_df)

        hits = [{
            "frame": c["frame"],
            "confidence": c["score"],
        } | ({"_refined_offset": c["_refined_offset"]} if "_refined_offset" in c else {})
                for c in candidates]
        hits_data = pd.DataFrame(hits)
        artifacts.set_parquet("hits", hits_data)

        logger.info("Localized hit frames", count=len(hits))

        return StageResult.success(
            artifacts={"hits": artifacts.path("hits")},
            metadata={
                "hit_count": len(hits),
                "frames_analyzed": len(shuttle_df),
                "refined_count": refined_count,
            }
        )
