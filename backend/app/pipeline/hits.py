import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings

# COCO-17 keypoint indices (used by wrist-speed detector)
_R_WRIST = 10
_L_WRIST = 9


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

    Four-signal fusion: direction change, speed delta, curvature,
    visibility transition.  No player-dependent signals (no proximity gate,
    no swing acceleration).  Candidate scoring is purely shuttle-centric.
    """

    def __init__(self, window: int = 3, direction_weight: float = 0.45,
                 speed_weight: float = 0.30, curvature_weight: float = 0.20,
                 visibility_weight: float = 0.05, threshold: float = 0.62,
                 min_gap_frames: int = 6):
        self.window = window
        self.direction_weight = direction_weight
        self.speed_weight = speed_weight
        self.curvature_weight = curvature_weight
        self.visibility_weight = visibility_weight
        self.threshold = threshold
        self.min_gap_frames = min_gap_frames

    @classmethod
    def from_settings(cls) -> "GlobalHitCandidateDetector":
        return cls(
            window=getattr(settings, 'hit_window_frames', 3),
            direction_weight=getattr(settings, 'hit_direction_weight', 0.45),
            speed_weight=getattr(settings, 'hit_speed_weight', 0.30),
            curvature_weight=getattr(settings, 'hit_curvature_weight', 0.20),
            visibility_weight=getattr(settings, 'hit_visibility_weight', 0.05),
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

        # 4. Visibility transition: shuttle appears/disappears (occlusion)
        visibility_signal = np.zeros(n, dtype=np.float64)
        vis_changes = np.diff((~orig_nan).astype(int))
        change_frames = np.where(np.abs(vis_changes) > 0)[0] + 1
        visibility_signal[change_frames] = 1.0

        # Normalize each signal
        direction_norm = _normalize_by_p95(direction_signal)
        speed_delta_norm = _normalize_by_p95(speed_delta_signal)
        curvature_norm = _normalize_by_p95(curvature_signal)

        # Combined event score
        combined = (
            self.direction_weight * direction_norm +
            self.speed_weight * speed_delta_norm +
            self.curvature_weight * curvature_norm +
            self.visibility_weight * visibility_signal
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
    search_window: int = 4,
    min_shuttle_conf: float = 0.20,
) -> int:
    """Refine a hit candidate using shuttle direction reversal + wrist proximity.

    Primary signal: shuttle direction reversal angle (near 180° = contact).
    Tiebreaker: wrist-to-shuttle distance across all detected players.

    Args:
        candidate_frame: Initial hit frame from shuttle trajectory.
        pose_df: DataFrame with 'frame', 'player_id', 'keypoints' columns.
        shuttle_df: DataFrame with 'frame', 'x', 'y', 'confidence' columns.
        search_window: ±frames to search around candidate_frame.
        min_shuttle_conf: Minimum shuttle detection confidence.

    Returns:
        Refined frame, or candidate_frame if refinement fails.
    """
    # Backward-only search range.  Pre-extract extra trajectory ahead for
    # v_after computation at the search boundary.
    lo = max(0, candidate_frame - search_window)
    hi = candidate_frame + search_window + 1

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
                         min_kp_conf: float = 0.30,
                         adaptive_frac: float = 0.0) -> list[dict]:
    """Detect hit candidates from wrist-speed peaks in pose data.

    When ``adaptive_frac > 0``, the per-player peak threshold is
    ``max(min_speed, max_wrist_speed * adaptive_frac)``, adapting to
    each player's natural speed range.

    Returns list of dicts with keys ``frame``, ``score``, ``source``,
    ``dominant_hand`` (``"right"`` or ``"left"``), sorted by frame.
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

        # Extract both wrists
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

        r_speed = _wrist_speed_series(r_wrist)
        l_speed = _wrist_speed_series(l_wrist)
        r_max = float(np.max(r_speed))
        l_max = float(np.max(l_speed))

        if max(r_max, l_max) < min_speed:
            continue

        # Adaptive threshold: fraction of player's max wrist speed
        effective_threshold = min_speed
        if adaptive_frac > 0 and max(r_max, l_max) > 0:
            effective_threshold = max(min_speed, max(r_max, l_max) * adaptive_frac)

        dominant = "right" if r_max >= l_max else "left"
        wrist_speed = r_speed if r_max >= l_max else l_speed
        peaks = _detect_peaks_1d(wrist_speed, min_height=effective_threshold,
                                 min_distance=min_dist)

        for p in peaks:
            if p < len(frames):
                candidates.append({
                    "frame": int(frames[p]),
                    "score": float(wrist_speed[p]),
                    "source": "wrist_peak",
                    "dominant_hand": dominant,
                })

    candidates.sort(key=lambda c: c["frame"])
    return candidates


# ── Audio-visual fusion hit detector ────────────────────────────────
# Adapted from Ryan-z-Feng-ccsf/badminton-coach: audio onset detection
# cross-validated with wrist velocity peaks.  Audio provides high precision
# (shuttle-hit "pop" sounds are distinctive); wrist velocity provides high
# recall (95% of hits within ±4 frames).  Fusion gives both.

def _extract_audio_ffmpeg(video_path: str, sr: int = 22050) -> str:
    """Extract audio to a temporary mono WAV file via ffmpeg.
    Returns the temp-file path, or raises on failure.
    """
    import subprocess, tempfile
    audio_path = tempfile.mktemp(suffix=".wav")
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
         "-y", audio_path],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode(errors='replace')[:200]}")
    return audio_path


def _detect_onset_peaks(audio_path: str, fps: float, sr: int = 22050,
                         delta: float = 0.5, wait: int = 6,
                         min_gap_frames: int = 6) -> list[dict]:
    """Detect onset peaks from a WAV audio file using librosa.

    Returns list[dict] with keys ``frame``, ``score`` (normalised onset
    strength), ``source`` (= ``"audio"``), sorted by frame and filtered
    by ``min_gap_frames``.
    """
    import librosa
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3, post_max=3,
        pre_avg=3, post_avg=5,
        delta=delta, wait=wait,
    )
    if len(peaks) == 0:
        return []

    times = librosa.frames_to_time(peaks, sr=sr)
    frames = [int(round(t * fps)) for t in times]
    scores = onset_env[peaks]
    smax = float(np.max(scores))
    if smax > 0:
        scores = scores / smax

    candidates = []
    prev = -min_gap_frames
    for f, s in sorted(zip(frames, scores)):
        if f - prev >= min_gap_frames:
            candidates.append({"frame": f, "score": float(s), "source": "audio"})
            prev = f
    return candidates


def _fuse_audio_wrist(audio_candidates: list[dict],
                      wrist_candidates: list[dict],
                      tolerance: int = 2,
                      min_gap_frames: int = 6) -> list[dict]:
    """Cross-validate audio onset peaks with wrist velocity peaks.

    Fusion strategy (highest to lowest confidence):
      1. **audio_visual_fusion** — audio peak with matching wrist peak
         within ``tolerance`` frames → score = 1.0.
      2. **audio_only** — audio peak without wrist match → score = 0.8
         (audio alone is still high precision).
      3. **wrist_only** — wrist peak without matching audio peak
         → score = 0.5 × original wrist score (might be hits audio missed).

    Returns fused list sorted by frame with minimum-gap enforcement.
    """
    if not audio_candidates:
        return list(wrist_candidates)
    if not wrist_candidates:
        return list(audio_candidates)

    wrist_frames = {w["frame"] for w in wrist_candidates}

    fused: list[dict] = []
    audio_frames: set[int] = set()

    # Build a lookup for wrist candidates with extra metadata
    wrist_by_frame: dict[int, dict] = {w["frame"]: w for w in wrist_candidates}

    for ac in audio_candidates:
        af = ac["frame"]
        audio_frames.add(af)
        matching_wf = None
        for wf in wrist_frames:
            if abs(af - wf) <= tolerance:
                matching_wf = wf
                break
        if matching_wf is not None:
            entry: dict = {"frame": af, "score": 1.0, "source": "audio_visual_fusion"}
            wc = wrist_by_frame.get(matching_wf, {})
            if "dominant_hand" in wc:
                entry["dominant_hand"] = wc["dominant_hand"]
            fused.append(entry)
        else:
            fused.append({"frame": af, "score": 0.8, "source": "audio_only"})

    for wc in wrist_candidates:
        wf = wc["frame"]
        if not any(abs(wf - af) <= tolerance for af in audio_frames):
            entry: dict = {
                "frame": wf,
                "score": 0.5 * wc["score"],
                "source": "wrist_only",
            }
            if "dominant_hand" in wc:
                entry["dominant_hand"] = wc["dominant_hand"]
            fused.append(entry)

    # Remove remaining collisions via min-gap (prefer higher score)
    fused.sort(key=lambda c: (-c["score"], c["frame"]))
    result = []
    suppressed: set[int] = set()
    for i, c in enumerate(fused):
        if i in suppressed:
            continue
        result.append(c)
        for j in range(i + 1, len(fused)):
            if abs(c["frame"] - fused[j]["frame"]) < min_gap_frames:
                suppressed.add(j)
    result.sort(key=lambda c: c["frame"])
    return result


class AudioFusionDetector:
    """Detect hit candidates by fusing audio onset + wrist-velocity peaks.

    Uses ``ffmpeg`` for audio extraction and ``librosa`` for onset
    detection.  Falls back to wrist-only when audio is unavailable or
    extraction fails.
    """

    def __init__(self, fps: float = 30.0, onset_delta: float = 0.5,
                 onset_wait: int = 6, fusion_tolerance: int = 2,
                 min_gap_frames: int = 6):
        self.fps = fps
        self.onset_delta = onset_delta
        self.onset_wait = onset_wait
        self.fusion_tolerance = fusion_tolerance
        self.min_gap_frames = min_gap_frames

    @classmethod
    def from_settings(cls) -> "AudioFusionDetector":
        return cls(
            fps=float(getattr(settings, "fps", 30.0)),
            onset_delta=getattr(settings, "audio_onset_delta", 0.5),
            onset_wait=getattr(settings, "audio_onset_wait", 6),
            fusion_tolerance=getattr(settings, "audio_fusion_tolerance", 2),
            min_gap_frames=getattr(settings, "hit_min_gap_frames", 6),
        )

    def detect(self, video_path: str, pose_df: pd.DataFrame | None,
               fps: float | None = None) -> list[dict]:
        """Fuse audio onset peaks with wrist velocity peaks.

        Parameters
        ----------
        video_path : str
            Path to video file with audio track.
        pose_df : pd.DataFrame or None
            Pose data for wrist-speed detection (may be None).
        fps : float, optional
            Processing FPS (defaults to ``self.fps``).

        Returns
        -------
        list[dict]
            Each dict has ``frame``, ``score``, ``source`` keys.
            Empty list if everything fails.
        """
        if fps is None:
            fps = self.fps

        wrist_candidates = []
        if pose_df is not None and len(pose_df) > 0:
            wrist_candidates = _detect_wrist_peaks(
                pose_df, fps,
                min_speed=getattr(settings, "wrist_hit_min_speed", 0.15),
                min_interval_s=getattr(settings, "wrist_hit_min_interval_s", 0.3),
                min_kp_conf=getattr(settings, "wrist_hit_min_conf", 0.30),
                adaptive_frac=getattr(settings, "wrist_hit_adaptive_frac", 0.40),
            )

        audio_candidates = []
        try:
            audio_path = _extract_audio_ffmpeg(video_path)
            audio_candidates = _detect_onset_peaks(
                audio_path, fps,
                delta=self.onset_delta, wait=self.onset_wait,
                min_gap_frames=self.min_gap_frames,
            )
            import os
            os.remove(audio_path)
        except Exception as exc:
            logger.warning("Audio extraction or onset detection failed",
                           error=str(exc))
            # If audio fails, fall back to wrist candidates
            if wrist_candidates:
                return wrist_candidates
            return []

        if not audio_candidates and not wrist_candidates:
            return []

        fused = _fuse_audio_wrist(
            audio_candidates, wrist_candidates,
            tolerance=self.fusion_tolerance,
            min_gap_frames=self.min_gap_frames,
        )

        if fused:
            n_fusion = sum(1 for c in fused if c["source"] == "audio_visual_fusion")
            n_audio = sum(1 for c in fused if c["source"] == "audio_only")
            n_wrist = sum(1 for c in fused if c["source"] == "wrist_only")
            logger.info("Audio-visual fusion completed",
                        total=len(fused), confirmed=n_fusion,
                        audio_only=n_audio, wrist_only=n_wrist)

        return fused


class HitFrameLocalizationStage:
    name = "hit_frame_localization"
    input_keys = ["shuttle", "pose"]
    output_keys = ["hits"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shuttle_raw_df = artifacts.get_parquet("shuttle_raw")
        shuttle_clean_df = artifacts.get_parquet("shuttle")
        if shuttle_raw_df is not None and len(shuttle_raw_df) > 0:
            shuttle_df = shuttle_raw_df.copy()
            # Keep NaNs — do not ffill for hit detection (preserves gaps/reversals)
        elif shuttle_clean_df is not None and len(shuttle_clean_df) > 0:
            shuttle_df = shuttle_clean_df
            logger.warning("Hit detection falling back to cleaned shuttle; shuttle_raw missing")
        else:
            return StageResult.from_error("Shuttle tracking data required")

        pose_df = artifacts.get_parquet("pose")
        video_path: str | None = config.extra.get("video_path", "")
        audio_enabled = (
            getattr(settings, "audio_hit_enabled", True)
            and bool(video_path)
        )

        detector = GlobalHitCandidateDetector.from_settings()

        # ── Phase 0: audio-visual fusion (or wrist-only fallback) ──
        fusion_candidates: list[dict] = []
        if audio_enabled:
            audio_fuser = AudioFusionDetector.from_settings()
            fusion_candidates = audio_fuser.detect(
                video_path, pose_df,
                fps=float(config.processing_fps or 30.0),
            )
        elif getattr(settings, "wrist_hit_enabled", True) and pose_df is not None and len(pose_df) > 0:
            fps = float(config.processing_fps or getattr(settings, "fps", 30.0))
            fusion_candidates = _detect_wrist_peaks(
                pose_df, fps,
                min_speed=getattr(settings, "wrist_hit_min_speed", 0.15),
                min_interval_s=getattr(settings, "wrist_hit_min_interval_s", 0.3),
                min_kp_conf=getattr(settings, "wrist_hit_min_conf", 0.30),
                adaptive_frac=getattr(settings, "wrist_hit_adaptive_frac", 0.40),
            )

        # ── Phase 1: shuttle trajectory detector ──
        shuttle_candidates = detector.detect(shuttle_df)

        # ── Merge: fusion candidates get priority, shuttle fills gaps ──
        candidates = list(fusion_candidates)
        fusion_frames = {c["frame"] for c in fusion_candidates}
        for c in shuttle_candidates:
            if c["frame"] not in fusion_frames:
                candidates.append(c)
        candidates = non_max_suppression(candidates, detector.min_gap_frames)

        if audio_enabled:
            n_source = {"fusion": sum(1 for c in candidates if c.get("source", "").startswith("audio")),
                        "shuttle": sum(1 for c in candidates if c.get("source") is None)}
            logger.info("Hit candidates merged (audio-enabled)",
                        total=len(candidates), sources=n_source)
        elif fusion_candidates:
            # Wrist-only fallback — adjust wrist scores and log
            wrist_weight = getattr(settings, "wrist_hit_score_weight", 0.40)
            for c in candidates:
                if c.get("source") == "wrist_peak":
                    c["score"] *= wrist_weight
            logger.info("Hit candidates merged (wrist-only fallback)",
                        total=len(candidates),
                        wrist_count=len(fusion_candidates))

        # ── Phase 2: pose-based contact refinement (unchanged) ──
        refine_window = getattr(settings, "hit_refine_window", 4)
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
            candidates = non_max_suppression(candidates, detector.min_gap_frames)
            logger.info("Refined hit frames via wrist-to-shuttle proximity",
                        refined_count=refined_count, total=len(candidates),
                        window=refine_window)

        # ── Phase 3: calibration offset ──
        # Systematic correction: shuttle trajectory inflection lags true
        # contact by ~8 frames (labelled vs 99 enriched labels, median error
        # was +8).  Subtract the offset to center the distribution.
        calib_offset = getattr(settings, "hit_frame_calibration_offset", 8)
        for c in candidates:
            c["frame"] = max(0, c["frame"] - calib_offset)
            c["_calib_offset"] = calib_offset

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

        hits = []
        for c in candidates:
            hit = {"frame": c["frame"], "confidence": c["score"]}
            if "_refined_offset" in c:
                hit["_refined_offset"] = c["_refined_offset"]
            if "dominant_hand" in c:
                hit["dominant_hand"] = c["dominant_hand"]
            hits.append(hit)
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
