import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings


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


class HitFrameLocalizationStage:
    name = "hit_frame_localization"
    input_keys = ["shuttle"]
    output_keys = ["hits"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shuttle_df = artifacts.get_parquet("shuttle")
        if shuttle_df is None or len(shuttle_df) == 0:
            # Fallback to shuttle_raw if cleaned shuttle unavailable
            shuttle_df = artifacts.get_parquet("shuttle_raw")
            if shuttle_df is None or len(shuttle_df) == 0:
                return StageResult.from_error("Shuttle tracking data required")
            shuttle_df["x"] = shuttle_df["x"].ffill().bfill()
            shuttle_df["y"] = shuttle_df["y"].ffill().bfill()

        detector = GlobalHitCandidateDetector.from_settings()
        candidates = detector.detect(shuttle_df)

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

        hits = [{"frame": c["frame"], "confidence": c["score"]} for c in candidates]
        hits_data = pd.DataFrame(hits)
        artifacts.set_parquet("hits", hits_data)

        logger.info(f"Localized {len(hits)} hit frames via shuttle-centric detector")

        return StageResult.success(
            artifacts={"hits": artifacts.path("hits")},
            metadata={"hit_count": len(hits), "frames_analyzed": len(shuttle_df)}
        )
