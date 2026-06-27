import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings


class HitFrameLocalizationStage:
    name = "hit_frame_localization"
    input_keys = ["shuttle_raw", "pose"]
    output_keys = ["hits"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shuttle_df = artifacts.get_parquet("shuttle_raw")
        if shuttle_df is None or len(shuttle_df) == 0:
            return StageResult.from_error("Shuttle tracking data required")

        # Fill NaN in x,y from trajectory cleaning: carry-forward fills gaps
        # with zero-motion (dx=0, dy=0), avoiding NaN propagation in np.diff
        # while producing no fake teleports or false peaks.
        shuttle_df["x"] = shuttle_df["x"].ffill().bfill()
        shuttle_df["y"] = shuttle_df["y"].ffill().bfill()

        pose_df = artifacts.get_parquet("pose")

        reversal_score = self._compute_reversal(shuttle_df)
        trajectory_score = self._compute_trajectory_change(shuttle_df)
        speed_score = self._compute_speed_peaks(shuttle_df)
        swing_score = self._compute_swing_acceleration(pose_df, n_frames=len(shuttle_df)) if pose_df is not None else np.zeros(len(shuttle_df))
        proximity_score = self._compute_proximity(shuttle_df, pose_df) if pose_df is not None else np.zeros(len(shuttle_df))

        # Proximity gate: zero out all other signals where shuttle is far from players
        gate = (proximity_score >= settings.hit_proximity_gate).astype(np.float64)

        combined = (
            gate * (
                settings.hit_reversal_weight * reversal_score +
                settings.hit_trajectory_weight * trajectory_score +
                settings.hit_speed_weight * speed_score +
                settings.hit_swing_weight * swing_score
            ) +
            settings.hit_proximity_weight * proximity_score
        )

        peaks, _ = find_peaks(
            combined,
            height=settings.hit_confidence_threshold,
            distance=3,
        )
        hit_frames = peaks

        hits = []
        for idx in hit_frames:
            frame = int(shuttle_df.iloc[idx]["frame"])
            hits.append({
                "frame": frame,
                "confidence": float(combined[idx]),
            })

        # Write debug hit scores if debug_level >= 2
        if config.debug_level >= 2:
            debug_hit_df = pd.DataFrame({
                "frame": shuttle_df["frame"].values,
                "reversal_raw": reversal_score,
                "trajectory_raw": trajectory_score,
                "speed_raw": speed_score,
                "swing_raw": swing_score,
                "proximity_raw": proximity_score,
                "combined": combined,
                "is_peak": [False] * len(shuttle_df),
            })
            debug_hit_df.loc[peaks, "is_peak"] = True
            artifacts.set_parquet("debug_hit_scores", debug_hit_df)

        if len(hits) > 1:
            fps = float(config.processing_fps or settings.fps)
            min_gap = max(3, int(fps * settings.hit_dedup_gap_seconds))
            hits = sorted(hits, key=lambda h: h["frame"])
            deduped = [hits[0]]
            for h in hits[1:]:
                gap = h["frame"] - deduped[-1]["frame"]
                if gap >= min_gap:
                    deduped.append(h)
                elif h["confidence"] > deduped[-1]["confidence"]:
                    deduped[-1] = h
            hits = deduped

        hits_data = pd.DataFrame(hits)
        artifacts.set_parquet("hits", hits_data)

        logger.info(f"Localized {len(hits)} hit frames from {len(shuttle_df)} shuttle samples")

        return StageResult.success(
            artifacts={"hits": artifacts.path("hits")},
            metadata={"hits": hits, "hit_count": len(hits), "frames_analyzed": len(shuttle_df)}
        )

    def _compute_reversal(self, shuttle_df: pd.DataFrame) -> np.ndarray:
        """Shuttle vertical-direction-reversal signal.

        A hit reverses the shuttle's vertical trajectory (up→down or down→up).
        This is the strongest, most camera-robust physical cue — it works with
        pixel coordinates regardless of court detection quality.
        """
        y = shuttle_df["y"].values
        dy = np.diff(y, prepend=y[0])

        # Sign of vertical velocity: +1 = moving down, -1 = moving up, 0 = stationary
        sign = np.sign(dy)
        sign[(sign > -0.01) & (sign < 0.01)] = 0

        # Reversal at frame t when sign flips AND both sides have non-trivial motion
        reversal = np.zeros(len(y), dtype=np.float64)
        rev_mask = (sign[:-1] * sign[1:] < 0) & (np.abs(dy[:-1]) > 1.0) & (np.abs(dy[1:]) > 1.0)
        reversal_indices = np.where(rev_mask)[0] + 1  # t+1 is the reversal frame
        reversal[reversal_indices] = 1.0

        # Spread energy across a narrow window around each reversal
        for ri in reversal_indices:
            start = max(0, ri - 2)
            end = min(len(y), ri + 3)
            # Triangular window: peak at reversal, taper off
            for ti in range(start, end):
                dist = abs(ti - ri)
                reversal[ti] = max(reversal[ti], 1.0 - dist * 0.3)

        return reversal

    def _compute_trajectory_change(self, shuttle_df: pd.DataFrame) -> np.ndarray:
        """Direction-change signal from shuttle trajectory angle."""
        x = shuttle_df["x"].values
        y = shuttle_df["y"].values
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        angle = np.arctan2(dy, dx)
        angle_diff = np.abs(np.diff(angle, prepend=angle[0]))
        score = angle_diff / (np.pi + 1e-6)
        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score

    def _compute_speed_peaks(self, shuttle_df: pd.DataFrame) -> np.ndarray:
        """Speed-peak signal from shuttle velocity."""
        x = shuttle_df["x"].values
        y = shuttle_df["y"].values
        speed = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
        peaks, _ = find_peaks(speed, distance=3)
        score = np.zeros(len(speed))
        score[peaks] = speed[peaks]
        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score

    def _compute_swing_acceleration(self, pose_df: pd.DataFrame, n_frames: int = 0) -> np.ndarray:
        """Wrist acceleration signal.

        Raw wrist velocity is noisy; acceleration (change in velocity) is a
        sharper indicator of impact — the racket suddenly decelerates upon
        contacting the shuttle.
        """
        if n_frames == 0:
            n_frames = pose_df["frame"].max() + 1
        score = np.zeros(n_frames)

        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            if len(player_poses) < 4:
                continue

            frames = player_poses["frame"].values
            wrist_positions = []
            for _, row in player_poses.iterrows():
                kps = np.array(row['keypoints'].tolist())
                if kps.ndim == 1:
                    kps = np.array(kps.tolist())
                if kps.shape == (17, 3):
                    wrist = (kps[9][:2] + kps[10][:2]) / 2
                    wrist_positions.append(wrist)
                else:
                    wrist_positions.append(np.array([np.nan, np.nan]))
            wrist_positions = np.array(wrist_positions)

            # Velocity: frame-to-frame displacement
            vel = np.zeros_like(wrist_positions)
            vel[1:] = wrist_positions[1:] - wrist_positions[:-1]
            vel[0] = vel[1]

            # Acceleration: change in velocity (sharp jerk at impact)
            acc = np.zeros(len(vel))
            acc[2:] = np.sqrt(np.sum((vel[2:] - vel[1:-1])**2, axis=1))
            acc = np.nan_to_num(acc, 0)

            for ti in range(min(len(frames), n_frames)):
                score[int(frames[ti])] = max(score[int(frames[ti])], acc[ti])

        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score

    def _compute_proximity(self, shuttle_df: pd.DataFrame, pose_df: pd.DataFrame) -> np.ndarray:
        """Wrist-to-shuttle proximity. Used as a gate, not an additive floor."""
        score = np.zeros(len(shuttle_df))
        shuttle_positions = shuttle_df[["x", "y"]].values
        shuttle_frames = shuttle_df["frame"].values

        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id]
            for _, row in player_poses.iterrows():
                frame_idx = int(row["frame"])
                if frame_idx >= len(score):
                    continue
                kps = np.array(row["keypoints"].tolist())
                if kps.ndim == 1:
                    kps = np.array(kps.tolist())
                if kps.shape == (17, 3):
                    wrist = (kps[9][:2] + kps[10][:2]) / 2
                    shuttle_pos_idx = min(np.searchsorted(shuttle_frames, frame_idx), len(shuttle_positions) - 1)
                    shuttle_pos = shuttle_positions[shuttle_pos_idx]
                    dist = np.sqrt(np.sum((wrist - shuttle_pos)**2))
                    score[frame_idx] = max(score[frame_idx], 1.0 / (1.0 + dist / 100.0))

        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score
