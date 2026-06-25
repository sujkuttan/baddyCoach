import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.config.settings import settings


class HitFrameLocalizationStage:
    name = "hit_frame_localization"
    input_keys = ["shuttle", "pose"]
    output_keys = ["hits"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shuttle_df = artifacts.get_parquet("shuttle")
        if shuttle_df is None or len(shuttle_df) == 0:
            return StageResult.from_error("Shuttle tracking data required")

        pose_df = artifacts.get_parquet("pose")

        trajectory_score = self._compute_trajectory_change(shuttle_df)
        speed_score = self._compute_speed_peaks(shuttle_df)
        proximity_score = self._compute_proximity(shuttle_df, pose_df) if pose_df is not None else np.zeros(len(shuttle_df))
        swing_score = self._compute_swing_peaks(pose_df, n_frames=len(shuttle_df)) if pose_df is not None else np.zeros(len(shuttle_df))

        combined = (
            settings.hit_trajectory_weight * trajectory_score +
            settings.hit_speed_weight * speed_score +
            settings.hit_proximity_weight * proximity_score +
            settings.hit_swing_weight * swing_score
        )

        peaks, _ = find_peaks(
            combined,
            height=settings.hit_confidence_threshold,
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
                "trajectory_raw": trajectory_score,
                "speed_raw": speed_score,
                "proximity_raw": proximity_score,
                "swing_raw": swing_score,
                "combined": combined,
                "is_peak": False,
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

    def _compute_trajectory_change(self, shuttle_df: pd.DataFrame) -> np.ndarray:
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
        x = shuttle_df["x"].values
        y = shuttle_df["y"].values
        speed = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
        peaks, _ = find_peaks(speed, distance=3)
        score = np.zeros(len(speed))
        score[peaks] = speed[peaks]
        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score

    def _compute_proximity(self, shuttle_df: pd.DataFrame, pose_df: pd.DataFrame) -> np.ndarray:
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
                    shuttle_pos_idx = np.searchsorted(shuttle_frames, frame_idx)
                    shuttle_pos_idx = min(shuttle_pos_idx, len(shuttle_positions) - 1)
                    shuttle_pos = shuttle_positions[shuttle_pos_idx]
                    dist = np.sqrt(np.sum((wrist - shuttle_pos)**2))
                    score[frame_idx] = max(score[frame_idx], 1.0 / (1.0 + dist / 100.0))

        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score

    def _compute_swing_peaks(self, pose_df: pd.DataFrame, n_frames: int = 0) -> np.ndarray:
        if n_frames == 0:
            n_frames = pose_df["frame"].max() + 1
        score = np.zeros(n_frames)

        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            if len(player_poses) < 3:
                continue
            prev_kps = None
            for _, row in player_poses.iterrows():
                kps = np.array(row['keypoints'].tolist())
                if kps.ndim == 1:
                    kps = np.array(kps.tolist())
                if prev_kps is not None and kps.shape == (17, 3) and prev_kps.shape == (17, 3):
                    wrist = (kps[9][:2] + kps[10][:2]) / 2
                    prev_wrist = (prev_kps[9][:2] + prev_kps[10][:2]) / 2
                    arm_velocity = np.sqrt(np.sum((wrist - prev_wrist)**2))
                    if int(row["frame"]) < n_frames:
                        score[int(row["frame"])] = arm_velocity
                prev_kps = kps

        m = np.percentile(score, 95)
        return score / (m + 1e-6) if m > 0 else score
