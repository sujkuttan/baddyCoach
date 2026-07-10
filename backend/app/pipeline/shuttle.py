from pathlib import Path

import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.pipeline.shared.shuttle_utils import clean_trajectory, ShuttleSmoother
from app.config.settings import settings


def _add_court_space_columns(df: pd.DataFrame, H: np.ndarray, fps: float) -> pd.DataFrame:
    """Add court-space coordinates, speed, and direction to shuttle dataframe.

    Computes x_court, y_court (metres via homography), speed_court (m/s),
    and direction_x, direction_y (normalised) for each valid detection. Points
    beyond the configured court margin or with an impossible consecutive speed
    are marked ``court_rejected`` and retain their raw pixel coordinates.
    """
    from app.pipeline.shared.court import image_to_court, COURT_LENGTH, COURT_WIDTH

    court_xs = np.full(len(df), np.nan, dtype=np.float64)
    court_ys = np.full(len(df), np.nan, dtype=np.float64)
    speeds = np.full(len(df), np.nan, dtype=np.float64)
    dir_xs = np.full(len(df), np.nan, dtype=np.float64)
    dir_ys = np.full(len(df), np.nan, dtype=np.float64)
    court_rejected = np.zeros(len(df), dtype=bool)

    prev_cx, prev_cy = None, None
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = row.get("x"), row.get("y")
        if pd.notna(x) and pd.notna(y):
            cx, cy = image_to_court(H, (float(x), float(y)))
            out_of_bounds = (
                not np.isfinite(cx)
                or not np.isfinite(cy)
                or cx < -settings.shuttle_oob_margin_meters
                or cx > COURT_LENGTH + settings.shuttle_oob_margin_meters
                or cy < -settings.shuttle_oob_margin_meters
                or cy > COURT_WIDTH + settings.shuttle_oob_margin_meters
            )
            if out_of_bounds:
                court_rejected[i] = True
                prev_cx, prev_cy = None, None
                continue

            speed = np.nan
            court_xs[i] = cx
            court_ys[i] = cy
            if prev_cx is not None:
                dx = cx - prev_cx
                dy = cy - prev_cy
                speed = np.sqrt(dx * dx + dy * dy) * fps
                if speed > settings.shuttle_max_speed_mps:
                    court_rejected[i] = True
                    court_xs[i] = np.nan
                    court_ys[i] = np.nan
                    prev_cx, prev_cy = None, None
                    continue
                speeds[i] = speed
                norm = np.sqrt(dx * dx + dy * dy) + 1e-8
                dir_xs[i] = dx / norm
                dir_ys[i] = dy / norm
            prev_cx, prev_cy = cx, cy
        else:
            prev_cx, prev_cy = None, None

    df = df.copy()
    df["x_court"] = court_xs
    df["y_court"] = court_ys
    df["speed_court"] = speeds
    df["direction_x"] = dir_xs
    df["direction_y"] = dir_ys
    df["court_rejected"] = court_rejected
    return df


def _count_big_jumps(x, y, threshold: float) -> int:
    """Count frame-to-frame displacements exceeding threshold."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return int((d > threshold).sum())


class ShuttleTrackingStage:
    name = "shuttle_tracking"
    input_keys = []
    output_keys = ["shuttle", "shuttle_raw"]

    def run(
        self,
        artifacts: ArtifactStore,
        config: StageConfig,
        frames: list[np.ndarray] | None = None,
        shuttle_data: list[dict] | None = None
    ) -> StageResult:
        """Run shuttle tracking.

        If frames provided, runs TrackNetV3 inference.
        If shuttle_data provided, uses pre-computed data.
        """
        if shuttle_data:
            return self._store_data(artifacts, shuttle_data)

        if frames:
            shuttle_data = self._run_tracknet(frames)
            if shuttle_data is None:
                return StageResult.from_error("TrackNet model not available, cannot perform shuttle tracking")
            self._store_resolution(artifacts, frames)
            return self._store_data(artifacts, shuttle_data)

        return StageResult.from_error("No frames or shuttle data provided")

    def _run_tracknet(self, frames: list[np.ndarray]) -> list[dict]:
        """Run TrackNetV3 on video frames with InpaintNet trajectory rectification.

        Model loading stays local to this stage (not via shared.models.setup_models)
        to keep the colab pipeline's self-contained model loading approach intact.
        """
        from app.pipeline.shared.models import get_tracknet

        model = get_tracknet()
        if model is None:
            logger.error("TrackNet model not available")
            return None

        original_size = (frames[0].shape[1], frames[0].shape[0]) if frames else (settings.default_frame_width, settings.default_frame_height)
        predictions = model.predict_batch(frames, original_size=original_size)

        shuttle_data = []
        for i, pred in enumerate(predictions):
            shuttle_data.append({
                "frame": i,
                "x": pred["x"],
                "y": pred["y"],
                "confidence": pred["confidence"],
                "was_repaired": pred.get("was_repaired", False),
            })

        return shuttle_data

    def _store_data(self, artifacts: ArtifactStore, shuttle_data: list[dict]) -> StageResult:
        """Store shuttle tracking data in artifacts."""
        df = pd.DataFrame(shuttle_data)
        required_cols = {"frame", "x", "y", "confidence"}
        if not required_cols.issubset(df.columns):
            return StageResult.from_error(f"Shuttle data must contain columns: {required_cols}")

        if settings.shuttle_clean_enabled:
            n_before = len(df)
            jumps_before = _count_big_jumps(df["x"].values, df["y"].values, settings.shuttle_max_jump_px)
            low_conf_before = int((df["confidence"] < settings.shuttle_clean_min_conf).sum())

            # Store raw (conf-gate only, no spike/interp/smooth) for hit detection
            df_raw = df.copy()
            raw_conf = df_raw["confidence"].values.astype(np.float64) < settings.shuttle_clean_min_conf
            df_raw.loc[raw_conf, "x"] = np.nan
            df_raw.loc[raw_conf, "y"] = np.nan
            artifacts.set_parquet("shuttle_raw", df_raw)

            df_orig = df.copy()
            df = clean_trajectory(df, settings)
            n_interp = int(df["was_interpolated"].sum())
            n_spike = n_interp - int(
                ((df_orig["confidence"] < settings.shuttle_clean_min_conf).values
                 & df["was_interpolated"].values).sum()
            )
            logger.info(
                f"Shuttle cleaned: {low_conf_before}/{n_before} low-conf "
                f"({100 * low_conf_before / max(n_before, 1):.1f}%), "
                f"{max(0, n_spike)} spikes removed, {n_interp} gaps filled, "
                f"{jumps_before} >{settings.shuttle_max_jump_px:.0f}px jumps → ~0"
            )

        if not settings.shuttle_clean_enabled:
            artifacts.set_parquet("shuttle_raw", df.copy())

        # Enrich cleaned shuttle with court-space coordinates when homography is available
        court = artifacts.get("court")
        metadata = artifacts.get("video_metadata") or {}
        fps = metadata.get("fps", settings.fps)
        if court and court.get("valid", False) and court.get("homography") is not None:
            H = np.array(court["homography"])
            df = _add_court_space_columns(df, H, float(fps))
            logger.info("Added court-space columns to shuttle data")

        # Compute derived kinematics (velocity, acceleration, curvature)
        smoother = ShuttleSmoother(settings)
        df["velocity"] = smoother.compute_velocity(df)
        df["acceleration"] = smoother.compute_acceleration(df)
        df["curvature"] = smoother.compute_curvature(df)

        artifacts.set_parquet("shuttle", df)

        avg_conf = df["confidence"].mean()
        logger.info(f"Stored {len(df)} shuttle tracking rows (avg_conf={avg_conf:.2f})")

        return StageResult.success(
            artifacts={"shuttle": artifacts.path("shuttle"), "shuttle_raw": artifacts.path("shuttle_raw")},
            metadata={
                "total_frames": len(df),
                "avg_confidence": float(avg_conf),
                "frames_with_shuttle": int((df["confidence"] > 0.5).sum()),
            },
        )

    @staticmethod
    def _store_resolution(artifacts: ArtifactStore, frames: list[np.ndarray]) -> None:
        """Store video resolution from frame shapes."""
        if frames:
            artifacts.set("video_resolution", {
                "width": int(frames[0].shape[1]),
                "height": int(frames[0].shape[0]),
            })
