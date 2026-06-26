from pathlib import Path

import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger
from app.pipeline.shared.shuttle_utils import clean_trajectory
from app.config.settings import settings


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
    output_keys = ["shuttle"]

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

        artifacts.set_parquet("shuttle", df)

        avg_conf = df["confidence"].mean()
        logger.info(f"Stored {len(df)} shuttle tracking rows (avg_conf={avg_conf:.2f})")

        return StageResult.success(
            artifacts={"shuttle": artifacts.path("shuttle")},
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
