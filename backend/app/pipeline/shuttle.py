import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.logging import logger


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
            self._store_resolution(artifacts, frames)
            return self._store_data(artifacts, shuttle_data)

        return StageResult.from_error("No frames or shuttle data provided")

    def _run_tracknet(self, frames: list[np.ndarray]) -> list[dict]:
        """Run TrackNetV3 on video frames.

        Model loading stays local to this stage (not via shared.models.setup_models)
        to keep the colab pipeline's self-contained model loading approach intact.
        """
        from app.models.tracknet import TrackNetV3
        from app.config.settings import settings

        model_path = str(settings.tracknet_model_path)
        device = settings.device

        model = TrackNetV3(model_path, device=device)

        original_size = (frames[0].shape[1], frames[0].shape[0]) if frames else (1280, 720)
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
