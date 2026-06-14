import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class ShuttleTrackingStage:
    name = "shuttle_tracking"
    input_keys = []
    output_keys = ["shuttle"]

    def run(self, artifacts: ArtifactStore, config: StageConfig, shuttle_data: list[dict] | None = None) -> StageResult:
        if not shuttle_data:
            return StageResult.from_error("No shuttle tracking data provided")

        df = pd.DataFrame(shuttle_data)
        required_cols = {"frame", "x", "y", "confidence"}
        if not required_cols.issubset(df.columns):
            return StageResult.from_error(f"Shuttle data must contain columns: {required_cols}")

        artifacts.set_parquet("shuttle", df)

        avg_conf = df["confidence"].mean()
        return StageResult.success(
            artifacts={"shuttle": artifacts.path("shuttle")},
            metadata={
                "total_frames": len(df),
                "avg_confidence": float(avg_conf),
                "frames_with_shuttle": int((df["confidence"] > 0.5).sum()),
            }
        )
