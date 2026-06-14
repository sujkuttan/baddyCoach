import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class RallySegmentationStage:
    name = "rally_segmentation"
    input_keys = ["shots"]
    output_keys = ["rallies"]

    DEFAULT_GAP_THRESHOLD = 30  # frames between rallies

    def run(self, artifacts: ArtifactStore, config: StageConfig, gap_threshold: int | None = None) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"rally_count": 0})

        threshold = gap_threshold or self.DEFAULT_GAP_THRESHOLD
        shots_df = shots_df.sort_values("frame").reset_index(drop=True)

        rallies = []
        rally_id = 1
        rally_start = shots_df.iloc[0]["frame"]
        rally_shots = [0]

        for i in range(1, len(shots_df)):
            frame_gap = shots_df.iloc[i]["frame"] - shots_df.iloc[i - 1]["frame"]
            if frame_gap > threshold:
                rallies.append({
                    "rally_id": rally_id,
                    "start_frame": int(rally_start),
                    "end_frame": int(shots_df.iloc[i - 1]["frame"]),
                    "shot_count": len(rally_shots),
                })
                rally_id += 1
                rally_start = shots_df.iloc[i]["frame"]
                rally_shots = [i]
            else:
                rally_shots.append(i)

        rallies.append({
            "rally_id": rally_id,
            "start_frame": int(rally_start),
            "end_frame": int(shots_df.iloc[-1]["frame"]),
            "shot_count": len(rally_shots),
        })

        rallies_df = pd.DataFrame(rallies)
        artifacts.set_parquet("rallies", rallies_df)

        return StageResult.success(
            artifacts={"rallies": artifacts.path("rallies")},
            metadata={"rally_count": len(rallies)}
        )
