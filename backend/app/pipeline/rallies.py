import pandas as pd
import numpy as np

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class RallySegmentationStage:
    name = "rally_segmentation"
    input_keys = ["shots"]
    output_keys = ["rallies"]

    DEFAULT_GAP_THRESHOLD = 45
    DEFAULT_MIN_SHOTS = 3

    def run(self, artifacts: ArtifactStore, config: StageConfig,
            gap_threshold: int | None = None, min_shots: int | None = None) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"rally_count": 0})

        threshold = gap_threshold or self.DEFAULT_GAP_THRESHOLD
        min_s = min_shots or self.DEFAULT_MIN_SHOTS
        shots_df = shots_df.sort_values("frame").reset_index(drop=True)

        rallies = []
        rally_id = 0
        rally_start = shots_df.iloc[0]["frame"]
        rally_shots_idx = [0]

        for i in range(1, len(shots_df)):
            frame_gap = shots_df.iloc[i]["frame"] - shots_df.iloc[i - 1]["frame"]
            if frame_gap > threshold:
                if len(rally_shots_idx) >= min_s:
                    rally_id += 1
                    end_frame = int(shots_df.iloc[rally_shots_idx[-1]]["frame"])
                    rallies.append({
                        "rally_id": rally_id,
                        "start_frame": int(rally_start),
                        "end_frame": end_frame,
                        "shot_count": len(rally_shots_idx),
                    })
                rally_start = shots_df.iloc[i]["frame"]
                rally_shots_idx = [i]
            else:
                rally_shots_idx.append(i)

        if len(rally_shots_idx) >= min_s:
            rally_id += 1
            rallies.append({
                "rally_id": rally_id,
                "start_frame": int(rally_start),
                "end_frame": int(shots_df.iloc[rally_shots_idx[-1]]["frame"]),
                "shot_count": len(rally_shots_idx),
            })

        rally_lookup = {}
        for r in rallies:
            for _, shot in shots_df.iterrows():
                f = int(shot["frame"])
                if r["start_frame"] <= f <= r["end_frame"]:
                    rally_lookup[f] = r["rally_id"]

        shots_df["rally_id"] = shots_df["frame"].map(rally_lookup)
        artifacts.set_parquet("shots", shots_df)

        for r in rallies:
            r_frames = shots_df[shots_df["rally_id"] == r["rally_id"]].sort_values("frame")
            if len(r_frames) == 0:
                continue
            last_pid = r_frames.iloc[-1].get("player_id")
            r["winner_player_id"] = last_pid
            r["end_reason"] = "winner"
            r["serving_player_id"] = None
            fps = 30.0
            r["start_ts"] = round(r["start_frame"] / fps, 3)
            r["end_ts"] = round(r["end_frame"] / fps, 3)
            r["match_id"] = None

        rallies_df = pd.DataFrame(rallies) if rallies else pd.DataFrame(
            columns=["rally_id", "start_frame", "end_frame", "shot_count"]
        )
        artifacts.set_parquet("rallies", rallies_df)

        return StageResult.success(
            artifacts={"rallies": artifacts.path("rallies")},
            metadata={"rally_count": len(rallies)}
        )
