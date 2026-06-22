import pandas as pd
import numpy as np

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


def _infer_end_reason(stroke_type: str, confidence: float) -> str:
    """Infer rally end reason from the last shot.

    Rules:
    - High-confidence smash/drop/kill -> winner (aggressive finishing shot)
    - Net shot -> net (hitter hit the net)
    - Low-confidence clear/drive/lift -> unforced_error (weak basic shot)
    - Everything else -> forced_error (opponent won, not necessarily an error by hitter)
    """
    if stroke_type in ("smash", "drop", "kill") and confidence >= 0.5:
        return "winner"
    if stroke_type in ("net_shot",):
        return "net"
    if stroke_type in ("clear", "drive", "lift") and confidence < 0.35:
        return "unforced_error"
    return "forced_error"


def _is_rally_ending_shot(stroke_type: str, confidence: float, next_gap: int) -> bool:
    """Determine if a shot likely ended the rally.

    Uses stroke type, confidence, AND the gap to the next shot as signals.
    A shot is considered rally-ending if:
    1. It's followed by a gap > 45 frames (primary signal)
    2. It's a high-confidence winner (smash/drop/kill with conf >= 0.6) AND gap > 25 frames
    3. It's a net shot AND gap > 15 frames (net shots often end rallies quickly)

    Args:
        stroke_type: The classified stroke type
        confidence: BST model confidence for this stroke
        next_gap: Frame gap to the next shot (0 if this is the last shot)

    Returns:
        True if this shot likely ended the rally
    """
    # Large gap always indicates rally end
    if next_gap > 45:
        return True

    # High-confidence aggressive shots with moderate gap
    if stroke_type in ("smash", "drop", "kill") and confidence >= 0.6 and next_gap > 25:
        return True

    # Net shots with small gap (shuttle hit the net = point over quickly)
    if stroke_type in ("net_shot",) and next_gap > 15:
        return True

    return False


class RallySegmentationStage:
    name = "rally_segmentation"
    input_keys = ["shots"]
    output_keys = ["rallies"]

    DEFAULT_GAP_THRESHOLD = 60
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

            # Check if current shot ended the rally
            stroke_type = shots_df.iloc[i - 1].get("stroke_type", "clear")
            stroke_confidence = shots_df.iloc[i - 1].get("stroke_confidence", 0.5)
            next_gap = shots_df.iloc[i]["frame"] - shots_df.iloc[i - 1]["frame"]

            rally_ending = _is_rally_ending_shot(stroke_type, stroke_confidence, next_gap)

            # Split rally if: time gap exceeded OR shot likely ended rally
            if frame_gap > threshold or rally_ending:
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
            last_shot = r_frames.iloc[-1]
            last_pid = last_shot.get("player_id")
            stroke_type = last_shot.get("stroke_type", "clear")
            stroke_confidence = last_shot.get("stroke_confidence", 0.5)

            end_reason = _infer_end_reason(stroke_type, stroke_confidence)
            r["end_reason"] = end_reason

            if end_reason == "winner":
                r["winner_player_id"] = last_pid
            elif end_reason in ("forced_error", "unforced_error", "net"):
                r["winner_player_id"] = "player_2" if last_pid == "player_1" else "player_1"
            else:
                r["winner_player_id"] = None

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
