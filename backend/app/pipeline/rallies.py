import pandas as pd
import numpy as np

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.utils import (
    _infer_end_reason, _is_rally_ending_shot,
    _find_dead_shuttle_window, _winner_from_shuttle_landing,
)
from app.pipeline.shared.logging import logger
from app.config.settings import settings


def finalize_rally_outcomes(rallies_df, shots_df, shuttle_raw=None,
                            court=None, players=None, fps=30.0):
    """Set serving_player_id and winner_player_id on rallies after attribution.

    Must be called after PlayerAttributionStage has set player_id on every
    shot.  This is the single point of truth for rally outcomes — all other
    computation sites (RallySegmentationStage, attribution.py stale block)
    have been removed.

    serve: first in-rally shot's player_id.
    winner: shuttle-landing → opponent, falls back to end_reason mapping.
    degeneracy guard: if every rally has the same winner, fall back to
    stroke-inference-based winner.
    """
    for r_idx in rallies_df.index:
        rally = rallies_df.loc[r_idx]
        r_frames = shots_df[shots_df["rally_id"] == rally["rally_id"]].sort_values("frame")
        if len(r_frames) == 0:
            continue

        # Serve: first shot's attributed player_id
        first_pid = r_frames.iloc[0].get("player_id")
        rallies_df.at[r_idx, "serving_player_id"] = first_pid

        # end_reason from last shot's stroke (set provisionally by
        # RallySegmentationStage; recompute here with final player_id)
        last_shot = r_frames.iloc[-1]
        stroke_type = last_shot.get("stroke_type", "clear")
        stroke_confidence = last_shot.get("stroke_confidence", 0.5)
        last_pid = last_shot.get("player_id")
        rallies_df.at[r_idx, "end_reason"] = _infer_end_reason(stroke_type, stroke_confidence)

        # Winner: landing-based, falls back to end_reason mapping
        winner = _compute_rally_winner_after_attribution(
            rally, shots_df, shuttle_raw, court, players,
            end_reason=rallies_df.at[r_idx, "end_reason"], last_pid=last_pid,
        )
        rallies_df.at[r_idx, "winner_player_id"] = winner

        # Timestamps
        rallies_df.at[r_idx, "start_ts"] = round(rally["start_frame"] / fps, 3)
        rallies_df.at[r_idx, "end_ts"] = round(rally["end_frame"] / fps, 3)
        rallies_df.at[r_idx, "match_id"] = None

    # Degeneracy guard: if every rally resolves to the same player,
    # fall back to stroke-inference-based winner
    if settings.rally_winner_degenerate_warn and len(rallies_df) >= 3:
        resolved = rallies_df[rallies_df["winner_player_id"].notna()]
        if len(resolved) == len(rallies_df) and resolved["winner_player_id"].nunique() == 1:
            unique_winner = resolved.iloc[0]["winner_player_id"]
            logger.warning(
                "All %d rallies resolve to %s — degenerate winner. "
                "Falling back to stroke inference.",
                len(rallies_df), unique_winner,
            )
            for r_idx in rallies_df.index:
                r_frames = shots_df[shots_df["rally_id"] == rallies_df.at[r_idx, "rally_id"]].sort_values("frame")
                if len(r_frames) == 0:
                    continue
                r_pid = r_frames.iloc[-1].get("player_id")
                r_end = rallies_df.at[r_idx, "end_reason"]
                if r_pid is None:
                    continue
                if r_end == "winner":
                    rallies_df.at[r_idx, "winner_player_id"] = r_pid
                elif r_end in ("forced_error", "unforced_error", "net"):
                    rallies_df.at[r_idx, "winner_player_id"] = (
                        "player_2" if r_pid == "player_1" else "player_1"
                    )

    return rallies_df


def _compute_rally_winner_after_attribution(
    rally_df, shots_df, shuttle_raw=None, court=None, players=None,
    end_reason=None, last_pid=None,
):
    """Compute rally winner after player attribution is complete.

    Primary method: shuttle landing position from shuttle_raw
    (high-confidence, non-interpolated) -> which side the shuttle died on
    -> opponent wins.

    Fallback: map end_reason -> winner deterministically:
      - "winner" -> last hitter wins
      - "forced_error" / "unforced_error" / "net" -> opponent of last hitter wins

    Returns None only when last_pid itself is missing.
    """
    # Primary: winner from shuttle landing using raw (non-interpolated) data
    if shuttle_raw is not None:
        winner = _winner_from_shuttle_landing(
            shuttle_raw, rally_df["start_frame"], rally_df["end_frame"], court, players,
        )
        if winner is not None:
            return winner

    # Fallback: map end_reason -> winner deterministically
    if last_pid is None or end_reason is None:
        return None

    if end_reason == "winner":
        return last_pid
    elif end_reason in ("forced_error", "unforced_error", "net"):
        return "player_2" if last_pid == "player_1" else "player_1"
    return None


class RallySegmentationStage:
    name = "rally_segmentation"
    input_keys = ["shots"]
    output_keys = ["rallies"]

    def run(self, artifacts: ArtifactStore, config: StageConfig,
            gap_threshold: int | None = None, min_shots: int | None = None) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"rally_count": 0})

        # Cleaned shuttle for dead-window detection during segmentation;
        # shuttle_raw (non-interpolated) for landing-point detection.
        shuttle_df = artifacts.get_parquet("shuttle")
        shuttle_raw = artifacts.get_parquet("shuttle_raw")
        if shuttle_raw is None:
            shuttle_raw = shuttle_df
        court = artifacts.get("court")
        players_data = artifacts.get("players")

        threshold = gap_threshold or settings.rally_gap_threshold
        min_s = min_shots or settings.rally_min_shots
        shots_df = shots_df.sort_values("frame").reset_index(drop=True)

        rallies = []
        rally_id = 0
        rally_start = shots_df.iloc[0]["frame"]
        rally_shots_idx = [0]

        for i in range(1, len(shots_df)):
            frame_gap = shots_df.iloc[i]["frame"] - shots_df.iloc[i - 1]["frame"]

            stroke_type = shots_df.iloc[i - 1].get("stroke_type", "clear")
            stroke_confidence = shots_df.iloc[i - 1].get("stroke_confidence", 0.5)
            next_gap = shots_df.iloc[i]["frame"] - shots_df.iloc[i - 1]["frame"]

            rally_ending = _is_rally_ending_shot(stroke_type, stroke_confidence, next_gap)

            # Dead-shuttle check: scan shuttle track between consecutive shots
            dead_shuttle = _find_dead_shuttle_window(
                shuttle_df,
                int(shots_df.iloc[i - 1]["frame"]),
                int(shots_df.iloc[i]["frame"]),
            )

            if frame_gap > threshold or dead_shuttle or rally_ending:
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

        fps = float(config.processing_fps or 30.0)
        for r in rallies:
            r_frames = shots_df[shots_df["rally_id"] == r["rally_id"]].sort_values("frame")
            if len(r_frames) == 0:
                continue

            # Provisional end_reason (finalized by finalize_rally_outcomes
            # after attribution sets player_id). Serve/winner/degeneracy
            # are not computed here — they need attributed player_ids.
            last_shot = r_frames.iloc[-1]
            stroke_type = last_shot.get("stroke_type", "clear")
            stroke_confidence = last_shot.get("stroke_confidence", 0.5)
            r["end_reason"] = _infer_end_reason(stroke_type, stroke_confidence)

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
