"""ShotContextStage — enriched rally graph.

Runs after rally_segmentation and player_attribution. Produces
``shot_events.parquet`` with one enriched row per shot, adding
zone, pressure context, outcome labels, and adjacency metadata.

This is the single join table that Items 1 (patterns) and 4 (drills) consume.
"""

import numpy as np
import pandas as pd

from app.config.settings import settings
from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import COURT_LENGTH, COURT_WIDTH
from app.pipeline.shared.logging import logger

ZONE_NAMES = [
    "front_left", "front_center", "front_right",
    "mid_left", "mid_center", "mid_right",
    "rear_left", "rear_center", "rear_right",
]


def _get_zone_from_court(court_x: float, court_y: float,
                         court_length: float, court_width: float) -> str:
    """Map court-space coordinates to a zone name (3×3 grid)."""
    row = min(int(court_x / court_length * 3), 2)
    col = min(int(court_y / court_width * 3), 2)
    return ZONE_NAMES[row * 3 + col]


def _compute_reaction_time(shot: pd.Series,
                           prev_shot: pd.Series | None,
                           fps: float = 30.0) -> float:
    """Time since the previous shot by either player."""
    if prev_shot is None:
        return 0.0
    return abs(float(shot.get("frame", 0)) - float(prev_shot.get("frame", 0))) / fps


def _compute_displacement(shot: pd.Series,
                          prev_same_player: pd.Series | None) -> float:
    """Court distance from this player's previous shot position."""
    if prev_same_player is None:
        return 0.0
    cx_s = shot.get("court_x")
    cy_s = shot.get("court_y")
    cx_p = prev_same_player.get("court_x")
    cy_p = prev_same_player.get("court_y")
    if pd.isna(cx_s) or pd.isna(cy_s) or pd.isna(cx_p) or pd.isna(cy_p):
        return 0.0
    return float(np.sqrt((float(cx_s) - float(cx_p)) ** 2 +
                         (float(cy_s) - float(cy_p)) ** 2))


def _compute_led_to_loss_within_k(shot_idx: int, rally_shots: list,
                                   winner_id: str | None, k: int = 2) -> bool:
    """Whether the rally ended in a loss for this player within k shots."""
    if winner_id is None:
        return False
    for i in range(shot_idx + 1, min(shot_idx + 1 + k, len(rally_shots))):
        if rally_shots[i].get("player_id") != winner_id:
            return False
    return winner_id != rally_shots[shot_idx].get("player_id")


class ShotContextStage:
    """Enrich shots with rally context, zone, pressure, and outcome labels.

    Produces ``shot_events.parquet`` with all original shot columns plus:
    zone, shot_index_in_rally, is_last_in_rally, prev_stroke_type,
    prev_player_id, next_stroke_type, next_player_id, prev_gap_s,
    next_gap_s, reaction_time_s, displacement_m, under_pressure,
    shot_outcome, won_point, lost_point, led_to_loss_within_k.
    """

    name = "shot_context"
    input_keys: list[str] = []
    output_keys = ["shot_events"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        rallies_df = artifacts.get_parquet("rallies")

        if shots_df is None or len(shots_df) == 0:
            return StageResult.from_error("Shots data required")
        if rallies_df is None or len(rallies_df) == 0:
            return StageResult.from_error("Rallies data required")

        court = artifacts.get("court") or {}
        court_length = court.get("court_length", COURT_LENGTH)
        court_width = court.get("court_width", COURT_WIDTH)

        fps = float(config.processing_fps or settings.fps)
        lookahead_k = settings.pattern_lookahead_k
        pressure_time = settings.pressure_time_s
        pressure_dist = settings.pressure_dist_m

        # Build rally lookup
        rally_map = {}
        for _, row in rallies_df.iterrows():
            rally_map[int(row["rally_id"])] = row

        rows = []
        shots_sorted = shots_df.sort_values(["rally_id", "frame"]).to_dict("records")

        # Group shots by rally
        from collections import defaultdict
        rally_groups = defaultdict(list)
        for s in shots_sorted:
            rid = s.get("rally_id")
            if rid is not None:
                rally_groups[int(rid)].append(s)

        for rid, rally_shots in rally_groups.items():
            rally = rally_map.get(rid)
            if rally is None:
                continue

            winner_id = rally.get("winner_player_id")
            n = len(rally_shots)

            # Track last shot per player for displacement calculation
            last_per_player: dict = {}

            for i, shot in enumerate(rally_shots):
                pid = shot.get("player_id", "unknown")
                frame = int(shot.get("frame", 0))
                cx = shot.get("court_x")
                cy = shot.get("court_y")

                # ── Zone ─────────────────────────────────────────
                zone = "unknown"
                if cx is not None and cy is not None and not (pd.isna(cx) or pd.isna(cy)):
                    zone = _get_zone_from_court(
                        float(cx), float(cy), court_length, court_width,
                    )

                # ── Adjacency ──────────────────────────────────────
                prev_shot = rally_shots[i - 1] if i > 0 else None
                next_shot = rally_shots[i + 1] if i < n - 1 else None

                prev_stroke = prev_shot.get("stroke_type") if prev_shot else None
                prev_player = prev_shot.get("player_id") if prev_shot else None
                next_stroke = next_shot.get("stroke_type") if next_shot else None
                next_player = next_shot.get("player_id") if next_shot else None

                prev_gap = (float(frame) - float(prev_shot["frame"])) / fps if prev_shot else 0.0
                next_gap = (float(next_shot["frame"]) - float(frame)) / fps if next_shot else 0.0

                # ── Pressure proxies ───────────────────────────────
                reaction_time = prev_gap if prev_player != pid else 0.0
                displacement = _compute_displacement(shot, last_per_player.get(pid))
                under_pressure = reaction_time < pressure_time or displacement > pressure_dist

                # ── Outcome labels ────────────────────────────────
                is_last = i == n - 1
                if is_last and winner_id is not None:
                    end_reason = rally.get("end_reason", "neutral")
                    if end_reason == "winner" and winner_id == pid:
                        outcome = "winner"
                    elif end_reason in ("unforced", "error") and winner_id != pid:
                        outcome = "unforced_error"
                    elif end_reason in ("unforced", "error") and winner_id == pid:
                        outcome = "forced_error"
                    elif end_reason == "net":
                        outcome = "net"
                    else:
                        outcome = "neutral"
                else:
                    outcome = "neutral"

                won_point = is_last and winner_id == pid
                lost_point = is_last and winner_id is not None and winner_id != pid
                led_to_loss = _compute_led_to_loss_within_k(i, rally_shots, winner_id, k=lookahead_k)

                # ── Build row ─────────────────────────────────────
                row = dict(shot)
                row.update({
                    "zone": zone,
                    "shot_index_in_rally": i,
                    "is_last_in_rally": is_last,
                    "prev_stroke_type": prev_stroke,
                    "prev_player_id": prev_player,
                    "next_stroke_type": next_stroke,
                    "next_player_id": next_player,
                    "prev_gap_s": round(prev_gap, 3),
                    "next_gap_s": round(next_gap, 3),
                    "reaction_time_s": round(reaction_time, 3),
                    "displacement_m": round(displacement, 3),
                    "under_pressure": bool(under_pressure),
                    "shot_outcome": outcome,
                    "won_point": bool(won_point),
                    "lost_point": bool(lost_point),
                    "led_to_loss_within_k": bool(led_to_loss),
                })
                rows.append(row)

                # Update last position per player
                if not (pd.isna(cx) or pd.isna(cy)):
                    last_per_player[pid] = shot

        shot_events = pd.DataFrame(rows)
        artifacts.set_parquet("shot_events", shot_events)

        logger.info(f"ShotContext: enriched {len(shot_events)} shots across "
                    f"{len(rally_groups)} rallies")

        return StageResult.success(
            artifacts={"shot_events": artifacts.path("shot_events")},
            metadata={"n_shots": len(shot_events), "n_rallies": len(rally_groups)},
        )
