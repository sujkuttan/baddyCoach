import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.court import (
    image_to_court, foot_midpoint_from_pose, foot_point_from_bbox,
    COURT_LENGTH, COURT_WIDTH,
)
from app.config.settings import settings


class PlayerAttributionStage:
    name = "player_attribution"
    input_keys = ["shots", "shuttle", "players", "court", "pose", "rallies"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"attributed": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        players_data = artifacts.get("players")
        court = artifacts.get("court") or {}
        pose_df = artifacts.get_parquet("pose")
        rallies_df = artifacts.get_parquet("rallies")

        if players_data is None:
            return StageResult.from_error("Player data required for attribution")

        # Check if court is valid
        if not court.get("valid", False):
            return StageResult.from_error("Court detection is invalid, cannot perform attribution")

        court_corners = court.get("corners_pixel", [])
        if court_corners and len(court_corners) >= 3:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = settings.default_frame_height / 2

        H = None
        if "homography" in court and court["homography"] is not None:
            H = np.array(court["homography"])

        shuttle_y_map = {}
        if shuttle_df is not None and len(shuttle_df) > 0:
            shuttle_sorted = shuttle_df.sort_values("frame").reset_index(drop=True)
            shuttle_y_map = dict(zip(shuttle_sorted["frame"].astype(int), shuttle_sorted["y"].astype(float)))

        LOOKBACK = settings.attribution_lookback_frames

        def _shuttle_direction_at(frame):
            y_at = shuttle_y_map.get(frame)
            if y_at is None or abs(y_at) <= 1:
                return None
            for lb in range(1, LOOKBACK + 1):
                y_prev = shuttle_y_map.get(frame - lb)
                if y_prev is not None and abs(y_prev) > 1:
                    dy = y_at - y_prev
                    if abs(dy) > 2:
                        return "player_1" if dy > 0 else "player_2"
                    return None
            return None

        # Initialize player_id column if it doesn't exist
        if "player_id" not in shots_df.columns:
            shots_df["player_id"] = None

        # Try to use BST shuttleset_class_id for Top/Bottom attribution
        if "shuttleset_class_id" in shots_df.columns:
            for idx, shot in shots_df.iterrows():
                class_id = shot.get("shuttleset_class_id", 0)
                if class_id <= 0:
                    continue
                if 1 <= class_id <= 12:
                    for p in players_data.get("players", []):
                        if p.get("side") == "far":
                            shots_df.at[idx, "player_id"] = p["id"]
                            break
                elif 13 <= class_id <= 24:
                    for p in players_data.get("players", []):
                        if p.get("side") == "near":
                            shots_df.at[idx, "player_id"] = p["id"]
                            break

        # Per-shot shuttle direction for unassigned shots
        for idx, shot in shots_df.iterrows():
            if pd.isna(shot.get("player_id")):
                result = _shuttle_direction_at(int(shot["frame"]))
                if result:
                    shots_df.at[idx, "player_id"] = result

        # Rally-based sequential alternation for remaining unassigned shots
        # Uses the last assigned/filled shot to determine the next player,
        # preserving alternation (physical law in badminton) without
        # overwriting model predictions.
        if rallies_df is not None and len(rallies_df) > 0:
            for _, rally in rallies_df.iterrows():
                start_f = int(rally["start_frame"])
                end_f = int(rally["end_frame"])
                rally_mask = (shots_df["frame"] >= start_f) & (shots_df["frame"] <= end_f)
                rally_shots = shots_df[rally_mask].sort_values("frame")
                if len(rally_shots) == 0:
                    continue

                last_player = None
                for _, s in rally_shots.iterrows():
                    pid = s.get("player_id")
                    if pd.notna(pid):
                        last_player = pid
                    elif last_player is not None:
                        next_player = "player_2" if last_player == "player_1" else "player_1"
                        shots_df.at[s.name, "player_id"] = next_player
                        last_player = next_player
                    else:
                        # No assigned shot yet in this rally — use court position
                        y_at = shuttle_y_map.get(int(s["frame"]))
                        fallback = "player_1" if (y_at or court_mid_y) > court_mid_y else "player_2"
                        shots_df.at[s.name, "player_id"] = fallback
                        last_player = fallback

        # Try to use tracking data for any remaining unassigned shots
        tracked_players = artifacts.get("tracked_players")
        if tracked_players:
            for frame, players in tracked_players.items():
                if len(players) >= 2:
                    idx_mask = (shots_df["frame"] == frame) & shots_df["player_id"].isna()
                    if idx_mask.any():
                        sorted_players = sorted(players, key=lambda p: p["bbox"][1], reverse=True)
                        assigned = "player_1" if sorted_players[0]["bbox"][1] > court_mid_y else "player_2"
                        shots_df.loc[idx_mask, "player_id"] = assigned

        # Final fallback: court-position heuristic for any still-unassigned shots
        unassigned = shots_df["player_id"].isna()
        if unassigned.any():
            shots_df.loc[unassigned, "player_id"] = shots_df.loc[unassigned, "frame"].apply(
                lambda f: "player_1" if shuttle_y_map.get(int(f), court_mid_y) > court_mid_y else "player_2"
            )

        if H is not None and pose_df is not None:
            for idx, shot in shots_df.iterrows():
                frame = int(shot["frame"])
                pid = shot.get("player_id", "player_1")
                row_matches = pose_df[(pose_df["frame"] == frame) & (pose_df["player_id"] == pid)]
                if len(row_matches) > 0:
                    row = row_matches.iloc[0]
                    kps = np.array(row["keypoints"].tolist()) if hasattr(row["keypoints"], 'tolist') else np.array(row["keypoints"])
                    if kps.shape == (17, 3) and np.any(kps != 0):
                        foot = foot_midpoint_from_pose(kps[:, :2], kps[:, 2])
                        if foot is not None:
                            try:
                                cx, cy = image_to_court(H, foot)
                                # Clamp to court bounds
                                cx = max(0.0, min(COURT_LENGTH, cx))
                                cy = max(0.0, min(COURT_WIDTH, cy))
                                shots_df.at[idx, "court_x"] = round(cx, 3)
                                shots_df.at[idx, "court_y"] = round(cy, 3)
                            except Exception:
                                pass

        artifacts.set_parquet("shots", shots_df)

        counts = shots_df["player_id"].value_counts().to_dict()
        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"attributed": len(shots_df), "distribution": counts, "court_mid_y": court_mid_y}
        )
