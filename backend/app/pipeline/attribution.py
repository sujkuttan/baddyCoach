import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.court import image_to_court, foot_midpoint_from_pose, foot_point_from_bbox


class PlayerAttributionStage:
    name = "player_attribution"
    input_keys = ["shots", "shuttle", "players", "court", "pose"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"attributed": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        players_data = artifacts.get("players")
        court = artifacts.get("court") or {}
        pose_df = artifacts.get_parquet("pose")

        if players_data is None:
            return StageResult.from_error("Player data required for attribution")

        players = {p["id"]: p for p in players_data["players"]}

        court_corners = court.get("corners_pixel", [])
        if court_corners and len(court_corners) >= 3:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = 360

        H = None
        if "homography" in court and court["homography"] is not None:
            H = np.array(court["homography"])

        pose_lookup = {}
        if pose_df is not None:
            for _, row in pose_df.iterrows():
                frame = int(row["frame"])
                pid = row.get("player_id", "player_1")
                kps = np.array(row["keypoints"].tolist()) if hasattr(row["keypoints"], 'tolist') else np.array(row["keypoints"])
                if kps.shape == (17, 3) and np.any(kps != 0):
                    if frame not in pose_lookup:
                        pose_lookup[frame] = {}
                    pose_lookup[frame][pid] = kps

        shuttle_y_map = {}
        shuttle_conf_map = {}
        if shuttle_df is not None and len(shuttle_df) > 0:
            shuttle_sorted = shuttle_df.sort_values("frame").reset_index(drop=True)
            shuttle_y_map = dict(zip(shuttle_sorted["frame"].astype(int), shuttle_sorted["y"].astype(float)))

        LOOKBACK = 5
        attributed = []
        court_coords = []

        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            did_attribute = False

            y_at = shuttle_y_map.get(frame)
            if y_at is not None:
                y_before = None
                for lookback in range(1, LOOKBACK + 1):
                    y_prev = shuttle_y_map.get(frame - lookback)
                    if y_prev is not None and abs(y_prev) > 1:
                        y_before = y_prev
                        break

                if y_before is not None and abs(y_at) > 1:
                    dy = y_at - y_before
                    if abs(dy) > 2:
                        attributed.append("player_1" if dy > 0 else "player_2")
                        did_attribute = True

            if not did_attribute and frame in pose_lookup:
                foot_positions = {}
                for pid, kps in pose_lookup[frame].items():
                    foot = foot_midpoint_from_pose(kps[:, :2], kps[:, 2])
                    if foot is None:
                        foot = foot_point_from_bbox([0, 0, 1280, 720])
                    foot_positions[pid] = foot

                if len(foot_positions) == 2:
                    pids = list(foot_positions.keys())
                    y1 = foot_positions[pids[0]][1]
                    y2 = foot_positions[pids[1]][1]
                    player_id = pids[0] if y1 > y2 else pids[1]
                    attributed.append(player_id)
                    did_attribute = True

                    if H is not None:
                        foot = foot_positions[player_id]
                        try:
                            cx, cy = image_to_court(H, foot)
                            court_coords.append({"frame": frame, "court_x": round(cx, 3), "court_y": round(cy, 3)})
                        except Exception:
                            court_coords.append({"frame": frame, "court_x": None, "court_y": None})

            if not did_attribute:
                player_id = self._assign_player(frame, shuttle_df, players, court_mid_y)
                attributed.append(player_id)
                court_coords.append({"frame": frame, "court_x": None, "court_y": None})

        shots_df["player_id"] = attributed

        if court_coords:
            court_df = pd.DataFrame(court_coords)
            if "court_x" not in shots_df.columns:
                shots_df = shots_df.merge(court_df, on="frame", how="left")

        artifacts.set_parquet("shots", shots_df)

        counts = shots_df["player_id"].value_counts().to_dict()
        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"attributed": len(shots_df), "distribution": counts, "court_mid_y": court_mid_y}
        )

    def _assign_player(self, frame: int, shuttle_df: pd.DataFrame | None, players: dict, court_mid_y: float) -> str:
        if shuttle_df is None or len(players) == 0:
            return list(players.keys())[0] if players else "unknown"

        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        if len(shuttle_row) == 0:
            return list(players.keys())[0]

        shuttle_y = float(shuttle_row.iloc[0]["y"])

        player_list = list(players.values())
        if len(player_list) == 2:
            sides = [p["side"] for p in player_list]
            if shuttle_y > court_mid_y and "near" in sides:
                return next(p["id"] for p in player_list if p["side"] == "near")
            elif shuttle_y <= court_mid_y and "far" in sides:
                return next(p["id"] for p in player_list if p["side"] == "far")

        return player_list[0]["id"]
