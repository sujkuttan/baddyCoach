import numpy as np
import pandas as pd

from app.config.settings import settings
from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.utils import (
    _compute_angle, _angle_score,
    _get_playing_arm_kps, _detect_handedness,
)


class TechnicalAnalyticsStage:
    name = "technical_analytics"
    input_keys = ["shots", "pose", "shuttle", "court"]
    output_keys = ["technical_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court")

        if shots_df is None or pose_df is None:
            return StageResult.from_error("Shot and pose data required")

        bst_clips = artifacts.get("bst_clips")
        technical = self._analyze_with_bst_clips(shots_df, bst_clips, pose_df)

        angular_vel = self._compute_angular_velocity_all(pose_df, float(config.processing_fps or settings.fps))
        symmetry = self._compute_symmetry_score(pose_df, angular_vel)
        keyframes = self._select_keyframes(pose_df)
        technical["angular_velocity"] = angular_vel
        technical["symmetry_score"] = symmetry
        technical["keyframes"] = keyframes

        artifacts.set("technical_analytics", technical)

        return StageResult.success(
            artifacts={"technical_analytics": artifacts.path("technical_analytics")},
            metadata={"technical_assessment": technical}
        )

    def _analyze_with_bst_clips(self, shots_df, bst_clips, pose_df):
        """Analyze shot technique using BST clip data for temporal analysis."""
        technical = {}

        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            player_assessments = {}
            player_features = {}

            for stroke_type in player_shots["stroke_type"].unique():
                type_shots = player_shots[player_shots["stroke_type"] == stroke_type]

                clip_scores = []
                all_feature_sets = []

                for _, shot in type_shots.iterrows():
                    frame = int(shot["frame"])
                    for clip_id, clip_data in bst_clips.items():
                        if frame in clip_data.get("frames", []):
                            clip_pose = []
                            for clip_frame in clip_data.get("frames", []):
                                pose_row = pose_df[(pose_df["frame"] == clip_frame) & (pose_df["player_id"] == player_id)]
                                if len(pose_row) > 0:
                                    raw = pose_row.iloc[0]["keypoints"]
                                    kps = np.array(raw.tolist()) if hasattr(raw, 'tolist') else np.array(raw)
                                    if kps.shape == (17, 3):
                                        clip_pose.append(kps)
                            clip_score, features = self._analyze_clip_detailed(clip_pose, stroke_type)
                            clip_scores.append(clip_score)
                            if features:
                                all_feature_sets.append(features)
                            break

                player_assessments[stroke_type] = {
                    "avg_score": float(np.mean(clip_scores)) if clip_scores else 0.5,
                    "shot_count": len(type_shots),
                    "scores": clip_scores or [0.5] * len(type_shots),
                    "analysis_method": "bst_clip_temporal",
                }

                # Persist per-stroke feature aggregates for technique reference (C1)
                if all_feature_sets:
                    aggregated = {}
                    for fname in all_feature_sets[0]:
                        vals = [fs[fname] for fs in all_feature_sets if fname in fs]
                        if vals:
                            aggregated[fname] = {
                                "p50": float(np.median(vals)),
                                "mean": float(np.mean(vals)),
                                "std": float(np.std(vals)),
                                "n": len(vals),
                            }
                    player_features[stroke_type] = aggregated

            technical[player_id] = player_assessments

            # Store feature aggregates separately for reference comparison
            if player_features:
                technical[f"{player_id}_features"] = player_features

        return technical

    # ── Angular velocity (Flagami/BadmintonCoach adaptation) ──
    @staticmethod
    def _compute_angular_velocity_all(pose_df: pd.DataFrame, fps: float) -> dict:
        """Compute angular velocity (°/s) per joint per player.

        Returns { player_id: { joint_name: {"mean": ..., "max": ...} } }.
        Joints: right_elbow, left_elbow, right_knee, left_knee, torso_lean.
        """
        if pose_df is None or len(pose_df) < 3:
            return {}

        result = {}
        for pid in pose_df["player_id"].unique():
            player = pose_df[pose_df["player_id"] == pid].sort_values("frame")
            frames = player["frame"].values
            if len(frames) < 3:
                continue

            angle_series: dict[str, list[float]] = {
                "right_elbow": [], "left_elbow": [],
                "right_knee": [], "left_knee": [],
                "torso_lean": [],
            }

            for _, row in player.iterrows():
                raw = row["keypoints"]
                kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
                if kps.shape != (17, 3):
                    continue
                conf = kps[:, 2]

                JOINT = {"R_S": 6, "R_E": 8, "R_W": 10,   "R_H": 12, "R_K": 14, "R_A": 16,
                         "L_S": 5, "L_E": 7, "L_W": 9,    "L_H": 11, "L_K": 13, "L_A": 15}

                def _angle(p1, p2, p3):
                    v1 = p1 - p2; v2 = p3 - p2
                    n = np.linalg.norm(v1) * np.linalg.norm(v2)
                    return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))

                def pts(*idx):
                    return [kps[i, :2] for i in idx] if all(conf[i] > 0.3 for i in idx) else None

                p = pts(JOINT["R_S"], JOINT["R_E"], JOINT["R_W"])
                if p: angle_series["right_elbow"].append(_angle(*p))
                p = pts(JOINT["L_S"], JOINT["L_E"], JOINT["L_W"])
                if p: angle_series["left_elbow"].append(_angle(*p))
                p = pts(JOINT["R_H"], JOINT["R_K"], JOINT["R_A"])
                if p: angle_series["right_knee"].append(_angle(*p))
                p = pts(JOINT["L_H"], JOINT["L_K"], JOINT["L_A"])
                if p: angle_series["left_knee"].append(_angle(*p))

                # Torso lean
                if all(conf[i] > 0.3 for i in [JOINT["L_S"], JOINT["R_S"], JOINT["L_H"], JOINT["R_H"]]):
                    s_mid = (kps[JOINT["L_S"], :2] + kps[JOINT["R_S"], :2]) / 2
                    h_mid = (kps[JOINT["L_H"], :2] + kps[JOINT["R_H"], :2]) / 2
                    tv = s_mid - h_mid
                    n = np.linalg.norm(tv)
                    if n > 1e-6:
                        angle_series["torso_lean"].append(float(np.degrees(np.arccos(np.clip(np.dot(tv/n, np.array([0.0, -1.0])), -1.0, 1.0)))))

            # Compute velocity from angle series
            vel: dict[str, dict] = {}
            for key, vals in angle_series.items():
                if len(vals) < 3:
                    continue
                arr = np.array(vals)
                diffs = np.abs(np.diff(arr))
                dt = 1.0 / max(fps, 1)
                vels = diffs / dt
                vel[key] = {"mean": round(float(np.mean(vels)), 1), "max": round(float(np.max(vels)), 1)}

            result[pid] = vel

        return result

    # ── Symmetry score (Flagami/BadmintonCoach adaptation) ──
    @staticmethod
    def _compute_symmetry_score(pose_df: pd.DataFrame, angular_vel: dict | None = None) -> float:
        """Left-right symmetry score (0-100): 0° mean difference = full score, ≥45° = 0."""
        if pose_df is None or len(pose_df) < 3:
            return 50.0

        l_elbow, r_elbow = [], []
        l_knee, r_knee = [], []
        for _, row in pose_df.iterrows():
            raw = row["keypoints"]
            kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
            if kps.shape != (17, 3):
                continue
            if kps[7, 2] > 0.3 and kps[5, 2] > 0.3 and kps[9, 2] > 0.3:
                l_elbow.append(_compute_angle(kps[9, :2], kps[7, :2], kps[5, :2]))
            if kps[8, 2] > 0.3 and kps[6, 2] > 0.3 and kps[10, 2] > 0.3:
                r_elbow.append(_compute_angle(kps[10, :2], kps[8, :2], kps[6, :2]))
            if kps[13, 2] > 0.3 and kps[11, 2] > 0.3 and kps[15, 2] > 0.3:
                l_knee.append(_compute_angle(kps[15, :2], kps[13, :2], kps[11, :2]))
            if kps[14, 2] > 0.3 and kps[12, 2] > 0.3 and kps[16, 2] > 0.3:
                r_knee.append(_compute_angle(kps[16, :2], kps[14, :2], kps[12, :2]))

        diffs = []
        if l_elbow and r_elbow:
            diffs.append(abs(float(np.mean(l_elbow)) - float(np.mean(r_elbow))))
        if l_knee and r_knee:
            diffs.append(abs(float(np.mean(l_knee)) - float(np.mean(r_knee))))
        if not diffs:
            return 50.0
        mean_diff = float(np.mean(diffs))
        return round(max(0.0, 100.0 - (mean_diff / 45.0) * 100.0), 1)

    # ── Keyframe selection (Flagami/BadmintonCoach adaptation) ──
    @staticmethod
    def _select_keyframes(pose_df: pd.DataFrame) -> list[dict]:
        """Pick up to 5 representative frames by motion salience.

        Returns list sorted by frame_id, each with frame, timestamp,
        label (max_arm_swing | deepest_squat | max_torso_deviation |
               most_asymmetric | most_complete), and key angles.
        """
        if pose_df is None or len(pose_df) < 5:
            return []

        frames_data: list[dict] = []
        for _, row in pose_df.iterrows():
            raw = row["keypoints"]
            kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
            if kps.shape != (17, 3):
                continue
            conf = kps[:, 2]
            angles: dict[str, float] = {}
            JOINT = {"R_S": 6, "R_E": 8, "R_W": 10, "R_H": 12, "R_K": 14, "R_A": 16,
                     "L_S": 5, "L_E": 7, "L_W": 9,  "L_H": 11, "L_K": 13, "L_A": 15}

            def _a(p1, p2, p3):
                v1 = p1 - p2; v2 = p3 - p2
                n = np.linalg.norm(v1) * np.linalg.norm(v2)
                return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))

            def pts(*idx):
                return [kps[i, :2] for i in idx] if all(conf[i] > 0.3 for i in idx) else None

            p = pts(JOINT["R_S"], JOINT["R_E"], JOINT["R_W"])
            if p: angles["right_elbow"] = _a(*p)
            p = pts(JOINT["L_S"], JOINT["L_E"], JOINT["L_W"])
            if p: angles["left_elbow"] = _a(*p)
            p = pts(JOINT["R_H"], JOINT["R_K"], JOINT["R_A"])
            if p: angles["right_knee"] = _a(*p)
            p = pts(JOINT["L_H"], JOINT["L_K"], JOINT["L_A"])
            if p: angles["left_knee"] = _a(*p)
            if all(conf[i] > 0.3 for i in [JOINT["L_S"], JOINT["R_S"], JOINT["L_H"], JOINT["R_H"]]):
                s_mid = (kps[JOINT["L_S"], :2] + kps[JOINT["R_S"], :2]) / 2
                h_mid = (kps[JOINT["L_H"], :2] + kps[JOINT["R_H"], :2]) / 2
                tv = s_mid - h_mid
                n = np.linalg.norm(tv)
                if n > 1e-6:
                    angles["torso_lean"] = float(np.degrees(np.arccos(np.clip(np.dot(tv/n, np.array([0.0, -1.0])), -1.0, 1.0))))

            if angles:
                frames_data.append({"frame": int(row["frame"]), "angles": angles})

        if not frames_data:
            return []

        candidates: dict[str, dict] = {}

        c = [f for f in frames_data if "right_elbow" in f["angles"]]
        if c: candidates["max_arm_swing"] = min(c, key=lambda f: f["angles"]["right_elbow"])

        c = [f for f in frames_data if "right_knee" in f["angles"] or "left_knee" in f["angles"]]
        if c:
            candidates["deepest_squat"] = min(c, key=lambda f: min(
                f["angles"].get("right_knee", 180), f["angles"].get("left_knee", 180)))

        c = [f for f in frames_data if "torso_lean" in f["angles"]]
        if c: candidates["max_torso_deviation"] = max(c, key=lambda f: abs(f["angles"]["torso_lean"]))

        c = [f for f in frames_data if "left_elbow" in f["angles"] and "right_elbow" in f["angles"]]
        if c: candidates["most_asymmetric"] = max(c, key=lambda f: abs(f["angles"]["left_elbow"] - f["angles"]["right_elbow"]))

        candidates["most_complete"] = max(frames_data, key=lambda f: len(f["angles"]))

        fps = getattr(settings, "fps", 30.0)
        seen: dict[int, dict] = {}
        for label, fr in candidates.items():
            fid = fr["frame"]
            if fid not in seen:
                seen[fid] = {"frame": fid, "timestamp": round(fid / fps, 2), "label": label, "angles": fr["angles"]}

        return sorted(seen.values(), key=lambda f: f["frame"])[:5]

    KNEE_BOUNDS = {
        "smash": (150, 170), "clear": (150, 170), "drive": (140, 170),
        "lift": (60, 90), "net_shot": (60, 90),
        "drop": (100, 140), "block": (90, 150), "rush": (80, 120),
    }

    def _analyze_clip(self, clip_pose, stroke_type="smash"):
        if not clip_pose or len(clip_pose) < 3:
            return 0.5
        score, _ = self._analyze_clip_detailed(clip_pose, stroke_type)
        return score

    def _analyze_clip_detailed(self, clip_pose, stroke_type="smash"):
        """Analyze a clip and return (score, feature_dict)."""
        if not clip_pose or len(clip_pose) < 3:
            return 0.5, {}
        return self._analyze_swing_mechanics_detailed(clip_pose, stroke_type)

    def _analyze_swing_mechanics_detailed(self, pose_data, stroke_type="smash"):
        """Temporal swing analysis returning (score, per-feature dict)."""
        if len(pose_data) < 3:
            return 0.5, {}

        elbow_angles = []
        shoulder_angles = []
        hip_shoulder_seps = []
        knee_angles = []
        shoulder_y = []

        for kps in pose_data:
            kps = np.array(kps)
            if kps.shape != (17, 3):
                continue
            handedness = _detect_handedness(kps)
            arm = _get_playing_arm_kps(kps, handedness)
            S, E, W, H = arm["shoulder"], arm["elbow"], arm["wrist"], arm["hip"]
            K, A = arm["knee"], arm["ankle"]

            elbow_angles.append(_compute_angle(W, E, S))
            shoulder_angles.append(_compute_angle(E, S, H))
            knee_angles.append(_compute_angle(A, K, H))
            shoulder_y.append(S[1])

            shoulder_mid = (kps[5][:2] + kps[6][:2]) / 2.0
            hip_mid = (kps[11][:2] + kps[12][:2]) / 2.0
            vec = hip_mid - shoulder_mid
            angle_from_vertical = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))
            hip_shoulder_seps.append(angle_from_vertical)

        if len(elbow_angles) < 3:
            return 0.5, {}

        mid = len(elbow_angles) // 2
        extension = np.mean(elbow_angles[mid:]) - np.mean(elbow_angles[:mid])

        peak_elbow_idx = int(np.argmax(elbow_angles))
        follow_through = max(0.0, shoulder_y[peak_elbow_idx] - np.mean(shoulder_y[peak_elbow_idx:]))

        features = {
            "elbow_extension": float(extension),
            "peak_shoulder_angle": float(max(shoulder_angles)),
            "hip_shoulder_sep": float(np.mean(hip_shoulder_seps)),
            "min_knee_angle": float(np.min(knee_angles)),
            "follow_through": float(follow_through),
        }

        feature_scores = [
            min(1.0, max(0.0, extension / 30.0)),
            _angle_score(max(shoulder_angles), 90, 180, 60),
            _angle_score(float(np.mean(hip_shoulder_seps)), 15, 35, 25),
            _angle_score(float(np.min(knee_angles)), *self.KNEE_BOUNDS.get(stroke_type, (90, 170)), 40),
            _angle_score(follow_through, 10, 40, 20),
        ]
        return float(np.mean(feature_scores)), features
