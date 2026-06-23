import numpy as np
import pandas as pd

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

            for stroke_type in player_shots["stroke_type"].unique():
                type_shots = player_shots[player_shots["stroke_type"] == stroke_type]

                clip_scores = []
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
                            clip_score = self._analyze_clip(clip_pose, stroke_type)
                            clip_scores.append(clip_score)
                            break

                player_assessments[stroke_type] = {
                    "avg_score": float(np.mean(clip_scores)) if clip_scores else 0.5,
                    "shot_count": len(type_shots),
                    "scores": clip_scores or [0.5] * len(type_shots),
                    "analysis_method": "bst_clip_temporal",
                }

            technical[player_id] = player_assessments

        return technical

    KNEE_BOUNDS = {
        "smash": (150, 170), "clear": (150, 170), "drive": (140, 170),
        "lift": (60, 90), "net_shot": (60, 90),
        "drop": (100, 140), "block": (90, 150), "rush": (80, 120),
    }

    def _analyze_clip(self, clip_pose, stroke_type="smash"):
        if not clip_pose or len(clip_pose) < 3:
            return 0.5
        return self._analyze_swing_mechanics(clip_pose, stroke_type)

    def _analyze_swing_mechanics(self, pose_data, stroke_type="smash"):
        if len(pose_data) < 3:
            return 0.5

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
            return 0.5

        mid = len(elbow_angles) // 2
        extension = np.mean(elbow_angles[mid:]) - np.mean(elbow_angles[:mid])

        peak_elbow_idx = int(np.argmax(elbow_angles))
        follow_through = max(0.0, shoulder_y[peak_elbow_idx] - np.mean(shoulder_y[peak_elbow_idx:]))

        feature_scores = [
            min(1.0, max(0.0, extension / 30.0)),
            _angle_score(max(shoulder_angles), 90, 180, 60),
            _angle_score(float(np.mean(hip_shoulder_seps)), 15, 35, 25),
            _angle_score(float(np.min(knee_angles)), *self.KNEE_BOUNDS.get(stroke_type, (90, 170)), 40),
            _angle_score(follow_through, 10, 40, 20),
        ]
        return float(np.mean(feature_scores))
