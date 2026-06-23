import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.shared.utils import (
    _evaluate_shot, _compute_angle, _angle_score,
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

        # Try to use BST clip data for temporal analysis
        bst_clips = artifacts.get("bst_clips")
        if bst_clips:
            technical = self._analyze_with_bst_clips(shots_df, bst_clips, pose_df)
        else:
            # Fallback to single-frame evaluation
            technical = self._analyze_single_frame(shots_df, pose_df)

        artifacts.set("technical_analytics", technical)

        return StageResult.success(
            artifacts={"technical_analytics": artifacts.path("technical_analytics")},
            metadata={"technical_assessment": technical}
        )

    def _analyze_with_bst_clips(self, shots_df, bst_clips, pose_df):
        """Analyze shot technique using BST clip data for temporal analysis."""
        technical = {}

        # Group shots by player and stroke type
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            player_assessments = {}

            for stroke_type in player_shots["stroke_type"].unique():
                type_shots = player_shots[player_shots["stroke_type"] == stroke_type]

                # Find corresponding BST clips for these shots
                clip_scores = []
                for _, shot in type_shots.iterrows():
                    frame = int(shot["frame"])
                    # Look for BST clip that contains this frame
                    for clip_id, clip_data in bst_clips.items():
                        if frame in clip_data.get("frames", []):
                            # Extract pose keypoints from pose_df for this clip
                            clip_pose = []
                            for clip_frame in clip_data.get("frames", []):
                                pose_row = pose_df[(pose_df["frame"] == clip_frame) & (pose_df["player_id"] == player_id)]
                                if len(pose_row) > 0:
                                    raw = pose_row.iloc[0]["keypoints"]
                                    kps = np.array(raw.tolist()) if hasattr(raw, 'tolist') else np.array(raw)
                                    if kps.shape == (17, 3):
                                        clip_pose.append(kps)
                            # Use clip-level analysis if available
                            clip_score = self._analyze_clip(clip_pose)
                            clip_scores.append(clip_score)
                            break
                
                if clip_scores:
                    player_assessments[stroke_type] = {
                        "avg_score": float(np.mean(clip_scores)),
                        "shot_count": len(type_shots),
                        "scores": clip_scores,
                        "analysis_method": "bst_clip_temporal"
                    }
                else:
                    # Fallback to single-frame evaluation
                    player_assessments[stroke_type] = {
                        "avg_score": 0.5,
                        "shot_count": len(type_shots),
                        "scores": [0.5] * len(type_shots),
                        "analysis_method": "single_frame_fallback"
                    }
            
            technical[player_id] = player_assessments
        
        return technical

    def _analyze_single_frame(self, shots_df, pose_df):
        """Analyze shot technique using single-frame evaluation (fallback)."""
        technical = {}
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            player_poses = pose_df[pose_df["player_id"] == player_id]

            assessments = {}
            for stroke_type in player_shots["stroke_type"].unique():
                type_shots = player_shots[player_shots["stroke_type"] == stroke_type]
                scores = []
                for _, shot in type_shots.iterrows():
                    frame = int(shot["frame"])
                    pose_row = player_poses[player_poses["frame"] == frame]
                    if len(pose_row) > 0:
                        kps = np.array(pose_row.iloc[0]["keypoints"].tolist()) if hasattr(pose_row.iloc[0]["keypoints"], 'tolist') else np.array(pose_row.iloc[0]["keypoints"])
                        score = _evaluate_shot(stroke_type, kps)
                        scores.append(score)

                assessments[stroke_type] = {
                    "avg_score": float(np.mean(scores)) if scores else 0,
                    "shot_count": len(type_shots),
                    "scores": scores,
                    "analysis_method": "single_frame"
                }

            technical[player_id] = assessments

        return technical

    def _analyze_clip(self, clip_pose):
        """Analyze a single BST clip for technical assessment."""
        if not clip_pose or len(clip_pose) < 3:
            return 0.5
        return self._analyze_swing_mechanics(clip_pose)

    def _analyze_swing_mechanics(self, pose_data):
        """Analyze swing using temporal angle trajectories across the clip."""
        if len(pose_data) < 3:
            return 0.5

        elbow_angles = []
        shoulder_angles = []
        for kps in pose_data:
            kps = np.array(kps)
            if kps.shape != (17, 3):
                continue
            handedness = _detect_handedness(kps)
            arm = _get_playing_arm_kps(kps, handedness)
            S, E, W, H = arm["shoulder"], arm["elbow"], arm["wrist"], arm["hip"]
            elbow_angles.append(_compute_angle(W, E, S))
            shoulder_angles.append(_compute_angle(E, S, H))

        if len(elbow_angles) < 3:
            return 0.5

        mid = len(elbow_angles) // 2
        extension = np.mean(elbow_angles[mid:]) - np.mean(elbow_angles[:mid])
        peak_shoulder = max(shoulder_angles)

        scores = [
            min(1.0, max(0.0, extension / 30.0)),
            _angle_score(peak_shoulder, 90, 180, 60),
        ]
        return float(np.mean(scores))
