import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class StrokeClassificationStage:
    name = "stroke_classification"
    input_keys = ["hits", "shuttle", "pose", "court"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        hits_df = artifacts.get_parquet("hits")
        if hits_df is None or len(hits_df) == 0:
            return StageResult.success(metadata={"shot_count": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        pose_df = artifacts.get_parquet("pose")

        from app.models.bst import STROKE_CLASSES

        shots = []
        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])

            shuttle_features = self._extract_shuttle_features(shuttle_df, frame) if shuttle_df is not None else np.zeros(6)
            pose_features = self._extract_pose_features(pose_df, frame) if pose_df is not None else np.zeros(8)
            combined = np.concatenate([shuttle_features, pose_features])

            stroke_type, confidence = self._classify(combined, STROKE_CLASSES)

            shots.append({
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
            })

        shots_df = pd.DataFrame(shots)
        artifacts.set_parquet("shots", shots_df)

        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"shot_count": len(shots)}
        )

    def _extract_shuttle_features(self, shuttle_df: pd.DataFrame, frame: int) -> np.ndarray:
        window = shuttle_df[(shuttle_df["frame"] >= frame - 5) & (shuttle_df["frame"] <= frame + 5)]
        if len(window) < 2:
            return np.zeros(6)

        x = window["x"].values
        y = window["y"].values
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        return np.array([
            speed.mean() if len(speed) > 0 else 0,
            speed.max() if len(speed) > 0 else 0,
            x[-1] - x[0],
            y[-1] - y[0],
            np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
        ])

    def _extract_pose_features(self, pose_df: pd.DataFrame, frame: int) -> np.ndarray:
        player_poses = pose_df[pose_df["frame"] == frame]
        if len(player_poses) == 0:
            return np.zeros(8)

        kps = np.array(player_poses.iloc[0]["keypoints"])
        if kps.shape != (17, 3):
            kps = np.array(kps.tolist())
        if kps.shape != (17, 3):
            return np.zeros(8)

        shoulder = kps[5][:2]
        elbow = kps[7][:2]
        wrist = kps[9][:2]
        hip = kps[11][:2]

        return np.array([
            np.sqrt(np.sum((shoulder - elbow)**2)),
            np.sqrt(np.sum((elbow - wrist)**2)),
            np.sqrt(np.sum((shoulder - hip)**2)),
            wrist[1] - shoulder[1],
            wrist[0] - shoulder[0],
            np.arctan2(elbow[1] - shoulder[1], elbow[0] - shoulder[0]),
            np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]),
            np.sqrt(np.sum((wrist - hip)**2)),
        ])

    def _classify(self, features: np.ndarray, classes: list[str]) -> tuple[str, float]:
        idx = np.random.randint(len(classes))
        return classes[idx], 0.8
