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
        court = artifacts.get("court") or {}

        from app.models.bst import BSTClassifier, COACH_STROKE_CLASSES
        from app.models.bst_features import BSTFeatureExtractor
        from app.config.settings import settings

        model_path = str(settings.bst_model_path) if settings.bst_model_path else None
        classifier = BSTClassifier(model_path, device=settings.device)
        
        frame_width = config.frame_width if hasattr(config, 'frame_width') else 640
        frame_height = config.frame_height if hasattr(config, 'frame_height') else 480
        extractor = BSTFeatureExtractor(
            frame_width=frame_width,
            frame_height=frame_height,
            court_length=court.get("court_length", 13.4),
            court_width=court.get("court_width", 5.18),
        )

        shots = []
        previous_shots = []
        
        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])
            
            # Extract 144-dim features for rule-based fallback
            features = extractor.extract(
                shuttle_df=shuttle_df,
                pose_df=pose_df,
                target_frame=frame,
                player_id="player_1",
                previous_shots=previous_shots,
            )
            
            # Create a clip for BST prediction
            clip = {
                'JnB': np.zeros((30, 2, 72), dtype=np.float32),
                'shuttle': np.zeros((30, 2), dtype=np.float32),
                'pos': np.zeros((30, 2, 2), dtype=np.float32),
                'video_len': 30,
            }
            
            # Fill shuttle from features
            if shuttle_df is not None and len(shuttle_df) > 0:
                shuttle_row = shuttle_df[shuttle_df['frame'] == frame]
                if len(shuttle_row) > 0:
                    clip['shuttle'][0] = [float(shuttle_row.iloc[0]['x']) / frame_width,
                                          float(shuttle_row.iloc[0]['y']) / frame_height]
            
            stroke_type, confidence = classifier.predict_single(clip)
            
            shot = {
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
            }
            shots.append(shot)
            
            previous_shots.append({
                "stroke_type": stroke_type,
                "frame": frame,
                "stroke_confidence": confidence,
            })

        shots_df = pd.DataFrame(shots)
        artifacts.set_parquet("shots", shots_df)

        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={
                "shot_count": len(shots),
                "stroke_distribution": self._compute_distribution(shots),
            }
        )
    
    @staticmethod
    def _compute_distribution(shots):
        if not shots:
            return {}
        from collections import Counter
        dist = Counter(s["stroke_type"] for s in shots)
        total = len(shots)
        return {k: v / total for k, v in dist.items()}
