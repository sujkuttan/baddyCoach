"""BST classifier for stroke classification.

Integrates the official BST_CG / BST_CG_AP model from:
https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
"""

import numpy as np
from pathlib import Path
from typing import Optional


COACH_STROKE_CLASSES = [
    "net_shot", "block", "smash", "lift", "clear", "drive",
    "drop", "push", "rush", "cross_court", "short_serve", "long_serve"
]

SHUTTLESET_CLASSES = [
    'unknown', 'Top_net_shot', 'Top_block', 'Top_smash', 'Top_lift',
    'Top_clear', 'Top_drive', 'Top_drop', 'Top_push', 'Top_rush',
    'Top_cross_court', 'Top_short_serve', 'Top_long_serve',
    'Bottom_net_shot', 'Bottom_block', 'Bottom_smash', 'Bottom_lift',
    'Bottom_clear', 'Bottom_drive', 'Bottom_drop', 'Bottom_push',
    'Bottom_rush', 'Bottom_cross_court', 'Bottom_short_serve', 'Bottom_long_serve'
]


def map_to_coach_class(shuttleset_class_id: int) -> str:
    """Map ShuttleSet class ID to simplified coaching class."""
    if shuttleset_class_id == 0:
        return "unknown"
    elif 1 <= shuttleset_class_id <= 12:
        return COACH_STROKE_CLASSES[shuttleset_class_id - 1]
    elif 13 <= shuttleset_class_id <= 24:
        base_class = COACH_STROKE_CLASSES[shuttleset_class_id - 13]
        return base_class
    return "unknown"


def get_shuttleset_class_info(class_id: int) -> tuple:
    """Return (stroke_type, side) from ShuttleSet class ID.

    side is 'top', 'bottom', or None for unknown.
    """
    if class_id == 0:
        return "unknown", None
    elif 1 <= class_id <= 12:
        return COACH_STROKE_CLASSES[class_id - 1], "top"
    elif 13 <= class_id <= 24:
        return COACH_STROKE_CLASSES[class_id - 13], "bottom"
    return "unknown", None


class BSTClassifier:
    """BST classifier using the official BST_CG / BST_CG_AP model.

    Supports loading from the official weight files and inference
    with proper preprocessing. Batch size is auto-detected from
    available GPU VRAM via gpu_batch config.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", default_seq_len: int = 30, batch_size: Optional[int] = None):
        self.device = device
        self.model = None
        self.seq_len = default_seq_len
        self.classes = COACH_STROKE_CLASSES
        self.batch_size = batch_size if batch_size is not None else self._default_batch_size()

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    @staticmethod
    def _default_batch_size() -> int:
        try:
            from app.config.gpu_batch import get_gpu_batch_config
            cfg = get_gpu_batch_config()
            return int(cfg.get("bst_batch", 32))
        except Exception:
            return 32

    def _load_model(self, path: str):
        """Load BST_CG or BST_CG_AP model from checkpoint."""
        from app.pipeline.shared.models import _checked_load, record_model_health
        try:
            import torch
            from app.models.bst_model import BST_CG, BST_CG_AP

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if not isinstance(checkpoint, dict):
                print(f"BST checkpoint format not recognized: {type(checkpoint)}")
                record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                            "n_missing": 0, "n_unexpected": 0,
                                            "core_missing": ["checkpoint not a dict"]})
                return

            state_dict = checkpoint

            # Detect dimensions from state_dict
            in_dim = 72
            seq_len = None
            n_classes = 25

            for k, v in state_dict.items():
                if 'tcn_pose.net.0.weight' in k:
                    in_dim = v.shape[1]
                if 'mlp_head.mlp.mlp.3.weight' in k:
                    n_classes = v.shape[0]
                if 'embedding_tem' in k:
                    seq_len = v.shape[1] - 1

            if seq_len is None:
                print("BST checkpoint missing embedding_tem, cannot determine seq_len")
                record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                            "n_missing": 0, "n_unexpected": 0,
                                            "core_missing": ["missing embedding_tem"]})
                return

            has_positions = any('mlp_positions' in k for k in state_dict)
            if not has_positions:
                print("BST checkpoint missing mlp_positions, cannot load")
                record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                            "n_missing": 0, "n_unexpected": 0,
                                            "core_missing": ["missing mlp_positions"]})
                return

            # Always use BST_CG_AP (AimPlayer) — all 94-key checkpoints (CG, CG_AP,
            # ckpts/bst) have identical state dicts since nn.CosineSimilarity has no
            # trainable params. BST_CG_AP is a strict superset of BST_CG with added
            # AimPlayer cosine-similarity weighting that improves player-aware
            # classification. No extra parameters needed beyond BST_CG's.
            model_class = BST_CG_AP
            model = model_class(
                in_dim=in_dim,
                seq_len=seq_len,
                n_class=n_classes,
                d_model=100,
                d_head=128,
                n_head=6,
                depth_tem=2,
                depth_inter=1,
            )

            status = _checked_load(model, state_dict,
                                   core_prefixes=("tcn_pose", "mlp_head", "embedding_tem"))
            record_model_health("bst", status)

            if not status["loaded"]:
                print(f"WARNING: BST core layers missing ({status['core_missing']}). "
                      "Model set to None — honest fallback.")
                return

            model.to(self.device).eval()

            self.model = model
            self.seq_len = seq_len
            print(f"BST loaded: class=BST_CG_AP (AimPlayer), in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
        except Exception as e:
            print(f"BST load error: {e}")
            import traceback
            traceback.print_exc()
            record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                        "n_missing": 0, "n_unexpected": 0,
                                        "core_missing": [str(e)]})

    def predict_from_clips(self, clips: list, batch_size: Optional[int] = None) -> list:
        """Predict stroke types from prepared BST clips.

        Args:
            clips: List of dicts with keys: JnB, shuttle, pos, video_len
            batch_size: Number of clips to process in parallel.
                       Defaults to self.batch_size (auto-detected from GPU VRAM).

        Returns:
            List of (stroke_type, confidence, raw_class_id) tuples
        """
        batch_size = batch_size or self.batch_size
        if self.model is None:
            return [(self._rule_based_predict(clip), 0.5, 0) for clip in clips]

        import torch

        results = [None] * len(clips)

        for batch_start in range(0, len(clips), batch_size):
            batch_end = min(batch_start + batch_size, len(clips))
            batch_clips = clips[batch_start:batch_end]

            try:
                JnB = torch.from_numpy(
                    np.stack([c['JnB'] for c in batch_clips])
                ).float().to(self.device)
                shuttle = torch.from_numpy(
                    np.stack([c['shuttle'] for c in batch_clips])
                ).float().to(self.device)
                pos = torch.from_numpy(
                    np.stack([c['pos'] for c in batch_clips])
                ).float().to(self.device)
                video_len = torch.tensor(
                    [c['video_len'] for c in batch_clips], dtype=torch.long
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(JnB, shuttle, pos, video_len)
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

                for j in range(len(batch_clips)):
                    pred_idx = int(np.argmax(probs[j]))
                    confidence = float(probs[j][pred_idx])

                    if pred_idx == 0:
                        second_idx = int(np.argsort(probs[j])[-2])
                        second_conf = float(probs[j][second_idx])
                        if second_conf > 0.3:
                            pred_idx = second_idx
                            confidence = second_conf
                        else:
                            # Model confidently predicted "unknown" with no viable
                            # second-best — fall back to rule-based with low confidence
                            fallback = self._rule_based_predict(batch_clips[j])
                            rule_conf = min(confidence, 0.3)
                            results[batch_start + j] = (fallback, rule_conf, 0)
                            continue

                    stroke_type = map_to_coach_class(pred_idx)
                    results[batch_start + j] = (stroke_type, confidence, pred_idx)

            except Exception as e:
                print(f"BST batch inference error: {e}")
                for j in range(len(batch_clips)):
                    if results[batch_start + j] is None:
                        fallback = self._rule_based_predict(batch_clips[j])
                        results[batch_start + j] = (fallback, 0.5, 0)

        return results

    def predict_single(self, clip: dict) -> tuple:
        """Predict stroke type for a single clip.

        Returns:
            (stroke_type, confidence, raw_class_id)
        """
        results = self.predict_from_clips([clip])
        return results[0] if results else ("unknown", 0.0, 0)

    def _rule_based_predict(self, clip: dict) -> str:
        """Fallback rule-based prediction using shuttle trajectory.

        Only classifies when the trajectory clearly matches one of the
        heuristics below. Returns 'unknown' for ambiguous trajectories,
        matching the BST model's class 0 semantics, so downstream stages
        can distinguish genuine predictions from forced guesses.
        """
        seq_len = self.seq_len if self.seq_len is not None else 30
        shuttle = clip.get('shuttle', np.zeros((seq_len, 2)))

        if len(shuttle) < 2:
            return "unknown"

        valid = (shuttle[:, 0] != 0) | (shuttle[:, 1] != 0)
        if valid.sum() < 2:
            return "unknown"
        shuttle = shuttle[valid]

        dy = np.diff(shuttle[:, 1])
        dx = np.diff(shuttle[:, 0])
        speed = np.sqrt(dx**2 + dy**2)

        mean_speed = float(np.mean(speed))
        max_speed = float(np.max(speed))
        mean_dy = float(np.mean(dy))
        end_y = float(shuttle[-1, 1])

        if max_speed > 0.15 and mean_dy > 0.05:
            return "smash"
        elif mean_speed < 0.03:
            return "net_shot"
        elif mean_dy < -0.03 and mean_speed > 0.05:
            return "clear"
        elif mean_speed > 0.08 and abs(mean_dy) < 0.02:
            return "drive"
        elif mean_dy > 0.04 and mean_speed > 0.05 and end_y > 0.5:
            return "lift"
        elif end_y > 0.7 and mean_speed < 0.06:
            return "drop"
        else:
            return "unknown"


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    from app.pipeline.shared.bst_preproc import normalize_shuttlecock as _ns
    return _ns(arr, v_width, v_height)


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance (batched)."""
    from app.pipeline.shared.bst_preproc import normalize_joints_batched
    return normalize_joints_batched(arr, bbox, center_align)
