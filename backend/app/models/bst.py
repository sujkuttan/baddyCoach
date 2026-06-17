"""BST classifier for stroke classification.

Integrates the official BST_CG model from:
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
        return COACH_STROKE_CLASSES[shuttleset_class_id - 13]
    return "unknown"


class BSTClassifier:
    """BST classifier using the official BST_CG model.

    Supports loading from the official weight files and inference
    with proper preprocessing.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.seq_len = 30
        self.classes = COACH_STROKE_CLASSES

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load BST_CG model from checkpoint."""
        try:
            import torch
            from app.models.bst_model import BST_CG

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if not isinstance(checkpoint, dict):
                print(f"BST checkpoint format not recognized: {type(checkpoint)}")
                return

            state_dict = checkpoint

            # Detect dimensions from state_dict
            in_dim = 72
            seq_len = self.seq_len
            n_classes = 25

            for k, v in state_dict.items():
                if 'tcn_pose.net.0.weight' in k:
                    in_dim = v.shape[1]
                if 'mlp_head.mlp.mlp.3.weight' in k:
                    n_classes = v.shape[0]
                if 'embedding_tem' in k:
                    seq_len = v.shape[1] - 1

            has_positions = any('mlp_positions' in k for k in state_dict)

            if has_positions:
                model = BST_CG(
                    in_dim=in_dim,
                    seq_len=seq_len,
                    n_class=n_classes,
                    d_model=100,
                    d_head=128,
                    n_head=6,
                    depth_tem=2,
                    depth_inter=1,
                )
            else:
                print(f"BST checkpoint missing mlp_positions, cannot load")
                return

            model.load_state_dict(state_dict)
            model.to(self.device).eval()

            self.model = model
            self.seq_len = seq_len
            print(f"BST_CG loaded: in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
        except Exception as e:
            print(f"BST load error: {e}")
            import traceback
            traceback.print_exc()

    def predict_from_clips(self, clips: list) -> list:
        """Predict stroke types from prepared BST clips.

        Args:
            clips: List of dicts with keys: JnB, shuttle, pos, video_len

        Returns:
            List of (stroke_type, confidence) tuples
        """
        if self.model is None:
            return [(self._rule_based_predict(clip), 0.5) for clip in clips]

        import torch

        results = []

        for clip in clips:
            try:
                JnB = torch.from_numpy(clip['JnB']).float().unsqueeze(0).to(self.device)
                shuttle = torch.from_numpy(clip['shuttle']).float().unsqueeze(0).to(self.device)
                pos = torch.from_numpy(clip['pos']).float().unsqueeze(0).to(self.device)
                video_len = torch.tensor([clip['video_len']], dtype=torch.long).to(self.device)

                with torch.no_grad():
                    logits = self.model(JnB, shuttle, pos, video_len)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                # If top prediction is "unknown", try the second-best class
                # or fall back to rule-based shuttle trajectory classification
                if pred_idx == 0:
                    second_idx = int(np.argsort(probs)[-2])
                    second_conf = float(probs[second_idx])
                    if second_conf > 0.10:
                        pred_idx = second_idx
                        confidence = second_conf
                    else:
                        stroke_type = self._rule_based_predict(clip)
                        results.append((stroke_type, confidence))
                        continue

                stroke_type = map_to_coach_class(pred_idx)
                results.append((stroke_type, confidence))
            except Exception as e:
                print(f"BST inference error: {e}")
                results.append((self._rule_based_predict(clip), 0.5))

        return results

    def predict_single(self, clip: dict) -> tuple:
        """Predict stroke type for a single clip."""
        results = self.predict_from_clips([clip])
        return results[0] if results else ("unknown", 0.0)

    def _rule_based_predict(self, clip: dict) -> str:
        """Fallback rule-based prediction using shuttle trajectory."""
        shuttle = clip.get('shuttle', np.zeros((30, 2)))

        if len(shuttle) < 2:
            return "clear"

        valid = (shuttle[:, 0] != 0) | (shuttle[:, 1] != 0)
        if valid.sum() < 2:
            return "clear"
        shuttle = shuttle[valid]

        dy = np.diff(shuttle[:, 1])
        dx = np.diff(shuttle[:, 0])
        speed = np.sqrt(dx**2 + dy**2)

        mean_speed = float(np.mean(speed))
        max_speed = float(np.max(speed))
        mean_dy = float(np.mean(dy))
        end_y = float(shuttle[-1, 1])
        start_y = float(shuttle[0, 1])

        if max_speed > 0.15 and mean_dy > 0.05:
            return "smash"
        elif mean_speed < 0.03:
            return "net_shot"
        elif mean_dy < -0.03 and mean_speed > 0.05:
            return "clear"
        elif mean_speed > 0.08 and abs(mean_dy) < 0.02:
            return "drive"
        elif mean_dy > 0.02 and mean_speed > 0.03:
            return "lift"
        elif end_y > 0.7 and mean_speed < 0.06:
            return "drop"
        else:
            return "clear"


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    return arr / np.array([v_width, v_height])


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance."""
    diag = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    diag = np.where(diag == 0, 1, diag)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / diag, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / diag, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / diag
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)
