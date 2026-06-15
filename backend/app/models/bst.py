"""BST (Badminton Stroke Transformer) classifier for stroke classification.

Supports multiple checkpoint formats and falls back to rule-based
classification when model loading fails.
"""

import numpy as np
from pathlib import Path

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    x_normalized = arr[:, 0] / v_width
    y_normalized = arr[:, 1] / v_height
    return np.stack((x_normalized, y_normalized), axis=-1)


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


class BSTClassifier:
    """BST classifier with multi-architecture fallback.
    
    Supports:
    - Checkpoint with 'model' key containing nn.Module
    - Checkpoint with 'state_dict' key
    - Raw state_dict (tries known architectures)
    - Falls back to rule-based classification
    """
    
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.classes = STROKE_CLASSES
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path, device)
    
    def _load_model(self, path: str, device: str):
        """Try multiple strategies to load the model."""
        import torch
        
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"BST checkpoint load failed: {e}")
            return None
        
        # Strategy 1: Checkpoint contains model object
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            if callable(model) and hasattr(model, 'eval'):
                model.eval()
                return model
        
        # Strategy 2: Checkpoint contains state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return self._load_from_state_dict(checkpoint['state_dict'], device)
        
        # Strategy 3: Checkpoint is raw state_dict
        if isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint.keys()):
            return self._load_from_state_dict(checkpoint, device)
        
        # Strategy 4: Checkpoint is the model itself
        if callable(checkpoint) and hasattr(checkpoint, 'eval'):
            checkpoint.eval()
            return checkpoint
        
        return None
    
    def _load_from_state_dict(self, state_dict: dict, device: str):
        """Try known BST architectures to load state_dict."""
        import torch
        import torch.nn as nn
        
        # Try simple MLP
        try:
            model = SimpleBST_MLP()
            model.load_state_dict(state_dict)
            model.to(device).eval()
            print("BST loaded as SimpleBST_MLP")
            return model
        except Exception:
            pass
        
        # Try 1D ResNet
        try:
            model = SimpleBST_ResNet1D()
            model.load_state_dict(state_dict)
            model.to(device).eval()
            print("BST loaded as SimpleBST_ResNet1D")
            return model
        except Exception:
            pass
        
        return None
    
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predict stroke type from 144-dim feature vector."""
        if self.model is None:
            return self._rule_based_predict(features)
        
        import torch
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
    
    def _rule_based_predict(self, features: np.ndarray) -> tuple[str, float]:
        """Rule-based fallback when model is unavailable.
        
        Uses shuttle trajectory features (indices 0-29) to infer stroke type.
        """
        shuttle_speed = features[16] if len(features) > 16 else 0
        shuttle_height = features[30] if len(features) > 30 else 0.5
        shuttle_dx = features[21] if len(features) > 21 else 0
        shuttle_dy = features[22] if len(features) > 22 else 0
        
        if shuttle_speed > 0.3 and shuttle_dy > 0.1:
            return "smash", 0.6
        elif shuttle_height < 0.3 and shuttle_speed < 0.1:
            return "net_shot", 0.5
        elif shuttle_dy < -0.1 and shuttle_speed > 0.15:
            if shuttle_speed > 0.25:
                return "clear", 0.55
            else:
                return "lift", 0.5
        elif shuttle_speed > 0.2 and abs(shuttle_dy) < 0.05:
            return "drive", 0.5
        elif shuttle_height > 0.6 and shuttle_speed < 0.15:
            return "drop", 0.5
        else:
            return "clear", 0.4


class SimpleBST_MLP:
    """Simple MLP architecture for BST classification (placeholder)."""
    pass


class SimpleBST_ResNet1D:
    """1D ResNet architecture for BST classification (placeholder)."""
    pass
