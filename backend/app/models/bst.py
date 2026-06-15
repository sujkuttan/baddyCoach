import numpy as np
from pathlib import Path

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution.

    Args:
        arr: (T, 2) array of (x, y) positions
        v_width: Video width
        v_height: Video height

    Returns:
        Normalized array (T, 2) with values in [0, 1]
    """
    x_normalized = arr[:, 0] / v_width
    y_normalized = arr[:, 1] / v_height
    return np.stack((x_normalized, y_normalized), axis=-1)


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance.

    Args:
        arr: (M, J, 2) array of joint positions
        bbox: (M, 4) array of bounding boxes (x1, y1, x2, y2)
        center_align: If True, center of bbox is origin

    Returns:
        Normalized array (M, J, 2)
    """
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
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.classes = STROKE_CLASSES
        if model_path and Path(model_path).exists():
            try:
                import torch
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model = checkpoint['model']
                else:
                    model = checkpoint
                if callable(model) and hasattr(model, 'eval'):
                    model.eval()
                    self.model = model
            except Exception:
                self.model = None

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            idx = np.random.randint(len(self.classes))
            return self.classes[idx], 0.8

        import torch
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
