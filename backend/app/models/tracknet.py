import numpy as np


class TrackNetV3:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        if model_path:
            import torch
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def predict(self, frames: list[np.ndarray]) -> list[dict]:
        if self.model is None or len(frames) < 5:
            return [{"x": 0, "y": 0, "confidence": 0} for _ in frames]

        import torch
        batch = np.stack(frames[-5:])
        tensor = torch.from_numpy(batch).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        heatmap = output.cpu().numpy()[0, 0]
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        confidence = float(heatmap.max())

        return [{"x": float(x), "y": float(y), "confidence": confidence}]
