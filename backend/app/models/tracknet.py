import numpy as np
from pathlib import Path


class TrackNetV3:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.input_height = 288
        self.input_width = 512
        if model_path and Path(model_path).exists():
            import torch
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def _preprocess(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch
        import cv2
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed.append(normalized)
        batch = np.stack(processed)
        batch = batch.transpose(3, 0, 1, 2)
        return torch.from_numpy(batch).unsqueeze(0).float().to(self.device)

    def _postprocess(self, output: np.ndarray, original_width: int, original_height: int) -> dict:
        y_idx, x_idx = np.unravel_index(output.argmax(), output.shape)
        confidence = float(output.max())
        x = x_idx * original_width / self.input_width
        y = y_idx * original_height / self.input_height
        return {"x": float(x), "y": float(y), "confidence": confidence}

    def predict(self, frames: list[np.ndarray], original_size: tuple | None = None) -> list[dict]:
        if self.model is None or len(frames) < 3:
            h = frames[0].shape[0] if frames else 720
            w = frames[0].shape[1] if frames else 1280
            return [{"x": 0, "y": 0, "confidence": 0}]
        import torch
        original_width = original_size[0] if original_size else frames[0].shape[1]
        original_height = original_size[1] if original_size else frames[0].shape[0]
        input_frames = frames[-3:]
        tensor = self._preprocess(input_frames)
        with torch.no_grad():
            output = self.model(tensor)
        heatmap = output.cpu().numpy()[0, 0]
        return [self._postprocess(heatmap, original_width, original_height)]

    def predict_batch(self, frames: list[np.ndarray], batch_size: int = 3, original_size: tuple | None = None) -> list[dict]:
        if len(frames) < 3:
            return [{"x": 0, "y": 0, "confidence": 0} for _ in frames]
        results = []
        for i in range(2, len(frames)):
            window = frames[i-2:i+1]
            pred = self.predict(window, original_size)
            results.append(pred[0])
        return results