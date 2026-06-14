import numpy as np


class RTMPoseEstimator:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        if model_path:
            import onnxruntime as ort
            self.model = ort.InferenceSession(model_path, providers=[f"CUDAExecutionProvider" if "cuda" in device else "CPUExecutionProvider"])

    def estimate(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if self.model is None:
            return np.random.rand(17, 3).astype(np.float32)

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((17, 3), dtype=np.float32)

        resized = np.resize(crop, (192, 192))
        input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)

        output = self.model.run(None, {"input": input_tensor})[0]
        keypoints = output.reshape(17, 3)
        return keypoints
