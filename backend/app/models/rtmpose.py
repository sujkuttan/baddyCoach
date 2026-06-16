import numpy as np
from pathlib import Path


class RTMPoseEstimator:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.input_height = 256
        self.input_width = 192

        if model_path and Path(model_path).exists():
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if 'cuda' in device:
                providers.insert(0, 'CUDAExecutionProvider')
            self.model = ort.InferenceSession(model_path, providers=providers)

    def _preprocess(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop and preprocess person region.

        Args:
            frame: Full video frame (H, W, C)
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)

        import cv2
        resized = cv2.resize(crop, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        return tensor

    def _postprocess(
        self,
        output: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple
    ) -> np.ndarray:
        """Convert model output to keypoints.

        Args:
            output: Model output (1, 17, 3) or similar
            bbox: Original bounding box
            frame_shape: (height, width) of original frame

        Returns:
            Keypoints array (17, 3) with (x, y, confidence)
        """
        if len(output.shape) == 3:
            keypoints = output[0]
        elif len(output.shape) == 4:
            keypoints = output[0, 0]
        else:
            keypoints = output.reshape(17, 3)

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        keypoints[:, 0] = x1 + keypoints[:, 0] * bbox_width
        keypoints[:, 1] = y1 + keypoints[:, 1] * bbox_height

        return keypoints

    def estimate(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Estimate pose for a single person.

        Args:
            frame: Video frame (H, W, C)
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Keypoints (17, 3) with (x, y, confidence)
        """
        if self.model is None:
            return np.random.rand(17, 3).astype(np.float32)

        input_tensor = self._preprocess(frame, bbox)
        output = self.model.run(None, {"input": input_tensor})[0]
        return self._postprocess(output, bbox, frame.shape[:2])

    def estimate_batch(self, frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        """Estimate pose for multiple persons.

        Args:
            frame: Video frame (H, W, C)
            bboxes: List of bounding boxes (x1, y1, x2, y2)

        Returns:
            List of keypoints arrays (17, 3) each
        """
        if self.model is None:
            return [np.random.rand(17, 3).astype(np.float32) for _ in bboxes]

        results = []
        for bbox in bboxes:
            input_tensor = self._preprocess(frame, bbox)
            output = self.model.run(None, {"input": input_tensor})[0]
            results.append(self._postprocess(output, bbox, frame.shape[:2]))
        return results
