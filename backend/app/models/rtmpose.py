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
        outputs: list[np.ndarray],
        bbox: tuple[int, int, int, int],
        frame_shape: tuple
    ) -> np.ndarray:
        """Convert model outputs to keypoints.

        Handles both:
        - simcc format: [simcc_x (1,17,W*2), simcc_y (1,17,H*2)]
        - direct format: (1, 17, 3) keypoints

        Returns:
            Keypoints array (17, 3) with (x, y, confidence)
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if len(outputs) == 2:
            # RTMPose simcc model: argmax to get peak positions
            simcc_x = outputs[0][0]  # (17, W*2)
            simcc_y = outputs[1][0]  # (17, H*2)
            x_coords = np.argmax(simcc_x, axis=1) / 2.0
            y_coords = np.argmax(simcc_y, axis=1) / 2.0
            x_conf = np.max(simcc_x, axis=1)
            y_conf = np.max(simcc_y, axis=1)
            conf = (x_conf + y_conf) / 2.0
            keypoints = np.zeros((17, 3), dtype=np.float32)
            keypoints[:, 0] = x1 + x_coords * (bbox_width / self.input_width)
            keypoints[:, 1] = y1 + y_coords * (bbox_height / self.input_height)
            keypoints[:, 2] = 1.0 / (1.0 + np.exp(-conf))
        else:
            output = outputs[0]
            if len(output.shape) == 3:
                keypoints = output[0]
            elif len(output.shape) == 4:
                keypoints = output[0, 0]
            else:
                keypoints = output.reshape(17, 3)
            keypoints[:, 0] = x1 + keypoints[:, 0] * bbox_width
            keypoints[:, 1] = y1 + keypoints[:, 1] * bbox_height

        return keypoints

    def estimate(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Estimate pose for a single person."""
        if self.model is None:
            return np.random.rand(17, 3).astype(np.float32)

        input_tensor = self._preprocess(frame, bbox)
        outputs = self.model.run(None, {"input": input_tensor})
        return self._postprocess(outputs, bbox, frame.shape[:2])

    def estimate_batch(self, frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        """Estimate pose for multiple persons."""
        if self.model is None:
            return [np.random.rand(17, 3).astype(np.float32) for _ in bboxes]

        results = []
        for bbox in bboxes:
            input_tensor = self._preprocess(frame, bbox)
            outputs = self.model.run(None, {"input": input_tensor})
            results.append(self._postprocess(outputs, bbox, frame.shape[:2]))
        return results
