import numpy as np


class RTMPoseEstimator:
    def __init__(self, model_path: str | None = None, device: str = "cpu", chunk_size: int | None = None):
        self.device = device
        self.model = None
        self.input_name = None
        if chunk_size is not None:
            self._chunk_size = chunk_size
        else:
            from app.config.gpu_batch import get_gpu_batch_config
            self._chunk_size = get_gpu_batch_config(device)["rtmpose_chunk"]
        if model_path:
            import os
            if os.path.exists(model_path):
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device.lower() else ["CPUExecutionProvider"]
                self.model = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.model.get_inputs()[0].name

    @staticmethod
    def _preprocess(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((1, 3, 256, 192), dtype=np.float32)

        import cv2
        resized = cv2.resize(crop, (192, 256))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        return rgb.transpose(2, 0, 1)[np.newaxis]

    def _postprocess(self, outputs, bbox, frame_shape):
        output = outputs[0]
        if output.ndim == 4:
            output = output[0]

        n_outputs = len(outputs)
        if n_outputs == 1:
            keypoints = output.reshape(17, 3)
        else:
            output2 = outputs[1]
            if output2.ndim == 4:
                output2 = output2[0]
            keypoints = np.concatenate([output, output2], axis=-1).reshape(17, -1)[:, :3]

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            return np.zeros((17, 3), dtype=np.float32)

        keypoints[:, 0] = keypoints[:, 0] / 192.0 * bw + x1
        keypoints[:, 1] = keypoints[:, 1] / 256.0 * bh + y1

        if keypoints.shape[1] > 2:
            keypoints[:, 2] = 1.0 / (1.0 + np.exp(-keypoints[:, 2]))

        return keypoints.astype(np.float32)

    def estimate(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if self.model is None:
            return np.zeros((17, 3), dtype=np.float32)

        input_tensor = self._preprocess(frame, bbox)
        outputs = self.model.run(None, {self.input_name: input_tensor})
        return self._postprocess(outputs, bbox, frame.shape[:2])

    def estimate_batch(self, frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        if self.model is None:
            return [np.zeros((17, 3), dtype=np.float32) for _ in bboxes]

        results = []
        chunk_size = self._chunk_size
        for chunk_start in range(0, len(bboxes), chunk_size):
            chunk = bboxes[chunk_start:chunk_start + chunk_size]
            batch_tensors = np.concatenate([self._preprocess(frame, bbox) for bbox in chunk], axis=0)
            outputs = self.model.run(None, {self.input_name: batch_tensors})
            for i, bbox in enumerate(chunk):
                per_output = [o[i:i+1] for o in outputs]
                results.append(self._postprocess(per_output, bbox, frame.shape[:2]))
        return results
