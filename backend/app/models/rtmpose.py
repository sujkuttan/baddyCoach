import numpy as np


class RTMPoseEstimator:
    def __init__(self, model_path: str | None = None, device: str = "cpu", chunk_size: int | None = None,
                 simcc_split_ratio: float = 2.0):
        self.device = device
        self.model = None
        self.input_name = None
        self.model_input_size = (192, 256)
        self.simcc_split_ratio = simcc_split_ratio
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
                h, w = self.model.get_inputs()[0].shape[2:]
                self.model_input_size = (w, h)

    @staticmethod
    def bbox_xyxy2cs(bbox: tuple[int, int, int, int], padding: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * padding
        return center, scale

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
        w, h = bbox_scale
        if w > h * aspect_ratio:
            return np.array([w, w / aspect_ratio], dtype=np.float32)
        return np.array([h * aspect_ratio, h], dtype=np.float32)

    @staticmethod
    def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        return np.array([[cs, -sn], [sn, cs]]) @ pt

    @staticmethod
    def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = a - b
        return b + np.array([-direction[1], direction[0]])

    @staticmethod
    def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float,
                        output_size: tuple[int, int],
                        shift: tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        import cv2
        shift = np.array(shift, dtype=np.float32)
        src_w = scale[0]
        dst_w, dst_h = output_size
        rot_rad = np.deg2rad(rot)
        src_dir = RTMPoseEstimator._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0] = center + scale * shift
        src[1] = center + src_dir + scale * shift
        src[2] = RTMPoseEstimator._get_3rd_point(src[0], src[1])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0] = [dst_w * 0.5, dst_h * 0.5]
        dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2] = RTMPoseEstimator._get_3rd_point(dst[0], dst[1])

        if inv:
            return cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return cv2.getAffineTransform(np.float32(src), np.float32(dst))

    def _preprocess(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2
        w, h = self.model_input_size
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))

        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            tensor = np.zeros((1, 3, h, w), dtype=np.float32)
            center = np.array([w, h], dtype=np.float32) * 0.5
            scale = np.array([w, h], dtype=np.float32)
            return tensor, center, scale

        center, scale = self.bbox_xyxy2cs((x1, y1, x2, y2), padding=1.25)
        scale = self._fix_aspect_ratio(scale, aspect_ratio=w / h)

        warp_mat = self.get_warp_matrix(center, scale, 0, output_size=(w, h))
        warped = cv2.warpAffine(frame, warp_mat, (w, h), flags=cv2.INTER_LINEAR)

        rgb = warped[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = rgb.transpose(2, 0, 1)[np.newaxis]
        return tensor, center, scale

    def _decode_simcc(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
        x_locs = np.argmax(simcc_x, axis=1).astype(np.float32)
        y_locs = np.argmax(simcc_y, axis=1).astype(np.float32)
        max_x = np.max(simcc_x, axis=1)
        max_y = np.max(simcc_y, axis=1)
        scores = np.where(max_x > max_y, max_y, max_x)
        scores = 1.0 / (1.0 + np.exp(-scores))
        keypoints = np.column_stack([x_locs, y_locs, scores])
        keypoints[:, :2] /= self.simcc_split_ratio
        return keypoints.astype(np.float32)

    def _decode_heatmap(self, heatmaps: np.ndarray) -> np.ndarray:
        K, Hm, Wm = heatmaps.shape
        flat = heatmaps.reshape(K, -1)
        scores = np.max(flat, axis=1)
        argmax_idx = np.argmax(flat, axis=1).astype(np.float32)
        xs = argmax_idx % Wm
        ys = argmax_idx // Wm
        w, h = self.model_input_size
        xs = xs * (w / Wm)
        ys = ys * (h / Hm)
        keypoints = np.column_stack([xs, ys, scores])
        return keypoints.astype(np.float32)

    def _postprocess(self, outputs: list[np.ndarray], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        output = outputs[0]
        if output.ndim == 4:
            output = output[0]

        n_outputs = len(outputs)

        if n_outputs == 1 and output.shape[-1] <= 4:
            keypoints = output.reshape(-1, 3)
        elif n_outputs == 2:
            simcc_y = outputs[1]
            if simcc_y.ndim == 4:
                simcc_y = simcc_y[0]
            keypoints = self._decode_simcc(output, simcc_y)
        elif n_outputs == 1 and output.ndim == 3 and output.shape[0] >= 17 and output.shape[1] > 4:
            keypoints = self._decode_heatmap(output)
        else:
            return np.zeros((17, 3), dtype=np.float32)

        if keypoints.shape[0] != 17:
            keypoints = keypoints[:17]
        if keypoints.shape[1] < 3:
            keypoints = np.column_stack([keypoints, np.ones(17, dtype=np.float32)])

        w, h = self.model_input_size
        keypoints[:, 0] = keypoints[:, 0] / w * scale[0] + center[0] - scale[0] / 2
        keypoints[:, 1] = keypoints[:, 1] / h * scale[1] + center[1] - scale[1] / 2

        return keypoints.astype(np.float32)

    def estimate(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if self.model is None:
            return np.zeros((17, 3), dtype=np.float32)

        tensor, center, scale = self._preprocess(frame, bbox)
        outputs = self.model.run(None, {self.input_name: tensor})
        return self._postprocess(outputs, center, scale)

    def estimate_batch(self, frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        if self.model is None:
            return [np.zeros((17, 3), dtype=np.float32) for _ in bboxes]

        results = []
        chunk_size = self._chunk_size
        for chunk_start in range(0, len(bboxes), chunk_size):
            chunk = bboxes[chunk_start:chunk_start + chunk_size]
            preprocessed = [self._preprocess(frame, bbox) for bbox in chunk]
            tensors = np.concatenate([p[0] for p in preprocessed], axis=0)
            centers_scales = [(p[1], p[2]) for p in preprocessed]
            outputs = self.model.run(None, {self.input_name: tensors})
            for i, (center, scale) in enumerate(centers_scales):
                per_output = [o[i:i+1] for o in outputs]
                results.append(self._postprocess(per_output, center, scale))
        return results
