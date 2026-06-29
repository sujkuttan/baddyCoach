"""TrackNetV3 — custom UNet architecture for shuttlecock tracking.

Architecture matches the checkpoint trained by the original authors:
  Input:  9 consecutive RGB frames stacked → 27 channels (9×3)
  Encoder: Conv2D-BN-ReLU blocks with MaxPool (27→64→128→256→512)
  Decoder: Interpolate + skip concat + Conv2D (512→256→128→64→8)
  Output: 8 heatmap channels (first channel used for peak extraction)

InpaintNet (trajectory rectification):
  Takes a window of (x, y, conf) detections and uses a small temporal CNN
  to fill gaps and smooth the trajectory.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# TrackNetV3 — custom UNet encoder-decoder backbone
# ═══════════════════════════════════════════════════════════════════════════════

class SingleConv(nn.Module):
    """Single conv block: Conv2D → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class TrackNetV3Model(nn.Module):
    """Custom UNet matching the original TrackNetV3 checkpoint.

    Input:  (B, 27, H, W)  — 9 consecutive RGB frames stacked
    Output: (B, 8, H, W)   — 8 heatmap channels (first used for detection)
    """

    def __init__(self):
        super().__init__()
        self.down_block_1 = nn.ModuleDict({
            'conv_1': SingleConv(27, 64),
            'conv_2': SingleConv(64, 64),
        })
        self.down_block_2 = nn.ModuleDict({
            'conv_1': SingleConv(64, 128),
            'conv_2': SingleConv(128, 128),
        })
        self.down_block_3 = nn.ModuleDict({
            'conv_1': SingleConv(128, 256),
            'conv_2': SingleConv(256, 256),
            'conv_3': SingleConv(256, 256),
        })
        self.bottleneck = nn.ModuleDict({
            'conv_1': SingleConv(256, 512),
            'conv_2': SingleConv(512, 512),
            'conv_3': SingleConv(512, 512),
        })
        self.up_block_1 = nn.ModuleDict({
            'conv_1': SingleConv(768, 256),
            'conv_2': SingleConv(256, 256),
            'conv_3': SingleConv(256, 256),
        })
        self.up_block_2 = nn.ModuleDict({
            'conv_1': SingleConv(384, 128),
            'conv_2': SingleConv(128, 128),
        })
        self.up_block_3 = nn.ModuleDict({
            'conv_1': SingleConv(192, 64),
            'conv_2': SingleConv(64, 64),
        })
        self.predictor = nn.Conv2d(64, 8, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down_block_1['conv_2'](self.down_block_1['conv_1'](x))
        d1_pool = nn.functional.max_pool2d(d1, 2)

        d2 = self.down_block_2['conv_2'](self.down_block_2['conv_1'](d1_pool))
        d2_pool = nn.functional.max_pool2d(d2, 2)

        d3 = self.down_block_3['conv_3'](
            self.down_block_3['conv_2'](self.down_block_3['conv_1'](d2_pool))
        )
        d3_pool = nn.functional.max_pool2d(d3, 2)

        b = self.bottleneck['conv_3'](
            self.bottleneck['conv_2'](self.bottleneck['conv_1'](d3_pool))
        )

        # Decoder with skip connections
        b_up = nn.functional.interpolate(b, size=d3.shape[2:], mode='bilinear', align_corners=True)
        u1 = self.up_block_1['conv_3'](
            self.up_block_1['conv_2'](self.up_block_1['conv_1'](torch.cat([b_up, d3], dim=1)))
        )

        u1_up = nn.functional.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=True)
        u2 = self.up_block_2['conv_2'](
            self.up_block_2['conv_1'](torch.cat([u1_up, d2], dim=1))
        )

        u2_up = nn.functional.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        u3 = self.up_block_3['conv_2'](
            self.up_block_3['conv_1'](torch.cat([u2_up, d1], dim=1))
        )

        return self.predictor(u3)


# ═══════════════════════════════════════════════════════════════════════════════
# InpaintNet — trajectory rectification and gap-filling
# ═══════════════════════════════════════════════════════════════════════════════

class InpaintNet(nn.Module):
    """Temporal trajectory rectification network.

    Takes a sliding window of (x, y, confidence) detections and applies
    a small temporal CNN to:
      1. Fill gaps (frames where no shuttle was detected)
      2. Smooth noisy detections
      3. Reject outliers (temporal consistency)

    Input:  (B, 3, T)  — x, y, confidence over a T-frame window
    Output: (B, 2, T)  — refined x, y positions
    """

    def __init__(self, window_size: int = 15, hidden_dim: int = 64):
        super().__init__()
        self.window_size = window_size

        self.conv1 = nn.Conv1d(3, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.out = nn.Conv1d(hidden_dim, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return self.out(x)


# ═══════════════════════════════════════════════════════════════════════════════
# TrackNetV3 wrapper — combines backbone + InpaintNet
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_HEIGHT = 288
INPUT_WIDTH = 512


def _build_9frame_window(preprocessed: list, center_idx: int) -> np.ndarray:
    """Build a 9-frame input window centered on center_idx.

    For boundaries (idx < 8), edge frames are repeated (pad).
    Returns: (27, H, W) tensor.
    """
    n = len(preprocessed)
    if n < 1:
        raise ValueError("Need at least 1 frame")
    start = max(0, center_idx - 8)
    window = preprocessed[start:center_idx + 1]
    pad_len = 9 - len(window)
    if pad_len > 0:
        window = [window[0]] * pad_len + window
    window = np.concatenate(window[-9:], axis=-1)  # (H, W, 27)
    return window.transpose(2, 0, 1)  # (27, H, W)


def _extract_peak(heatmap: np.ndarray, orig_w: int, orig_h: int) -> tuple[float, float, float]:
    """Extract argmax position and confidence from a heatmap."""
    hm = 1.0 / (1.0 + np.exp(-heatmap))
    y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
    conf = float(hm.max())
    x = float(x_idx * orig_w / INPUT_WIDTH)
    y = float(y_idx * orig_h / INPUT_HEIGHT)
    return x, y, conf


class TrackNetV3:
    """TrackNetV3 — custom UNet backbone + optional InpaintNet.

    Interface designed to be called by pipeline/shuttle.py.
    Built-in fallback to linear interpolation when InpaintNet is unavailable.
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 inpaintnet_path: str | None = None):
        self.device = device
        self.model: TrackNetV3Model | None = None
        self.inpaintnet: InpaintNet | None = None
        self.input_height = INPUT_HEIGHT
        self.input_width = INPUT_WIDTH

        if model_path and Path(model_path).exists():
            self._load_backbone(model_path)
        else:
            print(f"TrackNetV3 model file not found: {model_path}")

        if inpaintnet_path and Path(inpaintnet_path).exists():
            self._load_inpaintnet(inpaintnet_path)

    def _load_backbone(self, path: str):
        from app.pipeline.shared.models import _checked_load, record_model_health
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint if isinstance(checkpoint, dict) else {}
            if 'model' in state_dict:
                state_dict = state_dict['model']

            self.model = TrackNetV3Model()
            core_prefixes = ("down_block_1", "bottleneck", "up_block_1", "predictor")
            status = _checked_load(self.model, state_dict, core_prefixes=core_prefixes)
            record_model_health("tracknet", status)

            if not status["loaded"]:
                print(f"WARNING: TrackNetV3 core layers missing ({status['core_missing']}). "
                      "Model set to None.")
                self.model = None
                return

            self.model.to(self.device)
            self.model.eval()
            print(f"TrackNetV3 loaded successfully (custom UNet, 27→8) from {path}")
        except Exception as e:
            print(f"TrackNetV3 load error: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            status = {"loaded": False, "missing_frac": 1.0, "n_missing": 0,
                      "n_unexpected": 0, "core_missing": [str(e)]}
            record_model_health("tracknet", status)

    def _load_inpaintnet(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint if isinstance(checkpoint, dict) else {}
            if 'model' in state_dict:
                state_dict = state_dict['model']

            self.inpaintnet = InpaintNet()
            self.inpaintnet.load_state_dict(state_dict, strict=False)
            self.inpaintnet.to(self.device)
            self.inpaintnet.eval()
            from app.pipeline.shared.logging import logger
            logger.info(f"InpaintNet loaded successfully from {path}")
        except Exception as e:
            from app.pipeline.shared.logging import logger
            logger.warning(f"InpaintNet load error: {e}, proceeding without trajectory rectification")
            self.inpaintnet = None

    def _preprocess(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Preprocess each frame: resize → RGB → normalize.

        Returns list of (H, W, 3) float32 arrays in [0, 1].
        """
        import cv2
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed.append(normalized)
        return processed

    def predict_batch(self, frames: list[np.ndarray], batch_size: int | None = None,
                      original_size: tuple | None = None) -> list[dict]:
        """Run TrackNetV3 + InpaintNet on a batch of frames.

        Args:
            frames: List of video frames (H, W, 3) BGR.
            batch_size: Max frames per inference batch.
            original_size: (width, height) of original video for coordinate scaling.

        Returns:
            List of dicts with keys 'x', 'y', 'confidence' per frame.
        """
        if self.model is None:
            raise RuntimeError("TrackNetV3 backbone not loaded")

        if len(frames) < 1:
            raise RuntimeError("Need at least 1 frame")

        if batch_size is None:
            from app.config.gpu_batch import get_gpu_batch_config
            batch_size = get_gpu_batch_config(self.device)["tracknet_chunk"]

        orig_w, orig_h = original_size if original_size else (frames[0].shape[1], frames[0].shape[0])

        preprocessed = self._preprocess(frames)
        n_frames = len(preprocessed)

        all_raw = [None] * n_frames

        for chunk_start in range(0, n_frames, batch_size):
            chunk_end = min(chunk_start + batch_size, n_frames)
            batch_windows = []
            batch_indices = []

            for i in range(chunk_start, chunk_end):
                window = _build_9frame_window(preprocessed, i)
                batch_windows.append(window)
                batch_indices.append(i)

            batch_tensor = torch.from_numpy(np.stack(batch_windows)).float().to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                heatmaps = outputs.cpu().numpy()[:, 0]  # (B, H, W) — first of 8 channels

            for j, local_idx in enumerate(batch_indices):
                hm = heatmaps[j]  # (H, W)
                x, y, conf = _extract_peak(hm, orig_w, orig_h)
                all_raw[local_idx] = (x, y, conf)

            del batch_tensor
            if self.device != "cpu":
                torch.cuda.empty_cache()

        if self.inpaintnet is not None:
            all_raw = self._rectify_trajectory(all_raw, orig_w, orig_h)

        results = []
        for item in all_raw:
            if item is None:
                results.append({"x": 0.0, "y": 0.0, "confidence": 0.0})
            else:
                results.append({"x": item[0], "y": item[1], "confidence": item[2]})

        return results

    def predict(self, frames: list[np.ndarray], original_size: tuple | None = None) -> list[dict]:
        """Single-frame-group prediction (legacy interface)."""
        return self.predict_batch(frames, batch_size=1, original_size=original_size)

    def _rectify_trajectory(self, raw_detections: list[tuple | None],
                            orig_w: int, orig_h: int) -> list[tuple]:
        """Apply InpaintNet to smooth and gap-fill the trajectory.

        Falls back to linear interpolation when InpaintNet is unavailable.
        """
        n = len(raw_detections)

        xs = np.array([d[0] if d is not None else np.nan for d in raw_detections], dtype=np.float32)
        ys = np.array([d[1] if d is not None else np.nan for d in raw_detections], dtype=np.float32)
        confs = np.array([d[2] if d is not None else 0.0 for d in raw_detections], dtype=np.float32)

        mask = ~np.isnan(xs)
        if mask.sum() >= 2:
            indices = np.arange(n)
            xs = np.interp(indices, indices[mask], xs[mask])
            ys = np.interp(indices, indices[mask], ys[mask])
            confs = np.interp(indices, indices[mask], confs[mask])
        elif mask.sum() == 1:
            xs[:] = xs[mask][0]
            ys[:] = ys[mask][0]
            confs[:] = confs[mask][0] * 0.5
        else:
            return raw_detections

        if self.inpaintnet is not None:
            window = self.inpaintnet.window_size
            if n >= window:
                with torch.no_grad():
                    inpaint_input = np.stack([xs, ys, confs], axis=0)[np.newaxis, :, :]
                    inpaint_tensor = torch.from_numpy(inpaint_input).float().to(self.device)
                    refined = self.inpaintnet(inpaint_tensor).cpu().numpy()[0]
                    xs = 0.7 * xs + 0.3 * refined[0]
                    ys = 0.7 * ys + 0.3 * refined[1]

        window_smooth = 3
        if n >= window_smooth:
            kernel = np.ones(window_smooth) / window_smooth
            xs = np.concatenate([
                xs[:window_smooth // 2],
                np.convolve(xs, kernel, mode='valid'),
                xs[-(window_smooth // 2):],
            ])[:n]
            ys = np.concatenate([
                ys[:window_smooth // 2],
                np.convolve(ys, kernel, mode='valid'),
                ys[-(window_smooth // 2):],
            ])[:n]

        xs = np.clip(xs, 0, orig_w)
        ys = np.clip(ys, 0, orig_h)
        confs = np.clip(confs, 0, 1)

        return [(float(xs[i]), float(ys[i]), float(confs[i])) for i in range(n)]
