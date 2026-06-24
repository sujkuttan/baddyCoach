"""TrackNetV3 — published VGG-style architecture for shuttlecock tracking.

Based on the TrackNet paper series:
  TrackNetV1/V2: 3-frame RGB input → VGG encoder-decoder → single heatmap output
  TrackNetV3: adds temporal encoding + InpaintNet for trajectory rectification

This implementation uses the published VGG-style backbone (not the custom UNet
that was previously in this file). Architecture:
  Input:  3 consecutive RGB frames → 9 channels (3×3)
  Encoder: Conv2D-BN-ReLU blocks with MaxPool (64→128→256→512)
  Decoder: TransposedConv with skip connections (512→256→128→64→1)
  Output: Single heatmap for the middle frame

InpaintNet:
  Takes a window of (x, y, conf) detections and uses a small temporal CNN
  to fill gaps and smooth the trajectory.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# TrackNetV3 — VGG-style encoder-decoder backbone
# ═══════════════════════════════════════════════════════════════════════════════

class VGGBlock(nn.Module):
    """Single VGG-style conv block: Conv2D → BN → ReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class TrackNetV3Backbone(nn.Module):
    """Published VGG-style encoder-decoder for shuttlecock tracking.

    Input:  (B, 9, H, W)  — 3 consecutive RGB frames stacked
    Output: (B, 1, H, W)  — heatmap for the middle frame
    """

    def __init__(self, in_channels: int = 9):
        super().__init__()

        # ─── Encoder ──────────────────────────────────────────
        # Stage 1: 9 → 64
        self.enc1 = nn.Sequential(
            VGGBlock(in_channels, 64),
            VGGBlock(64, 64),
        )
        # Stage 2: 64 → 128
        self.enc2 = nn.Sequential(
            VGGBlock(64, 128),
            VGGBlock(128, 128),
        )
        # Stage 3: 128 → 256
        self.enc3 = nn.Sequential(
            VGGBlock(128, 256),
            VGGBlock(256, 256),
            VGGBlock(256, 256),
        )
        # Stage 4: 256 → 512
        self.enc4 = nn.Sequential(
            VGGBlock(256, 512),
            VGGBlock(512, 512),
            VGGBlock(512, 512),
        )
        # Stage 5: 512 → 512 (bottleneck)
        self.enc5 = nn.Sequential(
            VGGBlock(512, 512),
            VGGBlock(512, 512),
            VGGBlock(512, 512),
        )

        # ─── Decoder ──────────────────────────────────────────
        # Decoder uses transposed convolutions for upsampling
        # Each decoder block: UpConv → concat encoder skip → Conv → Conv
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            VGGBlock(256 + 512, 512),  # + skip from enc4
            VGGBlock(512, 256),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            VGGBlock(128 + 256, 256),  # + skip from enc3
            VGGBlock(256, 128),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            VGGBlock(64 + 128, 128),   # + skip from enc2
            VGGBlock(128, 64),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            VGGBlock(32 + 64, 64),     # + skip from enc1
            VGGBlock(64, 32),
        )

        # Output layer: produce single heatmap
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)       # (B, 64, H, W)
        p1 = nn.functional.max_pool2d(e1, 2)  # (B, 64, H/2, W/2)

        e2 = self.enc2(p1)      # (B, 128, H/2, W/2)
        p2 = nn.functional.max_pool2d(e2, 2)  # (B, 128, H/4, W/4)

        e3 = self.enc3(p2)      # (B, 256, H/4, W/4)
        p3 = nn.functional.max_pool2d(e3, 2)  # (B, 256, H/8, W/8)

        e4 = self.enc4(p3)      # (B, 512, H/8, W/8)
        p4 = nn.functional.max_pool2d(e4, 2)  # (B, 512, H/16, W/16)

        e5 = self.enc5(p4)      # (B, 512, H/16, W/16) bottleneck

        # Decoder with skip connections
        d4 = self.dec4[0](e5)   # UpConv: (B, 256, H/8, W/8)
        # Interpolate e4 to match d4 spatial dims (in case of rounding)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4[1](d4)
        d4 = self.dec4[2](d4)   # (B, 256, H/8, W/8)

        d3 = self.dec3[0](d4)   # UpConv: (B, 128, H/4, W/4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3[1](d3)
        d3 = self.dec3[2](d3)   # (B, 128, H/4, W/4)

        d2 = self.dec2[0](d3)   # UpConv: (B, 64, H/2, W/2)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2[1](d2)
        d2 = self.dec2[2](d2)   # (B, 64, H/2, W/2)

        d1 = self.dec1[0](d2)   # UpConv: (B, 32, H, W)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1[1](d1)
        d1 = self.dec1[2](d1)   # (B, 32, H, W)

        return self.out(d1)     # (B, 1, H, W)


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
        # x: (B, 3, T) — [x, y, conf] over time
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return self.out(x)  # (B, 2, T) — refined x, y


# ═══════════════════════════════════════════════════════════════════════════════
# TrackNetV3 wrapper — combines backbone + InpaintNet
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_HEIGHT = 288
INPUT_WIDTH = 512


def _build_3frame_window(preprocessed: list, center_idx: int) -> np.ndarray:
    """Build a 3-frame input window centered on center_idx.

    For boundaries (idx 0 or len-1), edge frames are repeated.
    Returns: (9, H, W) tensor.
    """
    n = len(preprocessed)
    if n < 3:
        raise ValueError("Need at least 3 frames")
    indices = []
    for offset in (-1, 0, 1):
        src = center_idx + offset
        src = max(0, min(src, n - 1))
        indices.append(src)
    window = np.concatenate([preprocessed[i] for i in indices], axis=-1)  # (H, W, 9)
    return window.transpose(2, 0, 1)  # (9, H, W)


def _extract_peak(heatmap: np.ndarray, orig_w: int, orig_h: int) -> tuple[float, float, float]:
    """Extract argmax position and confidence from a heatmap."""
    hm = 1.0 / (1.0 + np.exp(-heatmap))
    y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
    conf = float(hm.max())
    x = float(x_idx * orig_w / INPUT_WIDTH)
    y = float(y_idx * orig_h / INPUT_HEIGHT)
    return x, y, conf


class TrackNetV3:
    """Published TrackNetV3 — VGG backbone + optional InpaintNet.

    Interface matches the original custom UNet wrapper so callers
    (pipeline/shuttle.py) work without changes.
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 inpaintnet_path: str | None = None):
        self.device = device
        self.model: TrackNetV3Backbone | None = None
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

            # Detect input channels from state_dict to support both
            # 9-channel (3 frames) and 27-channel (old custom) weights
            in_channels = 9
            for k, v in state_dict.items():
                if 'enc1.0.conv.weight' in k or 'down_block_1' in k:
                    in_channels = v.shape[1]
                    break

            if in_channels == 27:
                status = {"loaded": False, "missing_frac": 1.0, "n_missing": 0,
                          "n_unexpected": 0, "core_missing": ["27-channel UNet weights incompatible"]}
                record_model_health("tracknet", status)
                print("WARNING: Detected 27-channel weights (old custom UNet format). "
                      "These are incompatible with the published VGG-style backbone. "
                      "The published TrackNetV3 uses 9 input channels (3 RGB frames).")
                self.model = None
                return

            self.model = TrackNetV3Backbone(in_channels=in_channels)
            status = _checked_load(self.model, state_dict,
                                   core_prefixes=("enc1", "enc5", "out"))
            record_model_health("tracknet", status)

            if not status["loaded"]:
                print(f"WARNING: TrackNetV3 core layers missing ({status['core_missing']}). "
                      "Model set to None — honest fallback.")
                self.model = None
                return

            self.model.to(self.device)
            self.model.eval()
            model_type = "TrackNetV3Backbone (VGG-style)"
            print(f"TrackNetV3 loaded successfully: {model_type} from {path}")
        except Exception as e:
            print(f"TrackNetV3 load error: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            status = {"loaded": False, "missing_frac": 1.0, "n_missing": 0,
                      "n_unexpected": 0, "core_missing": [str(e)]}
            record_model_health("tracknet", status)
            print("WARNING: TrackNetV3 expects the published VGG-style backbone "
                  "(9 input channels, 3 RGB frames), NOT the old custom UNet.")

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
            print(f"InpaintNet loaded successfully from {path}")
        except Exception as e:
            print(f"InpaintNet load error: {e}, proceeding without trajectory rectification")
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

        if len(frames) < 3:
            raise RuntimeError("TrackNetV3 requires at least 3 frames")

        if batch_size is None:
            from app.config.gpu_batch import get_gpu_batch_config
            batch_size = get_gpu_batch_config(self.device)["tracknet_chunk"]

        orig_w, orig_h = original_size if original_size else (frames[0].shape[1], frames[0].shape[0])

        # Preprocess all frames
        preprocessed = self._preprocess(frames)
        n_frames = len(preprocessed)

        # Build 3-frame windows and run inference
        all_raw = [None] * n_frames

        for chunk_start in range(0, n_frames, batch_size):
            chunk_end = min(chunk_start + batch_size, n_frames)
            batch_windows = []
            batch_indices = []

            for i in range(chunk_start, chunk_end):
                window = _build_3frame_window(preprocessed, i)
                batch_windows.append(window)
                batch_indices.append(i)

            batch_tensor = torch.from_numpy(np.stack(batch_windows)).float().to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                heatmaps = outputs.cpu().numpy()  # (B, 1, H, W)

            for j, local_idx in enumerate(batch_indices):
                hm = heatmaps[j, 0]  # (H, W)
                x, y, conf = _extract_peak(hm, orig_w, orig_h)
                all_raw[local_idx] = (x, y, conf)

            del batch_tensor
            if self.device != "cpu":
                torch.cuda.empty_cache()

        # Run InpaintNet trajectory rectification if model is available
        if self.inpaintnet is not None:
            all_raw = self._rectify_trajectory(all_raw, orig_w, orig_h)

        # Convert to output format
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

        # Step 1: Fill gaps via linear interpolation (pre-InpaintNet cleanup)
        xs = np.array([d[0] if d is not None else np.nan for d in raw_detections], dtype=np.float32)
        ys = np.array([d[1] if d is not None else np.nan for d in raw_detections], dtype=np.float32)
        confs = np.array([d[2] if d is not None else 0.0 for d in raw_detections], dtype=np.float32)

        # Linear interpolation for gaps
        mask = ~np.isnan(xs)
        if mask.sum() >= 2:
            indices = np.arange(n)
            xs = np.interp(indices, indices[mask], xs[mask])
            ys = np.interp(indices, indices[mask], ys[mask])
            # Fill confidence for interpolated frames
            confs = np.interp(indices, indices[mask], confs[mask])
        elif mask.sum() == 1:
            xs[:] = xs[mask][0]
            ys[:] = ys[mask][0]
            confs[:] = confs[mask][0] * 0.5
        else:
            return raw_detections

        # Step 2: Apply InpaintNet if available
        if self.inpaintnet is not None:
            window = self.inpaintnet.window_size
            if n >= window:
                with torch.no_grad():
                    # Build input: (x, y, conf) over sliding windows
                    inpaint_input = np.stack([xs, ys, confs], axis=0)[np.newaxis, :, :]  # (1, 3, N)
                    inpaint_tensor = torch.from_numpy(inpaint_input).float().to(self.device)
                    refined = self.inpaintnet(inpaint_tensor).cpu().numpy()[0]  # (2, N)
                    # Blend: weighted average of raw and InpaintNet output
                    xs = 0.7 * xs + 0.3 * refined[0]
                    ys = 0.7 * ys + 0.3 * refined[1]

        # Step 3: Temporal smoothing (Savitzky-Golay-like via simple moving average)
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

        # Clamp to valid range
        xs = np.clip(xs, 0, orig_w)
        ys = np.clip(ys, 0, orig_h)
        confs = np.clip(confs, 0, 1)

        return [(float(xs[i]), float(ys[i]), float(confs[i])) for i in range(n)]
