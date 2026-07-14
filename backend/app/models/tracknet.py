"""TrackNetV3 — custom UNet architecture for shuttlecock tracking.

Architecture matches the checkpoint trained by the original authors:
  Input:  static RGB background plus 8 RGB frames stacked → 27 channels
  Encoder: Conv2D-BN-ReLU blocks with MaxPool (27→64→128→256→512)
  Decoder: Interpolate + skip concat + Conv2D (512→256→128→64→8)
  Output: 8 heatmap channels (first channel used for peak extraction)

InpaintNet (trajectory rectification):
  Takes a window of (x, y, conf) detections and uses a small temporal CNN
  to fill gaps and smooth the trajectory.
"""

import io
import zipfile

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

class _Conv1DBlock(nn.Module):
    """Checkpoint-compatible Conv1d + LeakyReLU block."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding="same", bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class _Double1DConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = _Conv1DBlock(in_dim, out_dim)
        self.conv_2 = _Conv1DBlock(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(x))


class InpaintNet(nn.Module):
    """Official TrackNetV3 trajectory repair architecture.

    ``coords`` has shape ``(batch, sequence, 2)`` in normalized image space;
    ``mask`` has shape ``(batch, sequence, 1)`` and marks original misses.
    """

    def __init__(self):
        super().__init__()
        self.down_1 = _Conv1DBlock(3, 32)
        self.down_2 = _Conv1DBlock(32, 64)
        self.down_3 = _Conv1DBlock(64, 128)
        # Keep the published checkpoint's historical spelling.
        self.buttelneck = _Double1DConv(128, 256)
        self.up_1 = _Conv1DBlock(384, 128)
        self.up_2 = _Conv1DBlock(192, 64)
        self.up_3 = _Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([coords, mask], dim=2).permute(0, 2, 1)
        x1 = self.down_1(x)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x = self.buttelneck(x3)
        x = self.up_1(torch.cat([x, x3], dim=1))
        x = self.up_2(torch.cat([x, x2], dim=1))
        x = self.up_3(torch.cat([x, x1], dim=1))
        return self.sigmoid(self.predictor(x)).permute(0, 2, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# TrackNetV3 wrapper — combines backbone + InpaintNet
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_HEIGHT = 288
INPUT_WIDTH = 512
TRACKNET_SEQUENCE_LENGTH = 8
TRACKNET_BACKGROUND_MODE = "concat"


def _court_crop_rect(
    corners: list[tuple[int, int]] | list[list[int]],
    margins: dict[str, float],
    aspect: float = INPUT_WIDTH / INPUT_HEIGHT,
) -> tuple[float, float, float, float]:
    """Expand the detected court box and grow it to the TrackNet aspect ratio."""
    pts = np.asarray(corners, dtype=np.float64)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    court_w = max(x1 - x0, 1.0)
    court_h = max(y1 - y0, 1.0)

    x0 -= float(margins.get("left", 0.0)) * court_w
    x1 += float(margins.get("right", 0.0)) * court_w
    y0 -= float(margins.get("top", 0.0)) * court_h
    y1 += float(margins.get("bottom", 0.0)) * court_h

    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    if width / height < aspect:
        target_width = aspect * height
        center_x = (x0 + x1) * 0.5
        x0 = center_x - target_width * 0.5
        x1 = center_x + target_width * 0.5
    else:
        target_height = width / aspect
        center_y = (y0 + y1) * 0.5
        y0 = center_y - target_height * 0.5
        y1 = center_y + target_height * 0.5

    return x0, y0, x1, y1


def _clamp_crop_rect(
    crop_rect: tuple[float, float, float, float] | None,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Clamp a crop rectangle to a concrete frame and ensure non-zero size."""
    if crop_rect is None:
        return 0, 0, int(frame_width), int(frame_height)

    x0, y0, x1, y1 = crop_rect
    x0_i = int(np.floor(max(0.0, min(float(x0), max(frame_width - 1, 0)))))
    y0_i = int(np.floor(max(0.0, min(float(y0), max(frame_height - 1, 0)))))
    x1_i = int(np.ceil(max(float(x1), x0_i + 1.0)))
    y1_i = int(np.ceil(max(float(y1), y0_i + 1.0)))
    x1_i = min(x1_i, int(frame_width))
    y1_i = min(y1_i, int(frame_height))
    if x1_i <= x0_i:
        x1_i = min(int(frame_width), x0_i + 1)
    if y1_i <= y0_i:
        y1_i = min(int(frame_height), y0_i + 1)
    return x0_i, y0_i, x1_i, y1_i


def _map_detection_to_full_frame(
    x: float,
    y: float,
    crop_rect: tuple[float, float, float, float],
    *,
    input_width: int = INPUT_WIDTH,
    input_height: int = INPUT_HEIGHT,
) -> tuple[float, float]:
    """Map a TrackNet-space point back into full-frame image coordinates."""
    x0, y0, x1, y1 = crop_rect
    crop_w = max(float(x1 - x0), 1.0)
    crop_h = max(float(y1 - y0), 1.0)
    full_x = float(x0 + (float(x) / float(input_width)) * crop_w)
    full_y = float(y0 + (float(y) / float(input_height)) * crop_h)
    return full_x, full_y


def _build_input(frames: list[np.ndarray], background: np.ndarray) -> np.ndarray:
    """Build checkpoint input: background RGB followed by eight RGB frames."""
    if len(frames) != TRACKNET_SEQUENCE_LENGTH:
        raise ValueError(f"Expected {TRACKNET_SEQUENCE_LENGTH} frames, got {len(frames)}")
    stacked = np.concatenate([background, *frames], axis=-1)
    return stacked.transpose(2, 0, 1)


def _build_8frame_window(preprocessed: list[np.ndarray], start_idx: int) -> list[np.ndarray]:
    """Return eight boundary-padded frames beginning at ``start_idx``."""
    if not preprocessed:
        raise ValueError("Need at least 1 frame")
    indices = np.clip(np.arange(start_idx, start_idx + TRACKNET_SEQUENCE_LENGTH),
                      0, len(preprocessed) - 1)
    return [preprocessed[index] for index in indices]


def _build_9frame_window(preprocessed: list, center_idx: int) -> np.ndarray:
    """Legacy compatibility wrapper for callers that still construct 9-frame inputs.

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


def _triangular_weights(sequence_length: int) -> np.ndarray:
    """Symmetric temporal weights that favor the central TrackNet outputs."""
    return 1.0 + np.minimum(np.arange(sequence_length),
                            (sequence_length - 1) - np.arange(sequence_length))


def _extract_largest_component(probabilities: np.ndarray, orig_w: int, orig_h: int,
                               threshold: float) -> tuple[float, float, float]:
    """Decode the largest thresholded shuttle blob and scale it to source pixels."""
    candidates = _extract_component_candidates(
        probabilities, orig_w=orig_w, orig_h=orig_h, threshold=threshold, max_components=1
    )
    if not candidates:
        return 0.0, 0.0, 0.0
    x, y, confidence, _area = candidates[0]
    return x, y, confidence


def _extract_component_candidates(
    probabilities: np.ndarray,
    orig_w: int,
    orig_h: int,
    threshold: float,
    max_components: int,
    crop_rect: tuple[float, float, float, float] | None = None,
) -> list[tuple[float, float, float, int]]:
    """Return top thresholded shuttle blobs as (x, y, confidence, area)."""
    import cv2

    binary = (probabilities >= threshold).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return []

    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    candidates: list[tuple[float, float, float, int]] = []
    for offset in order[:max_components]:
        component = 1 + int(offset)
        component_mask = labels == component
        x_center, y_center = centroids[component]
        confidence = float(probabilities[component_mask].max())
        area = int(stats[component, cv2.CC_STAT_AREA])
        if crop_rect is None:
            mapped_x = float(x_center * orig_w / INPUT_WIDTH)
            mapped_y = float(y_center * orig_h / INPUT_HEIGHT)
        else:
            mapped_x, mapped_y = _map_detection_to_full_frame(x_center, y_center, crop_rect)
        candidates.append((
            mapped_x,
            mapped_y,
            confidence,
            area,
        ))
    return candidates


def _select_detection_candidate(
    candidates: list[tuple[float, float, float, int]],
    prev_accepted: tuple[float, float, float] | None,
    prev_prev_accepted: tuple[float, float, float] | None,
    *,
    motion_weight: float,
    confidence_weight: float,
    distance_scale_px: float,
) -> tuple[float, float, float, int] | None:
    """Choose the candidate most consistent with recent shuttle motion."""
    if not candidates:
        return None
    if prev_accepted is None:
        return candidates[0]

    if prev_prev_accepted is None:
        pred_x, pred_y = float(prev_accepted[0]), float(prev_accepted[1])
    else:
        pred_x = float(prev_accepted[0] + (prev_accepted[0] - prev_prev_accepted[0]))
        pred_y = float(prev_accepted[1] + (prev_accepted[1] - prev_prev_accepted[1]))

    scale = max(float(distance_scale_px), 1.0)
    best = None
    best_score = -np.inf
    for candidate in candidates:
        x, y, conf, _area = candidate
        dist = float(np.hypot(x - pred_x, y - pred_y))
        motion_score = 1.0 / (1.0 + dist / scale)
        score = float(motion_weight) * motion_score + float(confidence_weight) * float(conf)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _aggregate_overlapping_heatmaps(window_outputs: list[tuple[int, np.ndarray]], n_frames: int,
                                    sequence_length: int = TRACKNET_SEQUENCE_LENGTH) -> np.ndarray:
    """Reference aggregation helper used by tests and small offline callers.

    Production inference streams this same calculation to keep long-video memory bounded.
    """
    if not window_outputs:
        return np.empty((0,), dtype=np.float32)
    height, width = window_outputs[0][1].shape[-2:]
    weighted = np.zeros((n_frames, height, width), dtype=np.float32)
    weights = np.zeros(n_frames, dtype=np.float32)
    temporal_weights = _triangular_weights(sequence_length)
    for start, heatmaps in window_outputs:
        for offset, heatmap in enumerate(heatmaps):
            frame = start + offset
            if 0 <= frame < n_frames:
                weighted[frame] += temporal_weights[offset] * heatmap
                weights[frame] += temporal_weights[offset]
    valid = weights > 0
    weighted[valid] /= weights[valid, None, None]
    return weighted


def _extract_peak(heatmap: np.ndarray, orig_w: int, orig_h: int) -> tuple[float, float, float]:
    """Extract argmax position and confidence from a heatmap."""
    hm = 1.0 / (1.0 + np.exp(-heatmap))
    y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
    conf = float(hm.max())
    x = float(x_idx * orig_w / INPUT_WIDTH)
    y = float(y_idx * orig_h / INPUT_HEIGHT)
    return x, y, conf


def _accept_detection_candidate(
    candidate: tuple[float, float, float, int] | tuple[float, float, float] | None,
    prev_accepted: tuple[float, float, float] | None,
    prev_prev_accepted: tuple[float, float, float] | None,
    *,
    min_conf: float,
    trust_min_conf: float,
    low_conf_max_jump_px: float,
    distance_scale_px: float,
) -> bool:
    """Decide whether a decoded shuttle point is trustworthy enough to keep.

    Low-confidence detections are only accepted when they remain temporally
    consistent with the previous accepted point. This keeps weak blob flips
    from entering the track as observed shuttle positions.
    """
    if candidate is None:
        return False
    x, y, conf = candidate[:3]
    if conf < min_conf:
        return False
    if prev_accepted is None or conf >= trust_min_conf:
        return True
    direct_dx = float(x - prev_accepted[0])
    direct_dy = float(y - prev_accepted[1])
    direct_jump = float(np.hypot(direct_dx, direct_dy))
    if direct_jump <= float(low_conf_max_jump_px):
        return True

    if prev_prev_accepted is None:
        pred_x, pred_y = float(prev_accepted[0]), float(prev_accepted[1])
    else:
        pred_x = float(prev_accepted[0] + (prev_accepted[0] - prev_prev_accepted[0]))
        pred_y = float(prev_accepted[1] + (prev_accepted[1] - prev_prev_accepted[1]))
    motion_dist = float(np.hypot(x - pred_x, y - pred_y))
    return motion_dist <= max(float(distance_scale_px), 1.0)


def _gate_tracknet_spikes(
    shuttle_points: np.ndarray,
    *,
    max_step_px: float,
) -> tuple[np.ndarray, int]:
    """Null image-space out-and-back teleports before trajectory rectification."""
    src = np.asarray(shuttle_points, dtype=np.float64)
    out = src.copy()
    valid_indices = [index for index in range(len(src)) if np.isfinite(src[index]).all()]
    to_remove: list[int] = []
    for offset in range(1, len(valid_indices) - 1):
        index = valid_indices[offset]
        prev_point = src[valid_indices[offset - 1]]
        next_point = src[valid_indices[offset + 1]]
        dist_prev = float(np.linalg.norm(src[index] - prev_point))
        dist_next = float(np.linalg.norm(src[index] - next_point))
        if dist_prev > max_step_px and dist_next > max_step_px:
            to_remove.append(index)
    for index in to_remove:
        out[index] = np.nan
    return out, len(to_remove)


def _merge_far_tile_tracks(
    primary_points: np.ndarray,
    far_points: np.ndarray,
    *,
    net_y: float,
) -> tuple[np.ndarray, int]:
    """Use far-court detections only to fill missing primary detections above the net."""
    primary = np.asarray(primary_points, dtype=np.float64)
    far = np.asarray(far_points, dtype=np.float64)
    if primary.shape != far.shape:
        return primary.copy(), 0
    out = primary.copy()
    primary_missing = np.isnan(primary).any(axis=1)
    far_valid = np.isfinite(far).all(axis=1)
    far_half = far[:, 1] < float(net_y)
    fill_mask = primary_missing & far_valid & far_half
    out[fill_mask] = far[fill_mask]
    return out, int(fill_mask.sum())


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
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            except RuntimeError:
                # The distributed checkpoint is a ZIP bundle containing the real
                # checkpoint under ckpts/TrackNet_best.pt.
                with zipfile.ZipFile(path) as archive:
                    member = next(name for name in archive.namelist()
                                  if name.endswith("TrackNet_best.pt"))
                    checkpoint = torch.load(io.BytesIO(archive.read(member)),
                                            map_location=self.device, weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError("checkpoint does not contain metadata and weights")
            param_dict = checkpoint.get("param_dict", {})
            if (param_dict.get("seq_len") != TRACKNET_SEQUENCE_LENGTH or
                    param_dict.get("bg_mode") != TRACKNET_BACKGROUND_MODE):
                raise ValueError("TrackNet checkpoint contract must be seq_len=8, bg_mode='concat'")
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
            print(f"TrackNetV3 loaded successfully (background + 8 frames, 27→8) from {path}")
        except Exception as e:
            print(f"TrackNetV3 load error: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            status = {"loaded": False, "missing_frac": 1.0, "n_missing": 0,
                      "n_unexpected": 0, "core_missing": [str(e)]}
            record_model_health("tracknet", status)

    def _load_inpaintnet(self, path: str):
        from app.pipeline.shared.logging import logger
        from app.pipeline.shared.models import record_model_health
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint
            while isinstance(state_dict, dict):
                nested = next((state_dict[key] for key in ("model_state_dict", "model", "state_dict")
                               if key in state_dict), None)
                if nested is None:
                    break
                state_dict = nested
            if not isinstance(state_dict, dict):
                raise ValueError("checkpoint does not contain a state_dict")

            state_dict = {
                key.removeprefix("module.").replace("buttleneck", "buttelneck"): value
                for key, value in state_dict.items()
            }
            model = InpaintNet()
            expected = model.state_dict()
            missing = sorted(set(expected) - set(state_dict))
            unexpected = sorted(set(state_dict) - set(expected))
            shape_mismatches = sorted(
                key for key in set(expected) & set(state_dict)
                if tuple(expected[key].shape) != tuple(state_dict[key].shape)
            )
            if missing or unexpected or shape_mismatches:
                status = {
                    "loaded": False,
                    "error": "InpaintNet checkpoint key or tensor-shape mismatch",
                    "missing": missing,
                    "unexpected": unexpected,
                    "shape_mismatches": shape_mismatches,
                }
                record_model_health("inpaintnet", status)
                logger.warning("InpaintNet checkpoint incompatible; trajectory repair disabled",
                               missing=str(missing), unexpected=str(unexpected),
                               shape_mismatches=str(shape_mismatches))
                self.inpaintnet = None
                return

            model.load_state_dict(state_dict, strict=True)
            self.inpaintnet = model.to(self.device)
            self.inpaintnet.eval()
            record_model_health("inpaintnet", {"loaded": True, "error": None,
                                                "n_tensors": len(expected)})
            logger.info(f"InpaintNet loaded successfully from {path}")
        except Exception as e:
            logger.warning(f"InpaintNet load error: {e}, proceeding without trajectory rectification")
            self.inpaintnet = None
            record_model_health("inpaintnet", {"loaded": False, "error": str(e)})

    def _preprocess(
        self,
        frames: list[np.ndarray],
        crop_rect: tuple[float, float, float, float] | None = None,
    ) -> list[np.ndarray]:
        """Preprocess each frame: resize → RGB → normalize.

        Returns list of (H, W, 3) float32 arrays in [0, 1].
        """
        import cv2
        processed = []
        clamped_crop = _clamp_crop_rect(crop_rect, frames[0].shape[1], frames[0].shape[0]) if frames else None
        for frame in frames:
            if clamped_crop is not None:
                x0, y0, x1, y1 = clamped_crop
                frame = frame[y0:y1, x0:x1]
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            processed.append(normalized)
        return processed

    def _predict_raw_detections(
        self,
        frames: list[np.ndarray],
        *,
        batch_size: int,
        original_size: tuple[int, int],
        crop_rect: tuple[float, float, float, float] | None,
        heat_threshold: float,
    ) -> list[tuple[float, float, float] | None]:
        """Run TrackNet once and return raw per-frame detections before rectification."""
        if self.model is None:
            raise RuntimeError("TrackNetV3 backbone not loaded")
        if len(frames) < 1:
            raise RuntimeError("Need at least 1 frame")
        from app.config.settings import settings
        orig_w, orig_h = original_size
        clamped_crop = _clamp_crop_rect(crop_rect, orig_w, orig_h) if crop_rect is not None else None
        if clamped_crop is None:
            preprocessed = self._preprocess(frames)
        else:
            preprocessed = self._preprocess(frames, crop_rect=clamped_crop)
        n_frames = len(preprocessed)

        # A running mean produces the static background without materialising a
        # video-sized (frames, height, width, channels) temporary array.
        background = np.zeros_like(preprocessed[0], dtype=np.float32)
        for index, frame in enumerate(preprocessed, start=1):
            background += (frame - background) / index
        all_raw = [None] * n_frames
        temporal_weights = _triangular_weights(TRACKNET_SEQUENCE_LENGTH)
        pending_heatmaps: dict[int, np.ndarray] = {}
        pending_weights: dict[int, float] = {}
        prev_accepted: tuple[float, float, float] | None = None
        prev_prev_accepted: tuple[float, float, float] | None = None

        # Starts -7 through n-1 ensure every real frame is present at every
        # temporal output offset while boundary frames are edge-padded.
        window_starts = list(range(-(TRACKNET_SEQUENCE_LENGTH - 1), n_frames))
        for chunk_start in range(0, len(window_starts), batch_size):
            chunk_starts = window_starts[chunk_start:chunk_start + batch_size]
            batch_windows = []
            for start in chunk_starts:
                batch_windows.append(_build_input(_build_8frame_window(preprocessed, start), background))

            batch_tensor = torch.from_numpy(np.stack(batch_windows)).float().to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()

            for start, heatmaps in zip(chunk_starts, probabilities):
                for offset, heatmap in enumerate(heatmaps):
                    frame = start + offset
                    if not 0 <= frame < n_frames:
                        continue
                    weight = temporal_weights[offset]
                    if frame not in pending_heatmaps:
                        pending_heatmaps[frame] = weight * heatmap
                        pending_weights[frame] = weight
                    else:
                        pending_heatmaps[frame] += weight * heatmap
                        pending_weights[frame] += weight

                # Once this window has been seen, no later start can produce
                # frame ``start``. Decode and release it immediately.
                if start >= 0:
                    aggregate = pending_heatmaps.pop(start) / pending_weights.pop(start)
                    candidates = _extract_component_candidates(
                        aggregate,
                        orig_w,
                        orig_h,
                        threshold=heat_threshold,
                        max_components=settings.tracknet_candidate_components,
                        crop_rect=clamped_crop,
                    )
                    candidate = _select_detection_candidate(
                        candidates,
                        prev_accepted,
                        prev_prev_accepted,
                        motion_weight=settings.tracknet_component_motion_weight,
                        confidence_weight=settings.tracknet_component_confidence_weight,
                        distance_scale_px=settings.tracknet_component_distance_scale_px,
                    )
                    if _accept_detection_candidate(
                        candidate,
                        prev_accepted,
                        prev_prev_accepted,
                        min_conf=settings.shuttle_min_conf,
                        trust_min_conf=settings.tracknet_detection_min_conf,
                        low_conf_max_jump_px=settings.tracknet_low_conf_max_jump_px,
                        distance_scale_px=settings.tracknet_component_distance_scale_px,
                    ):
                        accepted = candidate[:3]
                        all_raw[start] = accepted
                        prev_prev_accepted = prev_accepted
                        prev_accepted = accepted
                    else:
                        all_raw[start] = None

            del batch_tensor
            if self.device != "cpu":
                torch.cuda.empty_cache()
        return all_raw

    def predict_batch(self, frames: list[np.ndarray], batch_size: int | None = None,
                      original_size: tuple | None = None,
                      crop_rect: tuple[float, float, float, float] | None = None,
                      far_crop_rect: tuple[float, float, float, float] | None = None,
                      far_threshold: float | None = None,
                      net_y: float | None = None) -> list[dict]:
        """Run TrackNetV3 + optional far-tile fill + InpaintNet on a batch of frames."""
        if self.model is None:
            raise RuntimeError("TrackNetV3 backbone not loaded")

        if len(frames) < 1:
            raise RuntimeError("Need at least 1 frame")

        if batch_size is None:
            from app.config.gpu_batch import get_gpu_batch_config
            batch_size = get_gpu_batch_config(self.device)["tracknet_chunk"]

        orig_w, orig_h = original_size if original_size else (frames[0].shape[1], frames[0].shape[0])
        from app.config.settings import settings

        primary_raw = self._predict_raw_detections(
            frames,
            batch_size=batch_size,
            original_size=(orig_w, orig_h),
            crop_rect=crop_rect,
            heat_threshold=settings.shuttle_min_conf,
        )

        primary_points = np.array([
            (item[0], item[1]) if item is not None else (np.nan, np.nan)
            for item in primary_raw
        ], dtype=np.float64)
        primary_points, primary_removed = _gate_tracknet_spikes(
            primary_points,
            max_step_px=settings.tracknet_pre_rectify_max_image_step_px,
        )
        all_raw = [
            item if np.isfinite(primary_points[index]).all() else None
            for index, item in enumerate(primary_raw)
        ]

        if far_crop_rect is not None and net_y is not None:
            far_raw = self._predict_raw_detections(
                frames,
                batch_size=batch_size,
                original_size=(orig_w, orig_h),
                crop_rect=far_crop_rect,
                heat_threshold=float(far_threshold if far_threshold is not None else settings.tracknet_far_heat_threshold),
            )
            far_points = np.array([
                (item[0], item[1]) if item is not None else (np.nan, np.nan)
                for item in far_raw
            ], dtype=np.float64)
            far_points, far_removed = _gate_tracknet_spikes(
                far_points,
                max_step_px=settings.tracknet_pre_rectify_max_image_step_px,
            )
            merged_points, far_filled = _merge_far_tile_tracks(
                primary_points,
                far_points,
                net_y=float(net_y),
            )
            for index in range(len(all_raw)):
                if np.isnan(merged_points[index]).any():
                    all_raw[index] = None
                elif np.isnan(primary_points[index]).any() and np.isfinite(far_points[index]).all():
                    all_raw[index] = far_raw[index]
            from app.pipeline.shared.logging import logger
            logger.info(
                "TrackNet pre-rectify gating",
                primary_removed=str(primary_removed),
                far_removed=str(far_removed),
                far_filled=str(far_filled),
            )
        else:
            from app.pipeline.shared.logging import logger
            logger.info(
                "TrackNet pre-rectify gating",
                primary_removed=str(primary_removed),
                far_removed="0",
                far_filled="0",
            )

        original_missing = [item is None for item in all_raw]
        if self.inpaintnet is not None:
            all_raw = self._rectify_trajectory(all_raw, orig_w, orig_h)
        results = []
        repaired_confidence = max(settings.shuttle_clean_min_conf, settings.shuttle_min_conf + 0.05)
        for frame, item in enumerate(all_raw):
            if item is None:
                results.append({"x": 0.0, "y": 0.0, "confidence": 0.0})
            else:
                result = {"x": item[0], "y": item[1], "confidence": item[2]}
                if original_missing[frame]:
                    result["confidence"] = repaired_confidence
                    result["was_repaired"] = True
                results.append(result)

        return results

    def predict(self, frames: list[np.ndarray], original_size: tuple | None = None) -> list[dict]:
        """Single-frame-group prediction (legacy interface)."""
        return self.predict_batch(frames, batch_size=1, original_size=original_size)

    def _rectify_trajectory(self, raw_detections: list[tuple | None],
                            orig_w: int, orig_h: int) -> list[tuple]:
        """Repair only originally missing points with masked InpaintNet inference."""
        n = len(raw_detections)
        if self.inpaintnet is None or n == 0:
            return raw_detections

        coords_px = np.array([
            (d[0], d[1]) if d is not None else (np.nan, np.nan)
            for d in raw_detections
        ], dtype=np.float64)
        valid = np.isfinite(coords_px).all(axis=1)
        if not valid.any() or valid.all():
            return raw_detections

        norm = coords_px.copy()
        norm[:, 0] = np.clip(norm[:, 0] / max(orig_w, 1), 0.0, 1.0)
        norm[:, 1] = np.clip(norm[:, 1] / max(orig_h, 1), 0.0, 1.0)
        filled = norm.copy()
        indices = np.arange(n)
        for coordinate in range(2):
            filled[~valid, coordinate] = np.interp(
                indices[~valid], indices[valid], norm[valid, coordinate]
            )

        sequence_length = min(16, n)
        center = (sequence_length - 1) / 2.0
        weights = np.exp(-((np.arange(sequence_length) - center) ** 2) /
                         (2 * (sequence_length / 2.0) ** 2))
        weighted_predictions = np.zeros((n, 2), dtype=np.float64)
        weight_sums = np.zeros(n, dtype=np.float64)
        missing = ~valid
        with torch.no_grad():
            for start in range(n - sequence_length + 1):
                stop = start + sequence_length
                coords_tensor = torch.tensor(filled[start:stop], dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                mask_tensor = torch.tensor(missing[start:stop, None], dtype=torch.float32,
                                           device=self.device).unsqueeze(0)
                predicted = self.inpaintnet(coords_tensor, mask_tensor)[0].cpu().numpy()
                for offset in range(sequence_length):
                    frame = start + offset
                    if missing[frame]:
                        weighted_predictions[frame] += weights[offset] * predicted[offset]
                        weight_sums[frame] += weights[offset]

        repaired = list(raw_detections)
        for frame in np.flatnonzero(missing):
            if weight_sums[frame] == 0:
                continue
            point = weighted_predictions[frame] / weight_sums[frame]
            repaired[frame] = (float(np.clip(point[0], 0.0, 1.0) * orig_w),
                               float(np.clip(point[1], 0.0, 1.0) * orig_h), 0.0)
        return repaired
