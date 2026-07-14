#!/usr/bin/env python3
"""
BMCA - Badminton Match Coaching Assistant
Self-contained pipeline for Colab/Kaggle GPU execution.

Keeps the GPU ML batch loop (YOLO/TrackNet/RTMPose) for memory efficiency,
then delegates CPU stages to backend pipeline via ArtifactStore.

Usage:
    python pipeline.py video.mp4 --output report.json --device cuda

Requirements:
    pip install torch torchvision ultralytics onnxruntime-gpu opencv-python-headless scipy numpy pyyaml gdown tqdm pydantic-settings
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Shared module imports ──────────────────────────────────────────────────
# Add backend to path for shared modules (unification with backend pipeline)
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config.settings import settings
from app.models.tracknet import (
    InpaintNet,
    _accept_detection_candidate,
    _clamp_crop_rect,
    _court_crop_rect,
    _extract_component_candidates,
    _gate_tracknet_spikes,
    _merge_far_tile_tracks,
    _select_detection_candidate,
)
from app.pipeline.shared.court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT,
    _correct_court_points, compute_homography, image_to_court, HomographySmoother,
    detect_court_hough_lines,
)
from app.pipeline.shared.utils import (
    stage_rally_stats,
)
from app.pipeline.shared.core import STROKE_CLASSES, _get_gpu_batch_config
from app.pipeline.shared.models import ensure_model, MODEL_REGISTRY
from app.pipeline.shuttle import _add_court_space_columns, compute_shuttle_in_court_fraction
from app.pipeline.shared.shuttle_utils import clean_trajectory

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)

# Resolve model paths via the centralized registry
TRACKNET_PATH = MODEL_REGISTRY["tracknet"][0]
INPAINTNET_PATH = MODEL_REGISTRY.get("inpaintnet", (CKPT_DIR / "InpaintNet_best.pt",))[0]
YOLOV8_MODEL = str(MODEL_REGISTRY["yolov8s"][0])
RTMOPOSE_PATH = MODEL_REGISTRY["rtmpose_colab"][0]
COURT_KP_MODEL_PATH = MODEL_REGISTRY["court_kprcnn"][0]
RTMOPOSE_PATH_ALT = MODEL_REGISTRY["rtmpose"][0]
BST_PATH = MODEL_REGISTRY["bst_colab"][0]
HRNET_PATH = MODEL_REGISTRY["hrnet"][0]

def setup_models(device: str, pose_model: str = "rtmpose"):
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except Exception:
            pass
    print("Setting up models...")

    # TrackNet (shuttle tracking)
    if not TRACKNET_PATH.exists():
        path = ensure_model("tracknet")
        if path is None:
            print("  Shuttle tracking will use fallback")

    # InpaintNet (trajectory rectification)
    if not INPAINTNET_PATH.exists():
        path = ensure_model("inpaintnet")
        if path is None:
            print("  InpaintNet will not be available")

    # Court keypoint model (SoloShuttlePose)
    if not COURT_KP_MODEL_PATH.exists():
        ensure_model("court_kprcnn")

    # YOLOv8 (Ultralytics auto-downloads on first use)
    from ultralytics import YOLO
    YOLO(YOLOV8_MODEL)

    # RTMPose (colab variant: _8xb64-270e_coco-256x192)
    if not RTMOPOSE_PATH.exists() and not RTMOPOSE_PATH_ALT.exists():
        path = ensure_model("rtmpose_colab")
        if path is None:
            ensure_model("rtmpose")  # fall back to backend variant

    # BST (colab variant: bst_CG_JnB_bone_merged)
    if not BST_PATH.exists():
        ensure_model("bst_colab")

    # HRNet (optional, for mmpose/hybrid mode)
    if not HRNET_PATH.exists() and pose_model in ("mmpose", "hybrid"):
        path = ensure_model("hrnet")
        if path is None:
            print("  HRNet not available, falling back to RTMPose")

    print("Models ready.\n")


def _install_mmpose_deps():
    """Install mmcv + mmpose dependencies.

    If mmcv is already installed (e.g. from Colab cell 1), skip.
    Otherwise download pre-built archive from Google Drive (MMCV_DRIVE_FILE_ID env var).
    """
    import subprocess

    # Skip if mmcv already present
    try:
        import mmcv
        print(f"    mmcv {mmcv.__version__} already installed")
    except ImportError:
        drive_file_id = os.environ.get("MMCV_DRIVE_FILE_ID", "")
        if not drive_file_id:
            raise RuntimeError(
                "MMCV_DRIVE_FILE_ID not set. Build mmcv locally with colab/build_mmcv.sh "
                "and upload mmcv_files.tar.gz to Google Drive."
            )
        import gdown, tarfile, sysconfig
        tar_path = str(CKPT_DIR / "mmcv_files.tar.gz")
        print(f"    Downloading pre-built mmcv from Google Drive...")
        gdown.download(id=drive_file_id, output=tar_path, quiet=False)
        site_dir = sysconfig.get_path("purelib")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(site_dir)
        os.remove(tar_path)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "mmpose", "mmdet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "mmengine", "chumpy", "json-tricks", "matplotlib", "munkres", "xtcocotools", "pillow"])


def _export_hrnet_onnx():
    """Export MMPose default human pose model to ONNX."""
    import torch

    # Bypass mmpose's overly strict mmcv<2.2.0 check (2.2.0 works fine at runtime)
    import mmcv
    if not hasattr(mmcv, "__version_checked__"):
        mmcv.__version__ = "2.1.0"

    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer('human')
    pose_estimator = inferencer.pose_estimator
    if hasattr(pose_estimator, 'cfg'):
        print(f"    Resolved config: {pose_estimator.cfg.filename}")
    dummy = torch.randn(1, 3, 256, 192)
    if torch.cuda.is_available():
        dummy = dummy.cuda()
        pose_estimator = pose_estimator.cuda()
    torch.onnx.export(
        pose_estimator, dummy, str(HRNET_PATH),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    print(f"    Exported HRNet to {HRNET_PATH} ({HRNET_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


class TrackNetV3:
    def __init__(self, model_path: str, device: str = "cuda", chunk_size: int = 16,
                 inpaintnet_path: str | None = None):
        import torch
        import torch.nn as nn

        self.device = device
        self.model = None
        self.inpaintnet = None
        self.input_height = 288
        self.input_width = 512
        self._tracknet_chunk = chunk_size

        if inpaintnet_path and Path(inpaintnet_path).exists():
            try:
                checkpoint = torch.load(inpaintnet_path, map_location=device, weights_only=False)
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
                    raise ValueError(
                        f"checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}, "
                        f"shape_mismatches={shape_mismatches[:5]}"
                    )
                model.load_state_dict(state_dict, strict=True)
                self.inpaintnet = model
                self.inpaintnet.to(device).eval()
                if device == "cuda":
                    self.inpaintnet = self.inpaintnet.half()
                print(f"  InpaintNet loaded from {Path(inpaintnet_path).name}")
            except Exception as e:
                print(f"  InpaintNet load failed: {e}")

        if not Path(model_path).exists():
            return

        class SingleConv(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
            def forward(self, x):
                return torch.relu(self.bn(self.conv(x)))

        class TrackNetV3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.down_block_1 = nn.ModuleDict({'conv_1': SingleConv(27, 64), 'conv_2': SingleConv(64, 64)})
                self.down_block_2 = nn.ModuleDict({'conv_1': SingleConv(64, 128), 'conv_2': SingleConv(128, 128)})
                self.down_block_3 = nn.ModuleDict({'conv_1': SingleConv(128, 256), 'conv_2': SingleConv(256, 256), 'conv_3': SingleConv(256, 256)})
                self.bottleneck = nn.ModuleDict({'conv_1': SingleConv(256, 512), 'conv_2': SingleConv(512, 512), 'conv_3': SingleConv(512, 512)})
                self.up_block_1 = nn.ModuleDict({'conv_1': SingleConv(768, 256), 'conv_2': SingleConv(256, 256), 'conv_3': SingleConv(256, 256)})
                self.up_block_2 = nn.ModuleDict({'conv_1': SingleConv(384, 128), 'conv_2': SingleConv(128, 128)})
                self.up_block_3 = nn.ModuleDict({'conv_1': SingleConv(192, 64), 'conv_2': SingleConv(64, 64)})
                self.predictor = nn.Conv2d(64, 8, 1)

            def forward(self, x):
                d1 = self.down_block_1['conv_2'](self.down_block_1['conv_1'](x))
                d1_pool = nn.functional.max_pool2d(d1, 2)
                d2 = self.down_block_2['conv_2'](self.down_block_2['conv_1'](d1_pool))
                d2_pool = nn.functional.max_pool2d(d2, 2)
                d3 = self.down_block_3['conv_3'](self.down_block_3['conv_2'](self.down_block_3['conv_1'](d2_pool)))
                d3_pool = nn.functional.max_pool2d(d3, 2)
                b = self.bottleneck['conv_3'](self.bottleneck['conv_2'](self.bottleneck['conv_1'](d3_pool)))
                b_up = nn.functional.interpolate(b, size=d3.shape[2:], mode='bilinear', align_corners=True)
                u1 = self.up_block_1['conv_3'](self.up_block_1['conv_2'](self.up_block_1['conv_1'](torch.cat([b_up, d3], dim=1))))
                u1_up = nn.functional.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=True)
                u2 = self.up_block_2['conv_2'](self.up_block_2['conv_1'](torch.cat([u1_up, d2], dim=1)))
                u2_up = nn.functional.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=True)
                u3 = self.up_block_3['conv_2'](self.up_block_3['conv_1'](torch.cat([u2_up, d1], dim=1)))
                return self.predictor(u3)

        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            if isinstance(state_dict, dict) and not callable(state_dict):
                self.model = TrackNetV3Model()
                self.model.load_state_dict(state_dict)
                self.model.to(device).eval()
                if device == "cuda":
                    self.model = self.model.half()
                print(f"  TrackNet loaded from {Path(model_path).name}")
            else:
                print(f"  TrackNet state_dict not recognized")
        except Exception as e:
            print(f"  TrackNet load failed: {e}")
            self.model = None

    def _rectify_trajectory(self, raw_detections: list[tuple | None],
                            orig_w: int, orig_h: int) -> list[tuple]:
        import torch
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
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        with torch.no_grad():
            for start in range(n - sequence_length + 1):
                stop = start + sequence_length
                coords_tensor = torch.tensor(filled[start:stop], dtype=dtype,
                                             device=self.device).unsqueeze(0)
                mask_tensor = torch.tensor(missing[start:stop, None], dtype=dtype,
                                           device=self.device).unsqueeze(0)
                predicted = self.inpaintnet(coords_tensor, mask_tensor)[0].float().cpu().numpy()
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

    def _predict_raw_detections(self, frames, *, original_size=None, crop_rect=None, heat_threshold=None):
        import torch
        if self.model is None or len(frames) < 1:
            return [None for _ in frames]

        ow = original_size[0] if original_size else frames[0].shape[1]
        oh = original_size[1] if original_size else frames[0].shape[0]
        clamped_crop = _clamp_crop_rect(crop_rect, ow, oh) if crop_rect is not None else None

        preprocessed = np.empty((len(frames), self.input_height, self.input_width, 3), dtype=np.float32)
        for i, f in enumerate(frames):
            if clamped_crop is not None:
                x0, y0, x1, y1 = clamped_crop
                f = f[y0:y1, x0:x1]
            r = cv2.resize(f, (self.input_width, self.input_height))
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            preprocessed[i] = r
        del frames

        CHUNK = self._tracknet_chunk
        raw_results = [None for _ in range(len(preprocessed))]
        prev_accepted = None
        prev_prev_accepted = None

        for chunk_start in range(0, len(preprocessed), CHUNK):
            chunk_end = min(chunk_start + CHUNK, len(preprocessed))
            windows = []
            for i in range(chunk_start, chunk_end):
                start = max(0, i - 8)
                window = preprocessed[start:i + 1]
                pad_len = 9 - len(window)
                if pad_len > 0:
                    window = np.concatenate([window[:1]] * pad_len + [window], axis=0)
                windows.append(window[-9:].reshape(self.input_height, self.input_width, 27))

            batch = np.stack(windows).transpose(0, 3, 1, 2)
            tensor = torch.from_numpy(batch).float().to(self.device)
            if self.device == "cuda":
                tensor = tensor.half()
            with torch.no_grad():
                out = self.model(tensor)
            heatmaps = 1 / (1 + np.exp(-out.cpu().numpy()[:, 0]))
            for j in range(len(windows)):
                hm = heatmaps[j]
                candidates = _extract_component_candidates(
                    hm,
                    ow,
                    oh,
                    float(heat_threshold if heat_threshold is not None else settings.shuttle_min_conf),
                    settings.tracknet_candidate_components,
                    crop_rect=clamped_crop,
                )
                detection = _select_detection_candidate(
                    candidates,
                    prev_accepted,
                    prev_prev_accepted,
                    motion_weight=settings.tracknet_component_motion_weight,
                    confidence_weight=settings.tracknet_component_confidence_weight,
                    distance_scale_px=settings.tracknet_component_distance_scale_px,
                )
                if _accept_detection_candidate(
                    detection,
                    prev_accepted,
                    prev_prev_accepted,
                    min_conf=settings.shuttle_min_conf,
                    trust_min_conf=settings.tracknet_detection_min_conf,
                    low_conf_max_jump_px=settings.tracknet_low_conf_max_jump_px,
                    distance_scale_px=settings.tracknet_component_distance_scale_px,
                ):
                    accepted = detection[:3]
                    raw_results[chunk_start + j] = accepted
                    prev_prev_accepted = prev_accepted
                    prev_accepted = accepted
                else:
                    raw_results[chunk_start + j] = None
            del tensor, out, heatmaps, batch, windows
        return raw_results

    def predict_batch(self, frames, original_size=None, crop_rect=None, far_crop_rect=None,
                      far_threshold=None, net_y=None):
        if self.model is None or len(frames) < 1:
            return [{"x": 0, "y": 0, "confidence": 0} for _ in frames]

        ow = original_size[0] if original_size else frames[0].shape[1]
        oh = original_size[1] if original_size else frames[0].shape[0]

        primary_raw = self._predict_raw_detections(
            list(frames),
            original_size=(ow, oh),
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
        raw_results = [
            item if np.isfinite(primary_points[index]).all() else None
            for index, item in enumerate(primary_raw)
        ]

        far_removed = 0
        far_filled = 0
        if far_crop_rect is not None and net_y is not None:
            far_raw = self._predict_raw_detections(
                list(frames),
                original_size=(ow, oh),
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
            for index in range(len(raw_results)):
                if np.isnan(merged_points[index]).any():
                    raw_results[index] = None
                elif np.isnan(primary_points[index]).any() and np.isfinite(far_points[index]).all():
                    raw_results[index] = far_raw[index]
        if primary_removed or far_removed or far_filled:
            print(f"  TrackNet pre-rectify gating: primary_removed={primary_removed} "
                  f"far_removed={far_removed} far_filled={far_filled}")

        original_missing = [item is None for item in raw_results]
        rectified = self._rectify_trajectory(raw_results, ow, oh)
        repaired_confidence = max(settings.shuttle_clean_min_conf, settings.shuttle_min_conf + 0.05)
        results = []
        for frame, result in enumerate(rectified):
            if result is None:
                results.append({"x": 0.0, "y": 0.0, "confidence": 0.0})
                continue
            item = {"x": result[0], "y": result[1], "confidence": result[2]}
            if original_missing[frame]:
                item["confidence"] = repaired_confidence
                item["was_repaired"] = True
            results.append(item)
        return results


class CourtKeypointDetector:
    """Court keypoint detector using Keypoint R-CNN (SoloShuttlePose).
    
    Detects 6 court keypoints per frame:
    - KP 0: far-left corner   (court: 0, 0)
    - KP 1: far-right corner  (court: 0, 6.10)
    - KP 2: net-left          (court: 6.7, 0)   — unreliable, often duplicates KP0
    - KP 3: net-right         (court: 6.7, 6.10)
    - KP 4: near-left corner  (court: 13.4, 0)
    - KP 5: near-right corner (court: 13.4, 6.10)
    
    Only KP0, KP1, KP4, KP5 (4 outer corners) are used for homography.
    Falls back to proportional estimation if model unavailable.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        import torchvision
        self.device = device
        self.model = None
        
        if not Path(model_path).exists():
            print(f"  Court KP model not found: {model_path}")
            return
        
        try:
            self.model = torch.load(model_path, map_location=device, weights_only=False)
            self.model.to(device).eval()
            print(f"  Court KP model loaded from {Path(model_path).name}")
        except Exception as e:
            print(f"  Court KP model load failed: {e}")
            self.model = None
    
    def detect(self, frame):
        """Detect court keypoints in a frame.
        
        Model outputs 6 keypoints:
        - 0: far-left corner, 1: far-right corner
        - 2: net-left (unreliable), 3: net-right  
        - 4: near-left corner, 5: near-right corner
        
        Returns: list of 6 [[x,y], ...] keypoints, or None if detection fails.
        """
        if self.model is None:
            return None
        
        import torch
        from torchvision.transforms import functional as F
        
        image = frame.copy()
        image = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
        
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        
        if len(high_scores_idxs) == 0:
            return None
        
        import torchvision
        post_nms_idxs = torchvision.ops.nms(
            output[0]['boxes'][high_scores_idxs],
            output[0]['scores'][high_scores_idxs], 0.3
        ).cpu().numpy()
        
        kps_list = output[0]['keypoints'][high_scores_idxs][post_nms_idxs]
        if len(kps_list) == 0:
            return None
        
        # Take the detection with highest score
        kps = kps_list[0].detach().cpu().numpy()
        
        points = [[int(kps[i][0]), int(kps[i][1])] for i in range(min(6, len(kps)))]
        
        if len(points) < 6:
            return None
        
        # Validate: bottom y must be below top y
        top_y = (points[0][1] + points[1][1]) / 2
        bot_y = (points[4][1] + points[5][1]) / 2
        if bot_y <= top_y:
            return None
        
        # Per-keypoint validation: near corners must be at bottom, far corners at top
        h = frame.shape[0]
        mid_y = h / 2
        # KP4 (near-left) and KP5 (near-right) must be in bottom half
        if points[4][1] < mid_y or points[5][1] < mid_y:
            return None
        # KP0 (far-left) and KP1 (far-right) must be in top half
        if points[0][1] > mid_y or points[1][1] > mid_y:
            return None
        
        return points
    
    def detect_with_fallback(self, frame):
        """Detect court with fallback chain: model → color+line.
        
        Returns: list of 4 corners [bl, br, tl, tr] for homography
        """
        # Try model first
        kps = self.detect(frame)
        if kps is not None and len(kps) == 6:
            # Use 4 outer corners only (KP2/KP3 ignored — unreliable):
            # KP0=far-left, KP1=far-right, KP4=near-left, KP5=near-right
            corners = [kps[4], kps[5], kps[0], kps[1]]
            if _corners_are_valid(corners):
                return corners
        
        # Fallback to color+line detection
        corners = detect_court_from_frame(frame)
        if _corners_are_valid(corners):
            return corners
        return None


class YOLOv8Tracker:
    def __init__(self, conf_threshold=0.3, device="cuda", yolo_chunk=100, yolo_batch=8):
        import torch
        torch.backends.cudnn.benchmark = False
        from ultralytics import YOLO
        self.model = YOLO(YOLOV8_MODEL)
        self.conf = conf_threshold
        self.device = device
        self._yolo_chunk = yolo_chunk
        self._yolo_batch = yolo_batch

    def track_batch(self, frames, global_frame_offsets, batch_size=None, yolo_chunk=None):
        all_det = {}
        if not frames:
            return all_det
        h, w = frames[0].shape[:2]
        if batch_size is None:
            batch_size = self._yolo_batch
        if yolo_chunk is None:
            yolo_chunk = self._yolo_chunk

        for chunk_start in range(0, len(frames), yolo_chunk):
            chunk = frames[chunk_start:chunk_start + yolo_chunk]
            from app.config.settings import settings
            results = self.model.track(
                chunk, classes=[0], conf=self.conf,
                verbose=False, persist=True, device=self.device,
                tracker=str(settings.tracker_config_path),
                batch=batch_size, stream=True
            )
            for local_idx, r in enumerate(results):
                global_idx = global_frame_offsets + chunk_start + local_idx
                dets = []
                if r.boxes is not None and r.boxes.id is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        bw, bh = x2 - x1, y2 - y1
                        bbox_area = bw * bh
                        frame_area = w * h
                        if bbox_area < frame_area * 0.001 or bbox_area > frame_area * 0.5:
                            continue
                        dets.append({"frame": global_idx, "bbox": [x1, y1, x2, y2],
                                   "confidence": box.conf[0].item(), "track_id": int(box.id[0].item())})
                dets.sort(key=lambda d: d["confidence"], reverse=True)
                dets = dets[:2]
                all_det[global_idx] = dets
            del results
            import gc; gc.collect()
        self._free_tracking_state()
        return all_det

    def _free_tracking_state(self):
        import torch, gc
        predictor = getattr(self.model, 'predictor', None)
        if predictor is not None:
            dataset = getattr(predictor, 'dataset', None)
            if dataset is not None:
                dataset.vid_cap = None
            predictor.vid_path = None
            predictor.vid_writer = None
            predictor.im0s = None
            predictor.s = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class RTMPoseEstimator:
    def __init__(self, model_path: str, device: str = "cuda", onnx_chunk: int = 64):
        self.model = None
        self.h, self.w = 256, 192
        self.model_type = "rtmpose"
        self._onnx_chunk = onnx_chunk
        if Path(model_path).exists():
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)
                n_outputs = len(self.model.get_outputs())
                if n_outputs == 1:
                    self.model_type = "hrnet"
                    print(f"  HRNet loaded from {Path(model_path).name}")
                else:
                    self.model_type = "rtmpose"
                    print(f"  RTMPose loaded from {Path(model_path).name}")
            except Exception as e:
                print(f"  Pose model load error: {e}")
        else:
            print(f"  Pose model not found: {model_path}")

    def _preprocess(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, (x1, y1, 0, 0)
        r = cv2.resize(crop, (self.w, self.h))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = (r - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = r.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return tensor, (x1, y1, x2 - x1, y2 - y1)

    def _decode_rtmpose(self, outputs, crop_info):
        x1, y1, crop_w, crop_h = crop_info
        simcc_x = outputs[0][0]
        simcc_y = outputs[1][0]
        x_coords = np.argmax(simcc_x, axis=1) / 2.0
        y_coords = np.argmax(simcc_y, axis=1) / 2.0
        x_conf = np.max(simcc_x, axis=1)
        y_conf = np.max(simcc_y, axis=1)
        conf = (x_conf + y_conf) / 2.0
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = x1 + x_coords * (crop_w / self.w)
        kps[:, 1] = y1 + y_coords * (crop_h / self.h)
        kps[:, 2] = 1.0 / (1.0 + np.exp(-conf))
        return kps

    def _decode_hrnet(self, outputs, crop_info):
        x1, y1, crop_w, crop_h = crop_info
        heatmap = outputs[0]
        if heatmap.ndim == 4:
            heatmap = heatmap[0]
        if heatmap.ndim == 3 and heatmap.shape[0] >= 17:
            kps = np.zeros((17, 3), dtype=np.float32)
            for k in range(min(17, heatmap.shape[0])):
                hm = heatmap[k]
                y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
                kps[k, 0] = x1 + (x_idx / hm.shape[1]) * crop_w
                kps[k, 1] = y1 + (y_idx / hm.shape[0]) * crop_h
                kps[k, 2] = float(hm.max())
            return kps
        out = heatmap
        kps = out.reshape(17, 3) if out.ndim == 3 else out[0]
        kps[:, 0] = x1 + kps[:, 0] * crop_w
        kps[:, 1] = y1 + kps[:, 1] * crop_h
        return kps

    def estimate(self, frame, bbox):
        if self.model is None:
            return np.zeros((17, 3), dtype=np.float32)
        tensor, crop_info = self._preprocess(frame, bbox)
        if tensor is None:
            return np.zeros((17, 3), dtype=np.float32)
        outputs = self.model.run(None, {"input": tensor})
        if self.model_type == "hrnet":
            return self._decode_hrnet(outputs, crop_info)
        return self._decode_rtmpose(outputs, crop_info)

    def estimate_batch(self, crops, onnx_chunk=None):
        if self.model is None:
            return [np.zeros((17, 3), dtype=np.float32) for _ in crops]
        if onnx_chunk is None:
            onnx_chunk = self._onnx_chunk

        results = [np.zeros((17, 3), dtype=np.float32) for _ in crops]
        input_name = self.model.get_inputs()[0].name

        for chunk_start in range(0, len(crops), onnx_chunk):
            chunk_end = min(chunk_start + onnx_chunk, len(crops))
            batch_tensors = []
            valid_indices = []
            crop_infos = []

            for i in range(chunk_start, chunk_end):
                bbox, frame = crops[i]
                tensor, crop_info = self._preprocess(frame, bbox)
                if tensor is None:
                    continue
                batch_tensors.append(tensor[0])
                valid_indices.append(i)
                crop_infos.append(crop_info)

            if not batch_tensors:
                continue

            batch_np = np.stack(batch_tensors)
            outputs = self.model.run(None, {input_name: batch_np})

            for j, idx in enumerate(valid_indices):
                single_outputs = [out[j:j+1] for out in outputs]
                if self.model_type == "hrnet":
                    results[idx] = self._decode_hrnet(single_outputs, crop_infos[j])
                else:
                    results[idx] = self._decode_rtmpose(single_outputs, crop_infos[j])

        return results


# ─── Video Reader (memory-efficient) ────────────────────────────────────────

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / video_fps
    return total_frames, video_fps, width, height, duration


def frame_generator(video_path, sample_interval=3, target_fps=10):
    """Yield one frame at a time — never holds more than 1 frame in memory."""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            yield frame_idx, frame
        frame_idx += 1
    cap.release()


# ─── Pipeline Stages ─────────────────────────────────────────────────────────

def detect_court_from_frame(frame):
    """Detect court corners via the shared Hough-line trapezoid detector."""
    return detect_court_hough_lines(frame)


def _corners_are_valid(corners):
    if corners is None or len(corners) != 4:
        return False
    corrected = _correct_court_points(corners)
    _, valid = compute_homography(corrected)
    return bool(valid)


def _parse_court_corners_arg(value):
    parts = [int(c.strip()) for c in value.split(",") if c.strip()]
    if len(parts) != 8:
        raise ValueError("--court-corners requires 8 comma-separated integers")
    return [(parts[i], parts[i + 1]) for i in range(0, 8, 2)]


# ─── PRD §2.5: Per-frame homography with geometric validation ───────────────

CORNER_NAMES = ["outer_bl", "outer_br", "outer_tl", "outer_tr"]


def stage_court_detection(corners):
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [COURT_WIDTH, 0], [0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return {"homography": H.tolist(), "corners_pixel": [list(c) for c in corners],
            "court_length": COURT_LENGTH, "court_width": COURT_WIDTH, "net_height": NET_HEIGHT}




# ─── Main Pipeline ───────────────────────────────────────────────────────────

BATCH_SIZE = 300


def _generate_report(court, players_data, shots, rallies, coach,
                     tactical, fitness, footwork, technical, court_analytics, fps=30,
                     data_quality=None, physics_summary=None):
    """Build the final report dict from all analytics."""
    shot_dist = {}
    for pid, data in tactical.items():
        shot_dist.update(data.get("shot_distribution", {}))
    data_quality = data_quality or {}

    shots_with_ts = []
    for shot_idx, s in enumerate(shots, 1):
        entry = {
            "shot_id": shot_idx,
            "frame": s["frame"],
            "start_ts": round(s["frame"] / fps, 3),
            "stroke_type": s["stroke_type"],
            "confidence": round(s.get("stroke_confidence", 0.5), 3),
            "player_id": s.get("player_id"),
            "rally_id": s.get("rally_id"),
            "owner_confident": bool(s.get("owner_confident", False)),
            "owner_source": s.get("owner_source", "unknown"),
            "owner_reason": s.get("owner_reason", "missing"),
            "aim_alpha_reliable": bool(s.get("aim_alpha_reliable", False)),
            "aim_alpha_route": s.get("aim_alpha_route", "alpha_abstain_quality"),
        }
        # ts_end: next shot's start_ts, or +1s for last shot
        if shot_idx < len(shots):
            entry["ts_end"] = round(shots[shot_idx]["frame"] / fps, 3)
        else:
            entry["ts_end"] = round(s["frame"] / fps + 1.0, 3)
        if "side" in s:
            entry["side"] = s["side"]
        if "logits" in s:
            entry["logits"] = s["logits"]
        if "stroke_source" in s:
            entry["stroke_source"] = s["stroke_source"]
        if "shuttleset_class_id" in s:
            entry["shuttleset_class_id"] = s["shuttleset_class_id"]
        shots_with_ts.append(entry)

    return {
        "court_analytics": court_analytics, "footwork": footwork, "fitness": fitness,
        "tactical": tactical, "technical": technical,
        "shot_distribution": shot_dist,
        "strengths": coach["strengths"], "weaknesses": coach["weaknesses"],
        "top_3_improvements": coach["top_3_improvements"],
        "recommended_drills": coach["recommended_drills"], "evidence": coach["evidence"],
        "rally_stats": coach.get("rally_stats", {}),
        "rallies": rallies, "shot_count": len(shots),
        "shots": shots_with_ts,
        "physics_summary": physics_summary or {},
        "data_quality": {k: v for k, v in data_quality.items()
                         if k != "court_valid" and k != "model_health"},
    }


def run_pipeline(video_path: str, output_path: str, device: str = "cuda", pose_model: str = "rtmpose", sample_rate: int = 0, court_corners: list[tuple[int, int]] | None = None):
    """Run the full BMCA pipeline.

    Keeps the GPU ML batch loop for memory efficiency (YOLO/TrackNet/RTMPose),
    then delegates CPU analytics stages to backend pipeline via ArtifactStore.

    If court_corners is None, checks for a manual_corners.json file alongside
    output_path.  This allows the backend-stored manual corners to propagate
    to the colab runtime automatically.
    """
    import tempfile
    import json
    from app.pipeline.base import StageConfig
    from app.pipeline.players import stitch_tracks
    from app.pipeline.hits import HitFrameLocalizationStage
    from app.pipeline.strokes import StrokeClassificationStage
    from app.pipeline.attribution import PlayerAttributionStage
    from app.pipeline.rallies import RallySegmentationStage, finalize_rally_outcomes
    from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage
    from app.pipeline.analytics.footwork import FootworkAnalyticsStage
    from app.pipeline.analytics.fitness import FitnessAnalyticsStage
    from app.pipeline.analytics.tactical import TacticalAnalyticsStage
    from app.pipeline.analytics.technical import TechnicalAnalyticsStage
    from app.pipeline.quality import DataQualityStage
    from app.shuttle_coach.engine import analyze_from_pipeline
    from app.storage.artifacts import ArtifactStore

    start_time = time.time()
    video_name = Path(video_path).name

    gpu_cfg = _get_gpu_batch_config(device)
    print(f"=" * 60)
    print(f"  BMCA Pipeline - {video_name}")
    print(f"  Device: {device}")
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = (props.total_memory if hasattr(props, 'total_memory') else props.total_mem) / (1024 ** 3)
            print(f"  GPU: {props.name} ({vram_gb:.1f} GB)")
        else:
            print("  GPU: CUDA requested but not available, using CPU")
    except Exception as e:
        print(f"  GPU: detection failed — {e}")
    print(f"  Batch config: YOLO chunk={gpu_cfg['yolo_chunk']} batch={gpu_cfg['yolo_batch']}, "
          f"TrackNet chunk={gpu_cfg['tracknet_chunk']}, RTMPose chunk={gpu_cfg['rtmpose_chunk']}, "
          f"BST batch={gpu_cfg['bst_batch']}")
    print(f"=" * 60)

    setup_models(device, pose_model)

    total_frames, video_fps, vid_w, vid_h, duration = get_video_info(video_path)
    if sample_rate > 0:
        sample_interval = sample_rate
    else:
        sample_interval = max(1, int(video_fps / 10))
    num_samples = total_frames // sample_interval
    target_fps = video_fps / sample_interval
    print(f"  Video: {duration:.0f}s, {total_frames} frames @ {video_fps:.0f}fps ({vid_w}x{vid_h})")
    print(f"  Sampling: every {sample_interval} frames -> ~{num_samples} frames ({target_fps:.0f}fps)")
    print(f"  Batch size: {BATCH_SIZE} frames")

    # ── Court detection ──
    print("\n[1/5] Court detection...")
    court_kp_detector = CourtKeypointDetector(str(COURT_KP_MODEL_PATH), device=device)
    smoother = HomographySmoother(alpha=0.6, win=5)

    cap = cv2.VideoCapture(str(video_path))
    ret, sample_frame = cap.read()
    cap.release()

    # Fallback chain: CLI arg → manual_corners.json from output → default_corners.json in repo → auto-detect
    detected_corners = None
    detection_method = "none"

    if court_corners is not None:
        detected_corners = court_corners
        detection_method = "manual (CLI arg)"
        print(f"  Using manual corners from CLI: {detected_corners}")

    if detected_corners is None:
        corners_path = Path(output_path).parent / "manual_corners.json"
        if corners_path.exists():
            try:
                stored = json.loads(corners_path.read_text())
                raw = stored.get("corners")
                if raw and len(raw) == 4:
                    detected_corners = [(int(pt[0]), int(pt[1])) for pt in raw]
                    detection_method = "manual (persisted JSON)"
                    print(f"  Using manual corners from {corners_path}: {detected_corners}")
            except Exception as e:
                print(f"  Warning: failed to load {corners_path}: {e}")

    if detected_corners is None:
        default_corners_path = _BACKEND_DIR / "app" / "config" / "default_corners.json"
        if default_corners_path.exists():
            try:
                raw = json.loads(default_corners_path.read_text())
                if raw and len(raw) == 4:
                    detected_corners = [(int(pt[0]), int(pt[1])) for pt in raw]
                    detection_method = "manual (default_corners.json)"
                    print(f"  Using default corners from {default_corners_path}: {detected_corners}")
            except Exception as e:
                print(f"  Warning: failed to load default_corners.json: {e}")

    if detected_corners is None and ret and sample_frame is not None:
        detected_corners = court_kp_detector.detect_with_fallback(sample_frame)
        if detected_corners is not None:
            detection_method = "court_kpRCNN" if court_kp_detector.model is not None else "color+line"
            print(f"  Auto-detected court ({detection_method}): {detected_corners}")

    if detected_corners:
        corners = detected_corners
    else:
        margin_x = int(vid_w * 0.08)
        court_top = int(vid_h * 0.28)
        court_bottom = int(vid_h * 0.72)
        corners = [(margin_x, court_bottom), (vid_w - margin_x, court_bottom),
                   (margin_x, court_top), (vid_w - margin_x, court_top)]
        print(f"  Using proportional corners: {corners}")

    corrected_corners = _correct_court_points(corners)
    court = stage_court_detection(corrected_corners)
    H_raw, valid = compute_homography(corrected_corners)
    H_smooth, valid = smoother.update(corrected_corners, H_raw, valid)
    court["homography"] = H_smooth if H_smooth is not None else H_raw
    court["valid"] = valid
    court["detection_method"] = detection_method
    print(f"  Court geometry valid: {valid}")

    # ── Initialize ML models ──
    print("\n  Loading ML models...")
    tracker = YOLOv8Tracker(conf_threshold=0.5, device=device, yolo_chunk=gpu_cfg["yolo_chunk"], yolo_batch=gpu_cfg["yolo_batch"])
    tracknet = TrackNetV3(str(TRACKNET_PATH), device=device, chunk_size=gpu_cfg["tracknet_chunk"],
                          inpaintnet_path=str(INPAINTNET_PATH) if INPAINTNET_PATH.exists() else None)
    tracknet_aspect = float(tracknet.input_width) / float(tracknet.input_height)
    tracknet_crop_rect = None
    tracknet_far_crop_rect = None
    if settings.tracknet_court_crop_enabled and corners and len(corners) >= 4:
        tracknet_crop_rect = _court_crop_rect(
            corners,
            {
                "left": settings.tracknet_crop_margin_left,
                "right": settings.tracknet_crop_margin_right,
                "top": settings.tracknet_crop_margin_top,
                "bottom": settings.tracknet_crop_margin_bottom,
            },
            aspect=tracknet_aspect,
        )
        print(f"  TrackNet court crop: {tuple(round(v) for v in tracknet_crop_rect)}")
    if settings.tracknet_far_tile_enabled and valid and corners and len(corners) >= 4:
        tracknet_far_crop_rect = _court_crop_rect(
            corners,
            {
                "left": settings.tracknet_far_margin_left,
                "right": settings.tracknet_far_margin_right,
                "top": settings.tracknet_far_margin_top,
                "bottom": settings.tracknet_far_margin_bottom,
            },
            aspect=tracknet_aspect,
        )
        print(f"  TrackNet far tile: {tuple(round(v) for v in tracknet_far_crop_rect)}")
    pose_estimator = None
    pose_estimator_secondary = None

    if pose_model == "hybrid":
        if HRNET_PATH.exists():
            print(f"  Using HYBRID mode: MMPose (strokes) + RTMPose (hits)")
            pose_estimator = RTMPoseEstimator(str(HRNET_PATH), device=device, onnx_chunk=gpu_cfg["rtmpose_chunk"])
            rtmpose_path = str(RTMOPOSE_PATH_ALT if RTMOPOSE_PATH_ALT.exists() else RTMOPOSE_PATH)
            if not Path(rtmpose_path).exists():
                rtmpose_dir = CKPT_DIR / "rtmpose"
                onnx_files = list(rtmpose_dir.rglob("*.onnx"))
                if onnx_files:
                    rtmpose_path = str(onnx_files[0])
            pose_estimator_secondary = RTMPoseEstimator(rtmpose_path, device=device, onnx_chunk=gpu_cfg["rtmpose_chunk"])
        else:
            print("  WARNING: HRNet not found, falling back to RTMPose only")
            pose_model = "rtmpose"

    if pose_model == "mmpose" and HRNET_PATH.exists():
        print(f"  Using MMPose HRNet-W32 (accurate)")
        pose_estimator = RTMPoseEstimator(str(HRNET_PATH), device=device, onnx_chunk=gpu_cfg["rtmpose_chunk"])
    elif pose_model != "hybrid":
        pose_path = str(RTMOPOSE_PATH_ALT if RTMOPOSE_PATH_ALT.exists() else RTMOPOSE_PATH)
        if not Path(pose_path).exists():
            rtmpose_dir = CKPT_DIR / "rtmpose"
            onnx_files = list(rtmpose_dir.rglob("*.onnx"))
            if onnx_files:
                pose_path = str(onnx_files[0])
                print(f"  Found RTMPose at: {onnx_files[0]}")
            else:
                print(f"  WARNING: No RTMPose .onnx found in {rtmpose_dir}")
        print(f"  Using RTMPose (fast)")
        pose_estimator = RTMPoseEstimator(pose_path, device=device, onnx_chunk=gpu_cfg["rtmpose_chunk"])

    pose_display_name = {"mmpose": "MMPose HRNet-W32", "hybrid": "MMPose HRNet-W32", "rtmpose": "RTMPose"}.get(pose_model, "RTMPose")
    print("  Models loaded")

    # ── ML batch loop (GPU) ──
    all_shuttle = []
    all_det = {}
    all_pose = []
    all_pose_secondary = []
    all_player_detections = []
    sample_idx = 0
    batch_count = 0

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n[2/5] Running ML stages on {num_samples} sampled frames ({total_batches} batches)...")

    batch_frames = []
    batch_global_indices = []
    frame_idx = 0
    batch_pbar = tqdm(total=total_video_frames, desc="  Video frames", unit="frame", ncols=80)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            batch_frames.append(frame)
            batch_global_indices.append(sample_idx)
            sample_idx += 1
            if len(batch_frames) >= BATCH_SIZE:
                batch_count += 1
                _process_batch(batch_frames, batch_global_indices, sample_idx - len(batch_frames),
                               tracker, tracknet, pose_estimator, device,
                               all_shuttle, all_det, all_pose, all_player_detections,
                               batch_count, total_batches,
                               pose_estimator_secondary=pose_estimator_secondary,
                               all_pose_secondary=all_pose_secondary,
                               pose_model_name=pose_display_name, corners=corners,
                               crop_rect=tracknet_crop_rect, far_crop_rect=tracknet_far_crop_rect)
                batch_frames = []
                batch_global_indices = []
                gc.collect()
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
        frame_idx += 1
        batch_pbar.update(1)
    batch_pbar.close()

    if batch_frames:
        batch_count += 1
        _process_batch(batch_frames, batch_global_indices, sample_idx - len(batch_frames),
                       tracker, tracknet, pose_estimator, device,
                       all_shuttle, all_det, all_pose, all_player_detections,
                       batch_count, total_batches,
                       pose_estimator_secondary=pose_estimator_secondary,
                       all_pose_secondary=all_pose_secondary,
                       pose_model_name=pose_display_name, corners=corners,
                       crop_rect=tracknet_crop_rect, far_crop_rect=tracknet_far_crop_rect)
        gc.collect()

    cap.release()

    print(f"\n  ML stages complete:")
    print(f"    Shuttle: {len(all_shuttle)} frames")
    print(f"    Players: {len(all_player_detections)} detections")
    print(f"    Pose:    {len(all_pose)} frames")

    # TrackNet outputs pixel coordinates (already rectified internally by
    # linear interpolation + moving average smoothing). The pixel→court-space
    # conversion happens inside _build_clip (strokes.py) via image_to_court:
    # the single homography transform that both pipelines share. Do NOT apply
    # another image_to_court here — that would double-transform the coordinates.

    # Log raw ByteTrack fragmentation before stitching
    raw_ids = set(d.get("track_id") for d in all_player_detections if d.get("track_id") is not None)
    n_frames = len(set(d["frame"] for d in all_player_detections))
    print(f"    ByteTrack raw: {len(raw_ids)} unique track IDs across {n_frames} frames "
          f"({len(all_player_detections)} detections)")

    # Build player summary via shared stitch_tracks (nearest-centroid joint assignment)
    court_mid_y = (corners[0][1] + corners[2][1]) / 2 if corners and len(corners) >= 4 else (vid_h * 0.5)
    players_data = stitch_tracks(all_player_detections, court_mid_y)

    # Free ML models from GPU
    del tracker, tracknet, pose_estimator, pose_estimator_secondary
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Store ML outputs in ArtifactStore ──
    debug_dir = Path(output_path).parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(Path(tmpdir))
        config = StageConfig(gpu_enabled=False, debug_level=3)

        # Convert numpy types to native Python for JSON serialization
        def _to_json_safe(v):
            if isinstance(v, dict):
                return {k: _to_json_safe(v) for k, v in v.items()}
            elif isinstance(v, list):
                return [_to_json_safe(x) for x in v]
            elif hasattr(v, 'tolist'):
                return v.tolist()
            elif hasattr(v, 'item'):
                return v.item()
            return v
        court = _to_json_safe(court)

        store.set("court", court)
        store.set("video_resolution", {"width": vid_w, "height": vid_h})
        store.set("players", players_data)

        shuttle_df = pd.DataFrame(all_shuttle)
        if settings.shuttle_clean_enabled:
            n_before = len(shuttle_df)
            low_conf_before = int((shuttle_df["confidence"] < settings.shuttle_clean_min_conf).sum())
            x_a = shuttle_df["x"].values.astype(np.float64)
            y_a = shuttle_df["y"].values.astype(np.float64)
            jumps_before = int((np.sqrt(np.diff(x_a) ** 2 + np.diff(y_a) ** 2) > settings.shuttle_max_jump_px).sum())

            # Store raw (conf-gate only) for hit detection before full cleaning
            df_raw = shuttle_df.copy()
            raw_conf = df_raw["confidence"].values.astype(np.float64) < settings.shuttle_clean_min_conf
            df_raw.loc[raw_conf, "x"] = np.nan
            df_raw.loc[raw_conf, "y"] = np.nan
            store.set_parquet("shuttle_raw", df_raw)

            df_orig = shuttle_df.copy()
            shuttle_df = clean_trajectory(shuttle_df, settings)
            n_interp = int(shuttle_df["was_interpolated"].sum())
            n_spike = n_interp - int(
                ((df_orig["confidence"] < settings.shuttle_clean_min_conf).values
                 & shuttle_df["was_interpolated"].values).sum()
            )
            print(f"  Shuttle cleaned: {low_conf_before}/{n_before} low-conf, "
                  f"{max(0, n_spike)} spikes, {n_interp} interp, "
                  f"{jumps_before} >{settings.shuttle_max_jump_px:.0f}px jumps → ~0")
        else:
            store.set_parquet("shuttle_raw", shuttle_df.copy())

        if court and court.get("valid", False) and court.get("homography") is not None:
            from app.pipeline.shared.court import court_geometry_reliable

            H = np.array(court["homography"])
            frac = compute_shuttle_in_court_fraction(
                shuttle_df, H,
                min_conf=settings.court_shuttle_reliability_min_conf,
                oob_margin=settings.shuttle_oob_margin_meters,
            )
            geom_ok = bool(court.get("valid")) and court_geometry_reliable(
                court.get("corners_pixel") or court.get("corners")
            )
            geom_ok = geom_ok and frac >= settings.court_shuttle_in_bounds_min_fraction
            court["geometry_reliable"] = geom_ok
            court["shuttle_in_court_fraction"] = frac
            shuttle_df = _add_court_space_columns(
                shuttle_df, H, float(video_fps), geometry_reliable=geom_ok
            )
            print(f"  Added court-space columns to shuttle data (reliable={geom_ok}, frac={frac:.2f})")

        store.set_parquet("shuttle", shuttle_df)

        pose_df = pd.DataFrame(all_pose)
        store.set_parquet("pose", pose_df)

        if all_pose_secondary:
            store.set("pose_secondary", all_pose_secondary)

        pd.DataFrame(all_player_detections).to_parquet(debug_dir / "player_detections.parquet", index=False)
        # Dump the same enriched artifacts the stage uses — not the pre-clean all_shuttle list.
        store.get_parquet("shuttle").to_parquet(debug_dir / "shuttle.parquet", index=False)
        raw_dbg = store.get_parquet("shuttle_raw")
        if raw_dbg is not None:
            raw_dbg.to_parquet(debug_dir / "shuttle_raw.parquet", index=False)
        pd.DataFrame(all_pose).to_parquet(debug_dir / "pose.parquet", index=False)

        # ── CPU stages via backend ──
        print("\n[3/5] Hit frame localization + stroke classification...")
        hits_result = HitFrameLocalizationStage().run(store, config)
        hit_count = hits_result.metadata.get("hit_count", 0)
        print(f"  Found {hit_count} hit frames")
        hits_df = store.get_parquet("hits")
        if hits_df is not None and len(hits_df) > 0:
            hits_df.to_parquet(debug_dir / "hits.parquet", index=False)

        # Use colab's BST for stroke classification (GPU-efficient, keeps BST in VRAM)
        print("\n  Stroke classification (colab BST)...")
        bst_pose = all_pose
        if pose_model == "hybrid" and all_pose_secondary:
            nonzero_count = sum(1 for p in all_pose[:100] if np.any(np.array(p["keypoints"]) != 0))
            if nonzero_count < 10:
                print(f"  HRNet keypoints mostly zero ({nonzero_count}/100 non-zero), using RTMPose for BST")
                bst_pose = all_pose_secondary
            else:
                print(f"  HRNet keypoints valid ({nonzero_count}/100 non-zero)")

        hits_df = store.get_parquet("hits")
        # Propagate colab's GPU-detected BST batch size to backend stage
        config.extra["bst_batch"] = gpu_cfg.get("bst_batch", 32)
        # bst_joint_norm defaults to "bbox" (settings.py) — matches training normalization
        settings.joint_velocity_amplification = 0.0
        shots_result = StrokeClassificationStage().run(store, config)
        shots_df = store.get_parquet("shots")
        shots = shots_df.to_dict("records") if shots_df is not None and len(shots_df) > 0 else []
        physics_summary = store.get("physics_summary") or shots_result.metadata.get("physics_summary", {})
        print(f"  Classified {len(shots)} shots")
        for shot_idx, s in enumerate(shots, 1):
            s["shot_id"] = shot_idx
            s["start_ts"] = round(s["frame"] / video_fps, 3)

        print("\n[4/5] Attribution + rallies + analytics...")
        # Rally segmentation
        rallies_result = RallySegmentationStage().run(store, config)
        rallies_df = store.get_parquet("rallies")
        rallies = rallies_df.to_dict("records") if rallies_df is not None and len(rallies_df) > 0 else []
        print(f"  Rallies: {len(rallies)}")

        # Player attribution
        attribution_result = PlayerAttributionStage().run(store, config)
        shots_df = store.get_parquet("shots")
        shots = shots_df.to_dict("records") if shots_df is not None and len(shots_df) > 0 else []
        print(f"  Attributed {len(shots)} shots")
        pd.DataFrame(shots).to_parquet(debug_dir / "shots.parquet", index=False)

        # Copy debug parquet files from store to output dir
        for debug_key in ["debug_bst_outputs", "debug_bst_input_quality", "debug_hit_scores"]:
            df = store.get_parquet(debug_key)
            if df is not None and len(df) > 0:
                df.to_parquet(debug_dir / f"{debug_key}.parquet", index=False)

        # Finalize rally outcomes now that player attribution is complete
        if rallies_df is not None and len(rallies_df) > 0:
            shuttle_raw = store.get_parquet("shuttle_raw")
            court = store.get("court")
            players_data = store.get("players")
            rallies_df = finalize_rally_outcomes(rallies_df, shots_df, shuttle_raw, court, players_data, video_fps)
            rallies = rallies_df.to_dict("records")
        pd.DataFrame(rallies).to_parquet(debug_dir / "rallies.parquet", index=False)

        # Analytics stages
        CourtPositionAnalyticsStage().run(store, config)
        FootworkAnalyticsStage().run(store, config)
        FitnessAnalyticsStage().run(store, config)
        TacticalAnalyticsStage().run(store, config)
        TechnicalAnalyticsStage().run(store, config)

        # DataQualityStage needs video_metadata for pose_coverage calculation
        store.set("video_metadata", {
            "total_frames": total_frames,
            "source_fps": video_fps,
            "fps": target_fps,
        })
        DataQualityStage().run(store, config)

        court_analytics = store.get("court_analytics") or {}
        footwork = store.get("footwork_analytics") or {}
        fitness = store.get("fitness_analytics") or {}
        tactical = store.get("tactical_analytics") or {}
        technical = store.get("technical_analytics") or {}
        data_quality = store.get("data_quality") or {}
        print(f"  Court: {len(court_analytics.get('zone_transitions', []))} transitions")

        # ── Coach recommendations (backend engine) ──
        print("\n[5/5] Coach recommendations...")
        all_players = set(list(tactical.keys()) + list(fitness.keys()))
        if not all_players:
            all_players = {"player_1"}

        coach = {"strengths": [], "weaknesses": [], "top_3_improvements": [],
                 "recommended_drills": [], "evidence": [], "rally_stats": None}

        for pid in sorted(all_players):
            analytics = {
                "fitness_analytics": fitness,
                "tactical_analytics": tactical,
                "footwork_analytics": footwork,
                "court_analytics": court_analytics,
                "_rallies_df": rallies_df,
                "_shots_df": shots_df,
            }
            result = analyze_from_pipeline(analytics, {}, player_id=pid, data_quality=data_quality)
            for key in coach:
                if key in result:
                    if isinstance(coach[key], list):
                        coach[key].extend(result[key])
                    else:
                        coach[key] = result[key]

        rally_stats = stage_rally_stats(shots, rallies)
        coach["rally_stats"] = rally_stats
        print(f"  {len(coach['strengths'])} strengths, {len(coach['weaknesses'])} weaknesses")

    # ── Build and save report ──
    report = _generate_report(court, players_data, shots, rallies, coach,
                              tactical, fitness, footwork, technical, court_analytics,
                              fps=video_fps, data_quality=data_quality,
                              physics_summary=physics_summary)

    from app.report.generator import _clean_nan
    report = _clean_nan(report)
    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2, default=str))

    # Export stroke_map.json for UI
    stroke_map = _clean_nan({
        "fps": video_fps,
        "duration_seconds": duration,
        "strokes": [
            {"frame": s["frame"], "timestamp": round(s["frame"] / video_fps, 2),
             "stroke_type": s["stroke_type"], "confidence": round(s.get("stroke_confidence", 0.5), 3),
             "player_id": s.get("player_id"), "rally_id": s.get("rally_id"),
             "side": s.get("side", "unknown"), "owner_confident": bool(s.get("owner_confident", False)),
             "owner_source": s.get("owner_source", "unknown"), "owner_reason": s.get("owner_reason", "missing"),
             "aim_alpha_reliable": bool(s.get("aim_alpha_reliable", False)),
             "aim_alpha_route": s.get("aim_alpha_route", "alpha_abstain_quality")}
            for s in shots
        ],
    })
    (output.parent / "stroke_map.json").write_text(json.dumps(stroke_map, indent=2))

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE in {elapsed:.1f}s")
    print(f"  Report saved to: {output}")
    print(f"{'=' * 60}")
    print(f"\n  Summary:")
    print(f"    Players:    {len(players_data.get('players', []))}")
    print(f"    Shots:      {len(shots)}")
    print(f"    Rallies:    {len(rallies)}")
    sd = report.get("shot_distribution", {})
    if sd:
        top3 = sorted(sd.items(), key=lambda x: -x[1])[:3]
        print(f"    Top shots:  {', '.join(f'{s} ({p*100:.0f}%)' for s, p in top3)}")
    if coach["strengths"]:
        print(f"\n  Strengths:")
        for s in coach["strengths"][:3]:
            print(f"    + {s[:70]}")
    if coach["weaknesses"]:
        print(f"\n  Areas to improve:")
        for w in coach["weaknesses"][:3]:
            print(f"    ! {w[:70]}")
    if coach["recommended_drills"]:
        print(f"\n  Suggested drills:")
        for d in coach["recommended_drills"][:3]:
            print(f"    > {d[:70]}")

    return report


def _interpolate_pose_bbox(
    frame_idx: int,
    player_side: str,
    frame_indices: list[int],
    detections_by_frame: dict[int, list[dict]],
    range_limit: int = 10,
) -> list[float] | None:
    """Resolve a pose bbox from detections of the requested side only."""
    before_bbox = None
    after_bbox = None
    before_frame = None
    after_frame = None

    candidate_frames = sorted(set(frame_indices).union(detections_by_frame))
    before_candidates = [
        candidate for candidate in candidate_frames
        if 0 < frame_idx - candidate <= range_limit
    ]
    after_candidates = [
        candidate for candidate in candidate_frames
        if 0 < candidate - frame_idx <= range_limit
    ]

    for candidate in reversed(before_candidates):
        match = next(
            (d for d in detections_by_frame.get(candidate, []) if d.get("side") == player_side),
            None,
        )
        if match is not None:
            before_frame, before_bbox = candidate, np.asarray(match["bbox"], dtype=np.float64)
            break

    for candidate in after_candidates:
        match = next(
            (d for d in detections_by_frame.get(candidate, []) if d.get("side") == player_side),
            None,
        )
        if match is not None:
            after_frame, after_bbox = candidate, np.asarray(match["bbox"], dtype=np.float64)
            break

    if before_bbox is not None and after_bbox is not None:
        fraction = (frame_idx - before_frame) / (after_frame - before_frame)
        return ((1.0 - fraction) * before_bbox + fraction * after_bbox).tolist()
    if before_bbox is not None:
        return before_bbox.tolist()
    if after_bbox is not None:
        return after_bbox.tolist()
    return None


def _process_batch(frames, global_indices, batch_start_offset,
                   tracker, tracknet, pose_estimator, device,
                   all_shuttle, all_det, all_pose, all_player_detections, batch_num=0, total_batches=0,
                   pose_estimator_secondary=None, all_pose_secondary=None,
                   pose_model_name="rtmpose", corners=None,
                   crop_rect=None, far_crop_rect=None):
    """Run ML stages on one batch of frames, append results to accumulators."""
    if not frames:
        return

    tag = f"  Batch {batch_num}/{total_batches}"

    # 1. Player tracking (YOLOv8)
    tqdm.write(f"{tag} | YOLOv8 tracking {len(frames)} frames...")
    batch_det = tracker.track_batch(frames, 0)
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    if corners and len(corners) >= 4:
        court_mid_y = (corners[0][1] + corners[2][1]) / 2
    else:
        h, w = frames[0].shape[:2]
        court_mid_y = h * 0.5
    # Persistent track centroids for joint per-frame assignment across batches
    if not hasattr(_process_batch, "tracks"):
        _process_batch.tracks = [
            {"id": "player_1", "side": "near", "last_center": None},
            {"id": "player_2", "side": "far", "last_center": None},
        ]

    for local_idx, global_idx in enumerate(global_indices):
        dets = batch_det.get(local_idx, [])
        if len(dets) >= 2:
            b0_center_y = (dets[0]["bbox"][1] + dets[0]["bbox"][3]) / 2
            b1_center_y = (dets[1]["bbox"][1] + dets[1]["bbox"][3]) / 2
            if b0_center_y > b1_center_y:
                dets[0]["side"] = "near"
                dets[1]["side"] = "far"
            else:
                dets[0]["side"] = "far"
                dets[1]["side"] = "near"
            for d in dets:
                c = np.array([(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2])
                idx = 0 if d["side"] == "near" else 1
                _process_batch.tracks[idx]["last_center"] = c
        elif len(dets) == 1:
            det = dets[0]
            c = np.array([(det["bbox"][0] + det["bbox"][2]) / 2, (det["bbox"][1] + det["bbox"][3]) / 2])
            near_track = _process_batch.tracks[0]
            far_track = _process_batch.tracks[1]
            if near_track["last_center"] is not None and far_track["last_center"] is not None:
                d_near = np.linalg.norm(c - near_track["last_center"])
                d_far = np.linalg.norm(c - far_track["last_center"])
                det["side"] = "near" if d_near <= d_far else "far"
            else:
                det["side"] = "near" if c[1] > court_mid_y else "far"
            idx = 0 if det["side"] == "near" else 1
            _process_batch.tracks[idx]["last_center"] = c
        for d in dets:
            d["frame"] = global_idx
            all_player_detections.append(d)
            all_det[global_idx] = all_det.get(global_idx, [])
            all_det[global_idx].append(d)

    # 2. Shuttle tracking (TrackNet)
    tqdm.write(f"{tag} | TrackNet shuttle tracking...")
    ow, oh = frames[0].shape[1], frames[0].shape[0]
    net_y = float(np.mean([corner[1] for corner in corners])) if corners and len(corners) >= 4 else None
    shuttle_preds = tracknet.predict_batch(
        frames,
        original_size=(ow, oh),
        crop_rect=crop_rect,
        far_crop_rect=far_crop_rect,
        far_threshold=settings.tracknet_far_heat_threshold,
        net_y=net_y,
    )
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    for local_idx, global_idx in enumerate(global_indices):
        pred_idx = local_idx - 2
        if pred_idx >= 0 and pred_idx < len(shuttle_preds):
            all_shuttle.append({"frame": global_idx, **shuttle_preds[pred_idx]})

    # 3. Pose estimation (RTMPose) — collect crops, then batch
    crop_list = []
    for local_idx, global_idx in enumerate(global_indices):
        frame = frames[local_idx]
        dets_for_frame = all_det.get(global_idx, [])
        if not dets_for_frame:
            for pid, side in [("player_1", "near"), ("player_2", "far")]:
                bbox = _interpolate_pose_bbox(global_idx, side, global_indices, all_det)
                if bbox is not None:
                    crop_list.append((global_idx, pid, bbox, frame))
            continue
        near_det = None
        far_det = None
        for d in dets_for_frame:
            if d.get("side") == "near":
                near_det = d
            elif d.get("side") == "far":
                far_det = d
        if near_det is None and len(dets_for_frame) >= 2:
            cy0 = (dets_for_frame[0]["bbox"][1] + dets_for_frame[0]["bbox"][3]) / 2
            cy1 = (dets_for_frame[1]["bbox"][1] + dets_for_frame[1]["bbox"][3]) / 2
            if cy0 > cy1:
                near_det, far_det = dets_for_frame[0], dets_for_frame[1]
            else:
                near_det, far_det = dets_for_frame[1], dets_for_frame[0]
        elif near_det is None and far_det is None and len(dets_for_frame) == 1:
            near_det = dets_for_frame[0]
        if near_det is None:
            bbox = _interpolate_pose_bbox(global_idx, "near", global_indices, all_det)
            if bbox is not None:
                near_det = {"bbox": bbox}
        if far_det is None:
            bbox = _interpolate_pose_bbox(global_idx, "far", global_indices, all_det)
            if bbox is not None:
                far_det = {"bbox": bbox}
        if near_det:
            crop_list.append((global_idx, "player_1", near_det["bbox"], frame))
        if far_det:
            crop_list.append((global_idx, "player_2", far_det["bbox"], frame))

    tqdm.write(f"{tag} | {pose_model_name} batch estimation ({len(crop_list)} crops)...")
    BATCH_CHUNK = 128
    for crop_chunk_start in range(0, len(crop_list), BATCH_CHUNK):
        chunk = crop_list[crop_chunk_start:crop_chunk_start + BATCH_CHUNK]
        crops = [(c[2], c[3]) for c in chunk]
        kps_batch = pose_estimator.estimate_batch(crops)
        for j, (global_idx, pid, _, _) in enumerate(chunk):
            all_pose.append({"frame": global_idx, "player_id": pid, "keypoints": kps_batch[j].tolist()})

    # Secondary pose estimation (for hybrid mode)
    if pose_estimator_secondary is not None and all_pose_secondary is not None:
        tqdm.write(f"{tag} | RTMPose (secondary) estimation ({len(crop_list)} crops)...")
        for crop_chunk_start in range(0, len(crop_list), BATCH_CHUNK):
            chunk = crop_list[crop_chunk_start:crop_chunk_start + BATCH_CHUNK]
            crops = [(c[2], c[3]) for c in chunk]
            kps_secondary = pose_estimator_secondary.estimate_batch(crops)
            for j, (global_idx, pid, _, _) in enumerate(chunk):
                all_pose_secondary.append({"frame": global_idx, "player_id": pid, "keypoints": kps_secondary[j].tolist()})

    tqdm.write(f"{tag} done | Shuttle: {len(all_shuttle)} | Players: {len(all_player_detections)} | Pose: {len(all_pose)}")
    if device == "cuda":
        import torch
        used_mb = torch.cuda.memory_allocated() / 1024 / 1024
        tqdm.write(f"  GPU memory: {used_mb:.0f} MB allocated")
    del frames, batch_det, shuttle_preds, crop_list
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMCA - Badminton Match Coaching Assistant")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="report.json", help="Output report path")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--pose-model", default="rtmpose", choices=["rtmpose", "mmpose", "hybrid"],
                        help="Pose model: rtmpose (fast), mmpose/hrnet (accurate), or hybrid (MMPose strokes + RTMPose hits)")
    parser.add_argument("--sample-rate", "-s", type=int, default=0,
                        help="Frame sample rate: 0=auto (10fps), 1=every frame, 2=every 2nd, etc.")
    parser.add_argument("--joint-norm", default="bbox", choices=["bbox", "hip_centered"],
                        help="Joint normalization mode: bbox (BST default) or hip_centered (experimental)")
    parser.add_argument("--log", default=None, help="Log file path (writes both console and file)")
    parser.add_argument("--court-corners", default=None,
                        help="Manual court corners as 8 ints: x1,y1,x2,y2,x3,y3,x4,y4 (order: BL,BR,TL,TR)")
    parser.add_argument("--mmaction2", action="store_true", default=False,
                        help="Enable MMAction2 ensemble (PoseC3D or SlowFast) alongside BST")
    parser.add_argument("--mmaction2-mode", default="posec3d",
                        choices=["posec3d", "slowfast", "pytorchvideo"],
                        help="MMAction2 model mode: posec3d (skeleton, default), slowfast (RGB), pytorchvideo (light RGB)")
    parser.add_argument("--mmaction2-weight", type=float, default=0.3,
                        help="Ensemble weight for MMAction2: (1-w)*BST + w*MMAction (default: 0.3)")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    log_file = None
    if args.log:
        log_file = open(args.log, "w")
        import io
        class TeeWriter(io.TextIOBase):
            def __init__(self, file_a, file_b):
                self.file_a = file_a
                self.file_b = file_b
            def write(self, data):
                self.file_a.write(data)
                self.file_b.write(data)
                self.file_a.flush()
                self.file_b.flush()
                return len(data)
            def flush(self):
                self.file_a.flush()
                self.file_b.flush()
        tee = TeeWriter(sys.stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        print(f"Logging to: {args.log}")

    try:
        court_corners = _parse_court_corners_arg(args.court_corners) if args.court_corners else None
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.mmaction2:
        settings.mmaction2_enabled = True
        settings.mmaction2_mode = args.mmaction2_mode
        settings.mmaction2_ensemble_weight = args.mmaction2_weight
        print(f"MMAction2 ensemble enabled: mode={args.mmaction2_mode}, weight={args.mmaction2_weight}")

    # Joint normalization mode
    settings.bst_joint_norm = args.joint_norm
    # Audio is too noisy in Colab (other courts) — disable audio-visual fusion
    settings.audio_hit_enabled = False
    if args.joint_norm != "bbox":
        print(f"Joint normalization mode: {args.joint_norm}")

    run_pipeline(args.video, args.output, args.device, pose_model=args.pose_model,
                 sample_rate=args.sample_rate, court_corners=court_corners)

    if log_file:
        log_file.close()
