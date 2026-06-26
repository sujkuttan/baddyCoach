#!/usr/bin/env python3
"""
BMCA - Badminton Match Coaching Assistant
Self-contained pipeline for Colab/Kaggle GPU execution.

Keeps the GPU ML batch loop (YOLO/TrackNet/RTMPose) for memory efficiency,
then delegates CPU stages to backend pipeline via ArtifactStore.

Usage:
    python pipeline.py video.mp4 --output report.json --device cuda

Requirements:
    pip install torch torchvision ultralytics onnxruntime-gpu opencv-python-headless scipy numpy pyyaml gdown tqdm
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

from app.pipeline.shared.court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT,
    _correct_court_points, compute_homography, image_to_court, HomographySmoother,
)
from app.pipeline.shared.utils import (
    _infer_end_reason, stage_rally_stats,
)
from app.pipeline.shared.core import STROKE_CLASSES, _get_gpu_batch_config
from app.pipeline.shared.models import ensure_model, MODEL_REGISTRY

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
                from app.models.tracknet import InpaintNet
                checkpoint = torch.load(inpaintnet_path, map_location=device, weights_only=False)
                state_dict = checkpoint if isinstance(checkpoint, dict) else {}
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.inpaintnet = InpaintNet()
                self.inpaintnet.load_state_dict(state_dict, strict=False)
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
                    if self.device == "cuda":
                        inpaint_tensor = inpaint_tensor.half()
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

    def predict_batch(self, frames, original_size=None):
        import torch
        if self.model is None or len(frames) < 3:
            return [{"x": 0, "y": 0, "confidence": 0} for _ in frames]

        ow = original_size[0] if original_size else frames[0].shape[1]
        oh = original_size[1] if original_size else frames[0].shape[0]

        preprocessed = np.empty((len(frames), self.input_height, self.input_width, 3), dtype=np.float32)
        for i, f in enumerate(frames):
            r = cv2.resize(f, (self.input_width, self.input_height))
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            preprocessed[i] = r
        del frames

        CHUNK = self._tracknet_chunk
        raw_results = [None for _ in range(len(preprocessed))]

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
                y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
                raw_results[chunk_start + j] = (
                    float(x_idx * ow / self.input_width),
                    float(y_idx * oh / self.input_height),
                    float(hm.max()),
                )
            del tensor, out, heatmaps, batch, windows

        rectified = self._rectify_trajectory(raw_results, ow, oh)
        return [{"x": r[0], "y": r[1], "confidence": r[2]} for r in rectified]


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
        """Detect court with fallback chain: model → proportional.
        
        Returns: list of 4 corners [bl, br, tl, tr] for homography
        """
        # Try model first
        kps = self.detect(frame)
        if kps is not None and len(kps) == 6:
            # Use 4 outer corners only (KP2/KP3 ignored — unreliable):
            # KP0=far-left, KP1=far-right, KP4=near-left, KP5=near-right
            corners = [kps[4], kps[5], kps[0], kps[1]]
            return corners
        
        # Fallback to color+line detection
        return detect_court_from_frame(frame)


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
            results = self.model.track(
                chunk, classes=[0], conf=self.conf,
                verbose=False, persist=True, device=self.device,
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
    """Detect badminton court from a video frame using multi-stage detection.
    
    Stage 1: Color segmentation + line detection (primary)
    Stage 2: HoughLinesP edge detection (fallback 1)
    Stage 3: Returns None to trigger proportional fallback (fallback 2)
    
    Returns list of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    ordered as bottom-left, bottom-right, top-left, top-right.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # ── Stage 1: Color-based court segmentation ──
    # Badminton courts are typically green or blue with white lines
    # Detect court floor using HSV color ranges
    court_mask = None
    
    # Green court (most common)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Blue court
    blue_lower = np.array([100, 40, 40])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Combine court floor colors
    floor_mask = cv2.bitwise_or(green_mask, blue_mask)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (should be the court)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Court should be at least 10% of frame area
        if area > w * h * 0.10:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float64)
                # Order: bottom-left, bottom-right, top-left, top-right (matches compute_homography)
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1).flatten()
                corners = [
                    pts[np.argmax(d)].tolist(),  # bottom-left (largest diff)
                    pts[np.argmax(s)].tolist(),  # bottom-right (largest sum)
                    pts[np.argmin(s)].tolist(),  # top-left (smallest sum)
                    pts[np.argmin(d)].tolist(),  # top-right (smallest diff)
                ]
    
    # ── Stage 2: HoughLinesP edge detection (fallback) ──
    if corners is None:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=w * 0.2, maxLineGap=10)
        
        if lines is not None and len(lines) >= 4:
            h_lines = []
            v_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if abs(angle) < 30:
                    h_lines.append((min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2), length))
                elif abs(abs(angle) - 90) < 30:
                    v_lines.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2), length))
            
            if len(h_lines) >= 2 and len(v_lines) >= 2:
                h_lines.sort(key=lambda l: l[4], reverse=True)
                v_lines.sort(key=lambda l: l[4], reverse=True)
                top_y = min(l[0] for l in h_lines[:4])
                bot_y = max(l[1] for l in h_lines[:4])
                left_x = min(l[0] for l in v_lines[:4])
                right_x = max(l[1] for l in v_lines[:4])
                
                court_w = right_x - left_x
                court_h = bot_y - top_y
                if court_w > w * 0.2 and court_h > h * 0.2 and court_w < w * 0.95 and court_h < h * 0.95:
                    corners = [[left_x, bot_y], [right_x, bot_y], [left_x, top_y], [right_x, top_y]]
    
    # ── Validate corners ──
    if corners is not None:
        # Check aspect ratio (court is ~2.59:1 length:width)
        pts = np.array(corners, dtype=np.float64)
        top_w = np.linalg.norm(pts[2] - pts[3])
        bot_w = np.linalg.norm(pts[0] - pts[1])
        left_h = np.linalg.norm(pts[0] - pts[2])
        right_h = np.linalg.norm(pts[1] - pts[3])
        avg_w = (top_w + bot_w) / 2
        avg_h = (left_h + right_h) / 2
        
        if avg_w > 0 and avg_h > 0:
            aspect = max(avg_w, avg_h) / min(avg_w, avg_h)
            # Badminton court aspect ratio is ~2.20 (13.4/6.10)
            if 1.5 < aspect < 4.0:
                return corners
    
    # ── Stage 3: Return None (proportional fallback) ──
    return None


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
                     data_quality=None):
    """Build the final report dict from all analytics."""
    shot_dist = {}
    for pid, data in tactical.items():
        shot_dist.update(data.get("shot_distribution", {}))
    data_quality = data_quality or {}

    shots_with_ts = []
    for shot_idx, s in enumerate(shots, 1):
        shots_with_ts.append({
            "shot_id": shot_idx,
            "frame": s["frame"],
            "start_ts": round(s["frame"] / fps, 3),
            "stroke_type": s["stroke_type"],
            "confidence": round(s.get("stroke_confidence", 0.5), 3),
            "player_id": s.get("player_id", "player_1"),
            "rally_id": s.get("rally_id"),
        })

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
        "data_quality": {k: v for k, v in data_quality.items()
                         if k != "court_valid" and k != "model_health"},
    }


def run_pipeline(video_path: str, output_path: str, device: str = "cuda", pose_model: str = "rtmpose", sample_rate: int = 0):
    """Run the full BMCA pipeline.

    Keeps the GPU ML batch loop for memory efficiency (YOLO/TrackNet/RTMPose),
    then delegates CPU analytics stages to backend pipeline via ArtifactStore.
    """
    import tempfile
    from app.pipeline.base import StageConfig
    from app.pipeline.hits import HitFrameLocalizationStage
    from app.pipeline.strokes import StrokeClassificationStage
    from app.pipeline.attribution import PlayerAttributionStage
    from app.pipeline.rallies import RallySegmentationStage
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

    detected_corners = None
    detection_method = "none"
    if ret and sample_frame is not None:
        detected_corners = court_kp_detector.detect_with_fallback(sample_frame)
        if detected_corners is not None:
            detection_method = "court_kpRCNN" if court_kp_detector.model is not None else "color+line"

    if detected_corners:
        corners = detected_corners
        print(f"  Detected court ({detection_method}): {corners}")
    else:
        margin_x = int(vid_w * 0.08)
        court_top = int(vid_h * 0.28)
        court_bottom = int(vid_h * 0.72)
        corners = [(margin_x, court_bottom), (vid_w - margin_x, court_bottom),
                   (margin_x, court_top), (vid_w - margin_x, court_top)]
        print(f"  Using proportional corners: {corners}")

    court = stage_court_detection(corners)
    corrected_corners = _correct_court_points(corners)
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
                               pose_model_name=pose_display_name, corners=corners)
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
                       pose_model_name=pose_display_name, corners=corners)
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

    # Build player summary
    players = {"player_1": {"id": "player_1", "side": "near", "detections": []},
               "player_2": {"id": "player_2", "side": "far", "detections": []}}
    for d in all_player_detections:
        side = d.get("side", "near")
        pid = "player_1" if side == "near" else "player_2"
        players[pid]["detections"].append(d)
    players_data = {"players": [{"id": p["id"], "side": p["side"], "detection_count": len(p["detections"]), "detections": p["detections"]} for p in players.values()]}

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
        store.set_parquet("shuttle", shuttle_df)

        pose_df = pd.DataFrame(all_pose)
        store.set_parquet("pose", pose_df)

        if all_pose_secondary:
            store.set("pose_secondary", all_pose_secondary)

        pd.DataFrame(all_player_detections).to_parquet(debug_dir / "player_detections.parquet", index=False)
        pd.DataFrame(all_shuttle).to_parquet(debug_dir / "shuttle.parquet", index=False)
        pd.DataFrame(all_pose).to_parquet(debug_dir / "pose.parquet", index=False)

        # ── CPU stages via backend ──
        print("\n[3/5] Hit frame localization + stroke classification...")
        hits_result = HitFrameLocalizationStage().run(store, config)
        hits = hits_result.metadata.get("hits", [])
        print(f"  Found {len(hits)} hits")
        pd.DataFrame(hits).to_parquet(debug_dir / "hits.parquet", index=False)

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
        from app.config.settings import settings as bst_settings
        bst_settings.joint_velocity_amplification = 2.0
        shots_result = StrokeClassificationStage().run(store, config)
        shots_df = store.get_parquet("shots")
        shots = shots_df.to_dict("records") if shots_df is not None and len(shots_df) > 0 else []
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
        for debug_key in ["debug_bst_outputs", "debug_hit_scores"]:
            df = store.get_parquet(debug_key)
            if df is not None and len(df) > 0:
                df.to_parquet(debug_dir / f"{debug_key}.parquet", index=False)

        # Enrich rallies with winner/end_reason
        for rally in rallies:
            rally_shots_list = [s for s in shots if s.get("rally_id") == rally["rally_id"]]
            rally_shots_list.sort(key=lambda s: s["frame"])
            if rally_shots_list:
                last_shot = rally_shots_list[-1]
                rally["end_reason"] = _infer_end_reason(
                    last_shot.get("stroke_type", "clear"),
                    last_shot.get("stroke_confidence", 0.5),
                )
                last_hitter = last_shot.get("player_id", "player_1")
                if rally["end_reason"] == "winner":
                    rally["winner_player_id"] = last_hitter
                elif rally["end_reason"] in ("forced_error", "unforced_error", "net"):
                    rally["winner_player_id"] = "player_2" if last_hitter == "player_1" else "player_1"
                else:
                    rally["winner_player_id"] = None
            else:
                rally["winner_player_id"] = None
                rally["end_reason"] = "unknown"
            rally["serving_player_id"] = "player_1" if rally["rally_id"] % 2 == 1 else "player_2"
        pd.DataFrame(rallies).to_parquet(debug_dir / "rallies.parquet", index=False)

        # Analytics stages
        CourtPositionAnalyticsStage().run(store, config)
        FootworkAnalyticsStage().run(store, config)
        FitnessAnalyticsStage().run(store, config)
        TacticalAnalyticsStage().run(store, config)
        TechnicalAnalyticsStage().run(store, config)
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
                              fps=video_fps, data_quality=data_quality)

    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2, default=str))

    # Export stroke_map.json for UI
    stroke_map = {
        "fps": video_fps,
        "duration_seconds": duration,
        "strokes": [
            {"frame": s["frame"], "timestamp": round(s["frame"] / video_fps, 2),
             "stroke_type": s["stroke_type"], "confidence": round(s.get("stroke_confidence", 0.5), 3),
             "player_id": s.get("player_id", "player_1"), "rally_id": s.get("rally_id")}
            for s in shots
        ],
    }
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


def _process_batch(frames, global_indices, batch_start_offset,
                   tracker, tracknet, pose_estimator, device,
                   all_shuttle, all_det, all_pose, all_player_detections, batch_num=0, total_batches=0,
                   pose_estimator_secondary=None, all_pose_secondary=None,
                   pose_model_name="rtmpose", corners=None):
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
        elif len(dets) == 1:
            dets[0]["side"] = "near" if (dets[0]["bbox"][1] + dets[0]["bbox"][3]) / 2 > court_mid_y else "far"
        for d in dets:
            d["frame"] = global_idx
            all_player_detections.append(d)
            all_det[global_idx] = all_det.get(global_idx, [])
            all_det[global_idx].append(d)

    # 2. Shuttle tracking (TrackNet)
    tqdm.write(f"{tag} | TrackNet shuttle tracking...")
    ow, oh = frames[0].shape[1], frames[0].shape[0]
    shuttle_preds = tracknet.predict_batch(frames, original_size=(ow, oh))
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
            for pid in ["player_1", "player_2"]:
                best_det = None
                best_dist = float('inf')
                for other_idx in range(max(0, local_idx - 10), min(len(global_indices), local_idx + 10)):
                    other_global = global_indices[other_idx]
                    for d in all_det.get(other_global, []):
                        if d.get("side") == ("near" if pid == "player_1" else "far"):
                            dist = abs(other_idx - local_idx)
                            if dist < best_dist:
                                best_dist = dist
                                best_det = d
                if best_det is None:
                    for other_idx in range(max(0, local_idx - 10), min(len(global_indices), local_idx + 10)):
                        other_global = global_indices[other_idx]
                        for d in all_det.get(other_global, []):
                            dist = abs(other_idx - local_idx)
                            if dist < best_dist:
                                best_dist = dist
                                best_det = d
                if best_det:
                    crop_list.append((global_idx, pid, best_det["bbox"], frame))
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
        elif near_det is None and len(dets_for_frame) == 1:
            near_det = dets_for_frame[0]
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
                        help="Frame sampling interval (0=auto for 10fps, 1=every frame, 2=every 2nd, etc.)")
    parser.add_argument("--log", default=None, help="Log file path (writes both console and file)")
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

    run_pipeline(args.video, args.output, args.device, pose_model=args.pose_model, sample_rate=args.sample_rate)

    if log_file:
        log_file.close()
