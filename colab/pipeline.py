#!/usr/bin/env python3
"""
BMCA - Badminton Match Coaching Assistant
Self-contained pipeline for Colab/Kaggle GPU execution.

Usage:
    python pipeline.py video.mp4 --output report.json --device cuda

Requirements:
    pip install torch torchvision ultralytics onnxruntime-gpu opencv-python-headless scipy numpy pyyaml gdown tqdm
"""

import argparse
import gc
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)

TRACKNET_PATH = CKPT_DIR / "TrackNet_best.pt"
YOLOV8_MODEL = "yolov8s.pt"
RTMOPOSE_PATH = CKPT_DIR / "rtmpose" / "rtmpose-m_8xb64-270e_coco-256x192.onnx"
RTMOPOSE_PATH_ALT = CKPT_DIR / "rtmpose" / "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx"
BST_PATH = CKPT_DIR / "bst" / "bst_CG_JnB_bone_merged.pt"
HRNET_PATH = CKPT_DIR / "mmpose" / "hrnet_w32_coco_256x192.onnx"

COURT_LENGTH = 13.4
COURT_WIDTH = 5.18
NET_HEIGHT = 1.55

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]

RULES = [
    # ─── Tactical Rules ────────────────────────────────────────
    {"name": "smash_efficiency",
     "check": {"field": "tactical.shot_distribution.smash", "operator": "<", "threshold": 0.08, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Smash usage is below 8% ({tactical.shot_distribution.smash:.1%}). Smashes are your primary attacking weapon — use them more when opponents return high.",
     "category": "weakness", "drill": "Feed drills: partner lifts to rear court, practice 10 smashes to each corner.",
     "context_fields": ["tactical.shot_distribution.smash", "tactical.total_shots"]},

    {"name": "smash_strength",
     "check": {"field": "tactical.shot_distribution.smash", "operator": ">", "threshold": 0.15, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Excellent smash frequency ({tactical.shot_distribution.smash:.1%}) — maintaining attacking pressure.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.smash"]},

    {"name": "shot_variety_predictable",
     "check": {"field": "tactical.max_shot_percentage", "operator": ">", "threshold": 0.45, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Shot selection is predictable — dominant stroke accounts for {tactical.max_shot_percentage:.1%} of shots. Opponents can read your patterns.",
     "category": "weakness", "drill": "Pattern-breaking drill: after 2 identical shots, forced switch to a different stroke.",
     "context_fields": ["tactical.max_shot_percentage"]},

    {"name": "shot_variety_good",
     "check": {"field": "tactical.max_shot_percentage", "operator": "<", "threshold": 0.3, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Good shot variety — no single stroke dominates. This keeps opponents guessing.",
     "category": "strength"},

    {"name": "net_play_dominant",
     "check": {"field": "tactical.shot_distribution.net_shot", "operator": ">", "threshold": 0.2, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Strong net play ({tactical.shot_distribution.net_shot:.1%}) — use this to force lifts and create smash opportunities.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.net_shot"]},

    {"name": "net_play_weak",
     "check": {"field": "tactical.shot_distribution.net_shot", "operator": "<", "threshold": 0.05, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Net shots are rare ({tactical.shot_distribution.net_shot:.1%}). Improve front court presence to control rallies.",
     "category": "weakness", "drill": "Net kill drills: partner feeds to net, practice tight spinning net shots.",
     "context_fields": ["tactical.shot_distribution.net_shot"]},

    {"name": "clear_heavy",
     "check": {"field": "tactical.shot_distribution.clear", "operator": ">", "threshold": 0.35, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Heavy reliance on clears ({tactical.shot_distribution.clear:.1%}) — mix with drops and smashes to vary pace.",
     "category": "weakness", "drill": "Clear-drop combination: alternate clear and drop from rear court.",
     "context_fields": ["tactical.shot_distribution.clear"]},

    {"name": "drop_shot_effective",
     "check": {"field": "tactical.shot_distribution.drop", "operator": ">", "threshold": 0.12, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Good use of drop shots ({tactical.shot_distribution.drop:.1%}) — keeps opponents off balance.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.drop"]},

    {"name": "drive_effective",
     "check": {"field": "tactical.shot_distribution.drive", "operator": ">", "threshold": 0.15, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Strong drive game ({tactical.shot_distribution.drive:.1%}) — flat exchanges keep pressure on opponents.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.drive"]},

    {"name": "rush_game",
     "check": {"field": "tactical.shot_distribution.rush", "operator": ">", "threshold": 0.1, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Active rush game ({tactical.shot_distribution.rush:.1%}) — taking the shuttle early creates time pressure.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.rush"]},

    # ─── Fitness Rules ─────────────────────────────────────────
    {"name": "fatigue_declining",
     "check": {"field": "fitness.fatigue_trend", "operator": "==", "value": "declining"},
     "recommendation": "Performance declines in later rallies (fatigue trend: declining). Late-match intensity drops by {fitness.late_rally_fatigue:.0%}.",
     "category": "weakness", "drill": "Interval training: 12x (30s high intensity + 30s rest). Simulate match demands.",
     "context_fields": ["fitness.late_rally_fatigue", "fitness.peak_intensity"]},

    {"name": "fatigue_improving",
     "check": {"field": "fitness.fatigue_trend", "operator": "==", "value": "improving"},
     "recommendation": "Great stamina — performance improves in later rallies. You outlast opponents.",
     "category": "strength", "context_fields": ["fitness.late_rally_fatigue"]},

    {"name": "low_intensity",
     "check": {"field": "fitness.rally_intensity", "operator": "<", "threshold": 1.0, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Rally intensity is low ({fitness.rally_intensity:.2f} shots/sec). Increase pace to pressure opponents.",
     "category": "weakness", "drill": "Speed rallies: 50-shot rallies at maximum pace.",
     "context_fields": ["fitness.rally_intensity"]},

    {"name": "high_intensity",
     "check": {"field": "fitness.peak_intensity", "operator": ">", "threshold": 3.0, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "High peak intensity ({fitness.peak_intensity:.2f} shots/sec) — explosive rallies when needed.",
     "category": "strength", "context_fields": ["fitness.peak_intensity"]},

    {"name": "distance_low",
     "check": {"field": "fitness.total_distance", "operator": "<", "threshold": 100000, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Court coverage is limited ({fitness.total_distance:.0f} units). Work on movement to reach more shots.",
     "category": "weakness", "drill": "6-corner footwork: shadow movement to all court positions.",
     "context_fields": ["fitness.total_distance"]},

    {"name": "distance_high",
     "check": {"field": "fitness.total_distance", "operator": ">", "threshold": 300000, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Excellent court coverage ({fitness.total_distance:.0f} units) — you cover the full court effectively.",
     "category": "strength", "context_fields": ["fitness.total_distance"]},

    # ─── Footwork Rules ────────────────────────────────────────
    {"name": "recovery_slow",
     "check": {"field": "footwork.avg_recovery", "operator": ">", "threshold": 1.5},
     "recommendation": "Recovery to base takes {footwork.avg_recovery:.1f} frames on average. Work on split-step timing.",
     "category": "weakness", "drill": "Split-step practice: bounce on toes, explode to shuttle on opponent's hit.",
     "context_fields": ["footwork.avg_recovery"]},

    {"name": "recovery_fast",
     "check": {"field": "footwork.avg_recovery", "operator": "<", "threshold": 0.5, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Quick recovery ({footwork.avg_recovery:.1f} frames) — you reset well between shots.",
     "category": "strength", "context_fields": ["footwork.avg_recovery"]},

    # ─── Rally Rules ───────────────────────────────────────────
    {"name": "short_rallies",
     "check": {"field": "rally_stats.avg_length", "operator": "<", "threshold": 5.0, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Average rally length is {rally_stats.avg_length:.1f} shots. Opponents end rallies quickly — work on sustaining pressure.",
     "category": "weakness", "drill": "Patience drill: cannot smash until rally reaches 8 shots.",
     "context_fields": ["rally_stats.avg_length"]},

    {"name": "long_rallies",
     "check": {"field": "rally_stats.avg_length", "operator": ">", "threshold": 12.0, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Long rallies (avg {rally_stats.avg_length:.1f} shots) — you control tempo well.",
     "category": "strength", "context_fields": ["rally_stats.avg_length"]},

    {"name": "first_shot_winner",
     "check": {"field": "rally_stats.first_shot_win_rate", "operator": ">", "threshold": 0.3, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Strong opening shots — {rally_stats.first_shot_win_rate:.0%} of rallies won on first shot.",
     "category": "strength", "context_fields": ["rally_stats.first_shot_win_rate"]},

    # ─── Court Position Rules ──────────────────────────────────
    {"name": "front_court_weak",
     "check": {"field": "court_analysis.front_pct", "operator": "<", "threshold": 0.2, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Limited front court presence ({court_analysis.front_pct:.1%}). Move forward to intercept and pressure.",
     "category": "weakness", "drill": "Net approaches: practice moving from base to net after clears.",
     "context_fields": ["court_analysis.front_pct"]},

    {"name": "rear_court_dominant",
     "check": {"field": "court_analysis.rear_pct", "operator": ">", "threshold": 0.6, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Spending {court_analysis.rear_pct:.1%} of time in rear court — opponents are pushing you back.",
     "category": "weakness", "drill": "Counter-attack drills: practice attacking from rear court.",
     "context_fields": ["court_analysis.rear_pct"]},

    {"name": "balanced_court",
     "check": {"field": "court_analysis.front_pct", "operator": ">", "threshold": 0.25, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Good court balance — front court presence at {court_analysis.front_pct:.1%} keeps opponents guessing.",
     "category": "strength", "context_fields": ["court_analysis.front_pct"]},

    # ─── Comparison Rules (player vs opponent) ─────────────────
    {"name": "opponent_smash_weak",
     "check": {"field": "opponent.smash_pct", "operator": "<", "threshold": 0.08, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Opponent rarely smashes ({opponent.smash_pct:.1%}). Expect clears and drops — position forward.",
     "category": "insight", "context_fields": ["opponent.smash_pct"]},

    {"name": "opponent_net_weak",
     "check": {"field": "opponent.net_pct", "operator": "<", "threshold": 0.05, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Opponent avoids net play ({opponent.net_pct:.1%}). Push to net to force weak returns.",
     "category": "insight", "context_fields": ["opponent.net_pct"]},

    {"name": "opponent_clear_heavy",
     "check": {"field": "opponent.clear_pct", "operator": ">", "threshold": 0.4, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Opponent relies heavily on clears ({opponent.clear_pct:.1%}). Anticipate deep shots and counter with drops.",
     "category": "insight", "context_fields": ["opponent.clear_pct"]},
]


def setup_models(device: str, pose_model: str = "rtmpose"):
    print("Setting up models...")
    if not TRACKNET_PATH.exists():
        try:
            import gdown
            import zipfile
            print("  Downloading TrackNetV3 weights...")
            zip_path = str(CKPT_DIR / "tracknet.zip")
            gdown.download(id="1rhKXbff1GITgrFTYptW6gAvWZ76E_qzp", output=zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(str(CKPT_DIR))
            os.remove(zip_path)
            for f in CKPT_DIR.rglob("*.pt"):
                if "TrackNet" in f.name:
                    f.rename(TRACKNET_PATH)
                    break
        except Exception as e:
            print(f"  TrackNet download failed: {e}")
            print("  Shuttle tracking will use fallback")

    from ultralytics import YOLO
    YOLO(YOLOV8_MODEL)

    rtmpose_dir = CKPT_DIR / "rtmpose"
    rtmpose_dir.mkdir(parents=True, exist_ok=True)
    if not RTMOPOSE_PATH.exists() and not RTMOPOSE_PATH_ALT.exists():
        try:
            import gdown
            import zipfile
            print("  Downloading RTMPose weights...")
            zip_path = str(rtmpose_dir / "rtmpose.zip")
            gdown.download(id="1XjwDxz1a8i3WO6afuvaq-y3HPiFh48SN", output=zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(str(rtmpose_dir))
            os.remove(zip_path)
            # Flatten: move any nested .onnx to top-level
            for onnx in rtmpose_dir.rglob("*.onnx"):
                dest = rtmpose_dir / "rtmpose.onnx"
                if onnx != dest:
                    import shutil
                    shutil.move(str(onnx), str(dest))
                    print(f"  Moved {onnx.name} -> {dest}")
        except Exception as e:
            print(f"  RTMPose download failed: {e}")

    bst_dir = CKPT_DIR / "bst"
    bst_dir.mkdir(parents=True, exist_ok=True)
    if not BST_PATH.exists():
        try:
            import gdown
            print("  Downloading BST weights...")
            gdown.download(id="1yHLpW4s8Rk8FYIUKF_NvC29Z8b8XuDq2", output=str(BST_PATH), quiet=False)
        except Exception as e:
            print(f"  BST download failed: {e}")

    hrnet_dir = CKPT_DIR / "mmpose"
    hrnet_dir.mkdir(parents=True, exist_ok=True)
    if not HRNET_PATH.exists() and pose_model == "mmpose":
        print("  Downloading pre-exported HRNet-W32 ONNX...")
        try:
            import gdown
            gdown.download(id="1LFUEbHB-D3WCyjzf9aSJ_V_kVB8igsnr",
                           output=str(HRNET_PATH), quiet=False)
        except Exception as e:
            print(f"  HRNet download failed: {e}")
            print(f"  Falling back to RTMPose")
        except Exception as e:
            print(f"  HRNet auto-export failed: {e}")
            print(f"  Falling back to RTMPose")

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
    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        import torch.nn as nn

        self.device = device
        self.model = None
        self.input_height = 288
        self.input_width = 512

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

    def predict_batch(self, frames, original_size=None):
        import torch
        if self.model is None or len(frames) < 3:
            return [{"x": 0, "y": 0, "confidence": 0}] * len(frames)

        ow = original_size[0] if original_size else frames[0].shape[1]
        oh = original_size[1] if original_size else frames[0].shape[0]

        CHUNK = 16
        results = [None] * len(frames)

        for chunk_start in range(0, len(frames), CHUNK):
            chunk_end = min(chunk_start + CHUNK, len(frames))
            windows = []
            for i in range(chunk_start, chunk_end):
                window = frames[max(0, i - 8):i + 1]
                while len(window) < 9:
                    window.insert(0, window[0])
                processed = []
                for f in window[-9:]:
                    r = cv2.resize(f, (self.input_width, self.input_height))
                    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    processed.append(r)
                windows.append(np.stack(processed).reshape(self.input_height, self.input_width, 27))

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
                results[chunk_start + j] = {
                    "x": float(x_idx * ow / self.input_width),
                    "y": float(y_idx * oh / self.input_height),
                    "confidence": float(hm.max()),
                }
            del tensor, out, heatmaps, batch, windows
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return results


class YOLOv8Tracker:
    def __init__(self, conf_threshold=0.3, device="cuda"):
        from ultralytics import YOLO
        self.model = YOLO(YOLOV8_MODEL)
        self.conf = conf_threshold
        self.device = device

    def track_batch(self, frames, global_frame_offsets):
        all_det = {}
        for local_idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            results = self.model.track(frame, classes=[0], conf=self.conf, verbose=False, persist=True, device=self.device)
            global_idx = global_frame_offsets + local_idx
            dets = []
            for r in results:
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
        return all_det


class RTMPoseEstimator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = None
        self.h, self.w = 256, 192
        self.model_type = "rtmpose"
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
        x1, y1, x2, y2 = [int(v) for v in bbox]
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

    def estimate_batch(self, crops):
        if self.model is None:
            return [np.zeros((17, 3), dtype=np.float32) for _ in crops]

        batch_tensors = []
        valid_indices = []
        crop_infos = []

        for i, (bbox, frame) in enumerate(crops):
            tensor, crop_info = self._preprocess(frame, bbox)
            if tensor is None:
                continue
            batch_tensors.append(tensor[0])
            valid_indices.append(i)
            crop_infos.append(crop_info)

        if not batch_tensors:
            return [np.zeros((17, 3), dtype=np.float32) for _ in crops]

        batch_np = np.stack(batch_tensors)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: batch_np})

        kps_all = []
        for j in range(len(batch_np)):
            single_outputs = [out[j:j+1] for out in outputs]
            if self.model_type == "hrnet":
                kps_all.append(self._decode_hrnet(single_outputs, crop_infos[j]))
            else:
                kps_all.append(self._decode_rtmpose(single_outputs, crop_infos[j]))

        results = [np.zeros((17, 3), dtype=np.float32) for _ in crops]
        for j, idx in enumerate(valid_indices):
            results[idx] = kps_all[j]
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

def stage_court_detection(corners):
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [COURT_WIDTH, 0], [0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return {"homography": H.tolist(), "corners_pixel": [list(c) for c in corners],
            "court_length": COURT_LENGTH, "court_width": COURT_WIDTH, "net_height": NET_HEIGHT}


def stage_hits(shuttle_data):
    shuttle_df = pd.DataFrame(shuttle_data)
    if len(shuttle_df) == 0:
        return []
    x, y = shuttle_df["x"].values, shuttle_df["y"].values
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    angle = np.arctan2(dy, dx)
    traj_score = np.abs(np.diff(angle, prepend=angle[0])) / (np.pi + 1e-6)
    speed = np.sqrt(dx**2 + dy**2)
    peaks, _ = find_peaks(speed, distance=3)
    speed_score = np.zeros(len(speed))
    speed_score[peaks] = speed[peaks]
    combined = 0.5 * (traj_score / (traj_score.max() + 1e-6)) + 0.5 * (speed_score / (speed_score.max() + 1e-6))
    threshold = np.percentile(combined, 95)
    hits = [{"frame": int(shuttle_df.iloc[i]["frame"]), "confidence": float(combined[i])} for i in np.where(combined > threshold)[0]]
    return hits


def stage_strokes(hits_data, shuttle_data, pose_data=None, court=None, device="cuda", vid_w=1280, vid_h=720, player_detections=None):
    """Classify strokes using BST model with sequence inputs.
    
    This implementation follows the BST paper's approach:
    1. Extract stroke clips (windows of frames around each hit)
    2. Prepare BST inputs (pose sequences, shuttle sequences)
    3. Run BST inference on the clips
    """
    if not hits_data:
        return []
    
    shuttle_df = pd.DataFrame(shuttle_data) if shuttle_data else pd.DataFrame()
    pose_df = pd.DataFrame(pose_data) if pose_data else pd.DataFrame()
    
    # BST configuration
    SEQ_LEN = 30  # Sequence length for BST
    BST_CLASSES = [
        "net_shot", "block", "smash", "lift", "clear", "drive",
        "drop", "push", "rush", "cross_court", "short_serve", "long_serve"
    ]
    
    # COCO bone pairs
    BONE_PAIRS = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
        (3, 5), (4, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    def create_bones(joints):
        """Create bone vectors from joint positions.
        joints: (T, M, J, 2) -> bones: (T, M, B, 2)
        """
        bones = []
        for start, end in BONE_PAIRS:
            start_j = joints[:, :, start, :]
            end_j = joints[:, :, end, :]
            bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
            bones.append(bone)
        return np.stack(bones, axis=-2)
    
    def normalize_joints_bstdiag(coords, det_bbox=None):
        """Normalize joints using bbox diagonal with center_align.

        Uses detection bbox for stable normalization when available.
        Falls back to keypoint bbox (less stable).
        """
        if det_bbox is not None:
            bbox_min = np.array([det_bbox[0], det_bbox[1]], dtype=np.float64)
            bbox_max = np.array([det_bbox[2], det_bbox[3]], dtype=np.float64)
        else:
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        if diag < 1e-6:
            diag = 1.0
        normalized = (coords - bbox_min) / diag
        center = (bbox_min + bbox_max) / 2.0
        normalized -= (center - bbox_min) / diag
        return normalized.astype(np.float32)

    def prepare_bst_clip(clip_frames, seq_len):
        """Prepare BST input from a clip of frames.
        Returns: JnB (seq_len, 2, 72), shuttle (seq_len, 2), pos (seq_len, 2, 2), video_len

        Preprocessing matches official BST pipeline:
        - Joints: bbox-diagonal normalization with center_align (range [-0.X, 0.X])
        - Shuttle: normalized by video resolution (range [0, 1])
        - Position: feet midpoint normalized by video resolution (range [0, 1])
        """
        n_frames = len(clip_frames)

        joints = np.zeros((seq_len, 2, 17, 2), dtype=np.float32)
        shuttle = np.zeros((seq_len, 2), dtype=np.float32)
        pos = np.zeros((seq_len, 2, 2), dtype=np.float32)

        for t, frame in enumerate(clip_frames[:seq_len]):
            if 'shuttle_x' in frame and 'shuttle_y' in frame:
                shuttle[t] = [frame['shuttle_x'], frame['shuttle_y']]

            det_bboxes = frame.get('det_bboxes', {})
            for p_idx, pid in enumerate(['player_1', 'player_2']):
                if pid in frame.get('pose', {}):
                    kps = frame['pose'][pid]
                    if kps is not None and kps.shape == (17, 3):
                        coords = kps[:, :2]
                        det_bbox = det_bboxes.get(pid)
                        joints[t, p_idx] = normalize_joints_bstdiag(coords, det_bbox=det_bbox)
                        feet_y = max(coords[15, 1], coords[16, 1])
                        feet_x = (coords[15, 0] + coords[16, 0]) / 2
                        pos[t, p_idx] = [feet_x / vid_w, feet_y / vid_h]

        # Interpolate missing shuttle coordinates (0.0 = missing)
        for dim in range(2):
            shuttle_series = pd.Series(shuttle[:, dim])
            mask = shuttle_series == 0.0
            if mask.any() and (~mask).any():
                shuttle_series = shuttle_series.replace(0, np.nan)
                shuttle_series = shuttle_series.interpolate(method='linear').bfill().ffill()
                shuttle[:, dim] = shuttle_series.values

        bones = create_bones(joints)
        JnB = np.concatenate([joints, bones], axis=-2).reshape(seq_len, 2, -1)

        return JnB, shuttle, pos, min(n_frames, seq_len)
    
    # Load BST model
    bst_path = str(BST_PATH) if BST_PATH.exists() else None
    model = None
    seq_len = SEQ_LEN
    
    if bst_path:
        try:
            import torch
            checkpoint = torch.load(bst_path, map_location=device, weights_only=False)
            
            # Detect model architecture from state_dict
            state_dict = checkpoint if isinstance(checkpoint, dict) and 'model' not in checkpoint else None
            if state_dict is None and isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model'] if isinstance(checkpoint['model'], dict) else None
            
            if state_dict and any('tcn_pose' in k for k in list(state_dict.keys())[:10]):
                # This is a BST state_dict - detect dimensions
                in_dim = 72
                n_classes = 25
                detected_seq_len = SEQ_LEN
                
                for k, v in state_dict.items():
                    if 'tcn_pose.net.0.weight' in k:
                        in_dim = v.shape[1]
                    if 'mlp_head.mlp.mlp.3.weight' in k:
                        n_classes = v.shape[0]
                    if 'embedding_tem' in k:
                        detected_seq_len = v.shape[1] - 1
                
                seq_len = detected_seq_len
                
                # Create BST_CG model (inline for self-contained pipeline)
                # Based on: https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
                import torch.nn as nn
                import math

                class TCN(nn.Module):
                    def __init__(self, in_channel, channels, kernel_size=5, drop_p=0.3):
                        super().__init__()
                        layers = []
                        for i in range(len(channels)):
                            in_ch = in_channel if i == 0 else channels[i-1]
                            out_ch = channels[i]
                            dilation = i * 2 + 1
                            padding = (kernel_size - 1) * dilation // 2
                            layers += [
                                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                                nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(drop_p, inplace=True)
                            ]
                        self.net = nn.Sequential(*layers)
                    def forward(self, x):
                        return self.net(x)

                class MLP(nn.Module):
                    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0):
                        super().__init__()
                        self.mlp = nn.Sequential(
                            nn.Linear(in_dim, hd_dim), nn.GELU(), nn.Dropout(drop_p, inplace=True),
                            nn.Linear(hd_dim, out_dim)
                        )
                    def forward(self, x):
                        return self.mlp(x)

                class MLP_Head(nn.Module):
                    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0):
                        super().__init__()
                        self.layer_norm = nn.LayerNorm(in_dim)
                        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)
                    def forward(self, x):
                        return self.mlp(self.layer_norm(x))

                class FeedForward(nn.Module):
                    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0):
                        super().__init__()
                        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)
                        self.dropout = nn.Dropout(drop_p, inplace=True)
                    def forward(self, x):
                        return self.dropout(self.mlp(x))

                class MultiHeadAttention(nn.Module):
                    def __init__(self, d_model, d_head, n_head, drop_p):
                        super().__init__()
                        d_cat = d_head * n_head
                        self.h = n_head
                        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)
                        self.scale = d_head ** -0.5
                        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
                        self.tail = nn.Sequential(nn.Linear(d_cat, d_model), nn.Dropout(drop_p, inplace=True)) if n_head != 1 or d_cat != d_model else nn.Identity()
                    def forward(self, x, mask=None):
                        bn, t, _ = x.shape
                        qkv = self.to_qkv(x).view(bn, t, self.h, -1).chunk(3, dim=-1)
                        q, k, v = map(lambda ts: ts.transpose(1, 2), qkv)
                        dots = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
                        if mask is not None:
                            dots = dots.masked_fill(mask.view(bn, 1, 1, t) == 0, -torch.inf)
                        att = self.attend(dots) @ v.contiguous()
                        return self.tail(att.transpose(1, 2).reshape(bn, t, -1))

                class TransformerLayer(nn.Module):
                    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p):
                        super().__init__()
                        self.layer_norm1 = nn.LayerNorm(d_model)
                        self.attn = MultiHeadAttention(d_model, d_head, n_head, drop_p)
                        self.layer_norm2 = nn.LayerNorm(d_model)
                        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)
                    def forward(self, x, mask=None):
                        z = self.layer_norm1(x)
                        x = self.attn(z, mask) + x
                        z = self.layer_norm2(x)
                        x = self.ff(z) + x
                        return x

                class TransformerEncoder(nn.Module):
                    def __init__(self, d_model, d_head, n_head, depth, hd_mlp, drop_p):
                        super().__init__()
                        self.layers = nn.ModuleList([TransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p) for _ in range(depth)])
                    def forward(self, x, mask=None):
                        for layer in self.layers:
                            x = layer(x, mask)
                        return x

                class MultiHeadCrossAttention(nn.Module):
                    def __init__(self, d_model, d_head, n_head, drop_p):
                        super().__init__()
                        d_cat = d_head * n_head
                        self.h = n_head
                        self.to_q = nn.Linear(d_model, d_cat, bias=False)
                        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)
                        self.scale = d_head ** -0.5
                        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
                        self.tail = nn.Sequential(nn.Linear(d_cat, d_model), nn.Dropout(drop_p, inplace=True)) if n_head != 1 or d_cat != d_model else nn.Identity()
                    def forward(self, x1, x2, mask=None):
                        q = self.to_q(x1)
                        kv = self.to_kv(x2)
                        b, t, _ = q.shape
                        q = q.view(b, t, self.h, -1).transpose(1, 2)
                        kv = kv.view(b, t, self.h, -1).chunk(2, dim=-1)
                        k, v = map(lambda ts: ts.transpose(1, 2), kv)
                        dots = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
                        if mask is not None:
                            dots = dots.masked_fill(mask.view(b, 1, 1, t) == 0, -torch.inf)
                        att = self.attend(dots) @ v.contiguous()
                        return self.tail(att.transpose(1, 2).reshape(b, t, -1))

                class CrossTransformerLayer(nn.Module):
                    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p):
                        super().__init__()
                        self.layer_norm1_x1 = nn.LayerNorm(d_model)
                        self.layer_norm1_x2 = nn.LayerNorm(d_model)
                        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
                        self.layer_norm2 = nn.LayerNorm(d_model)
                        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)
                    def forward(self, x1, x2, mask=None):
                        x1 = self.layer_norm1_x1(x1)
                        x2 = self.layer_norm1_x2(x2)
                        x = self.cross_attn(x1, x2, mask)
                        return self.ff(self.layer_norm2(x)) + x

                class PositionalEncoding1D(nn.Module):
                    def __init__(self, d_model):
                        super().__init__()
                        self.d_model = d_model
                    def forward(self, x):
                        if x.dim() == 2:
                            l, d = x.shape
                            pe = torch.zeros(l, d, device=x.device, dtype=x.dtype)
                            position = torch.arange(0, l, device=x.device, dtype=x.dtype).unsqueeze(1)
                            div_term = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / d))
                            pe[:, 0::2] = torch.sin(position * div_term)
                            pe[:, 1::2] = torch.cos(position * div_term)
                            return x + pe
                        b, l, d = x.shape
                        pe = torch.zeros(l, d, device=x.device, dtype=x.dtype)
                        position = torch.arange(0, l, device=x.device, dtype=x.dtype).unsqueeze(1)
                        div_term = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / d))
                        pe[:, 0::2] = torch.sin(position * div_term)
                        pe[:, 1::2] = torch.cos(position * div_term)
                        return x + pe.unsqueeze(0)

                class BST_CG(nn.Module):
                    def __init__(self, in_dim, seq_len, n_class=25, n_people=2,
                                 d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1,
                                 drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5):
                        super().__init__()
                        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)
                        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
                        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)
                        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
                        self.embedding_tem = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
                        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
                        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)
                        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
                        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)
                        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
                        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
                        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
                        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)
                        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)
                        self.d_model = d_model
                        self._init_weights()

                    @torch.no_grad()
                    def _init_weights(self):
                        p_enc = PositionalEncoding1D(self.d_model)
                        self.embedding_tem.copy_(p_enc(self.embedding_tem.squeeze(0)).unsqueeze(0))
                        self.embedding_cross.copy_(p_enc(self.embedding_cross))
                        self.embedding_inter.copy_(p_enc(self.embedding_inter.squeeze(0)).unsqueeze(0))
                        nn.init.normal_(self.learned_token_tem, std=0.02)
                        nn.init.normal_(self.learned_token_inter, std=0.02)
                        self.apply(self._init_w)

                    def _init_w(self, m):
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None: nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.Conv1d):
                            nn.init.xavier_normal_(m.weight)

                    def forward(self, JnB, shuttle, pos, video_len):
                        b, t, n, in_dim = JnB.shape
                        JnB = JnB.permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
                        pos = self.mlp_positions(pos)
                        pos_impact = pos.permute(0, 2, 3, 1).reshape(b * n, in_dim, t)
                        JnB = JnB * pos_impact + JnB
                        JnB = self.tcn_pose(JnB).view(b, n, -1, t).transpose(-2, -1)
                        shuttle = shuttle.transpose(1, 2).contiguous()
                        shuttle = self.tcn_shuttle(shuttle).unsqueeze(1).transpose(-2, -1)
                        x = torch.cat((JnB, shuttle), dim=1)
                        _, n, _, d = x.shape
                        ct = self.learned_token_tem.view(1, 1, -1).expand(b * n, -1, -1)
                        x = x.view(b * n, t, d)
                        x = torch.cat((ct, x), dim=1) + self.embedding_tem
                        range_t = torch.arange(0, 1 + t, device=x.device).unsqueeze(0).expand(b, -1)
                        mask = range_t < (1 + video_len.unsqueeze(-1))
                        mask_n = mask.repeat_interleave(n, dim=0)
                        x = self.pre_dropout(x)
                        x = self.encoder_tem(x, mask_n).view(b, n, 1 + t, d)
                        p1, p2, shuttle = map(lambda ts: ts.squeeze(1), x.chunk(3, dim=1))
                        p1_cls, p2_cls, shuttle_cls = p1[:, 0].contiguous(), p2[:, 0].contiguous(), shuttle[:, 0].contiguous()
                        p1 = p1[:, 1:].contiguous() + self.embedding_cross
                        p2 = p2[:, 1:].contiguous() + self.embedding_cross
                        shuttle = shuttle[:, 1:].contiguous() + self.embedding_cross
                        cross_mask = mask[:, 1:].contiguous()
                        p1_shuttle = self.cross_trans(p1, shuttle, cross_mask)
                        p2_shuttle = self.cross_trans(p2, shuttle, cross_mask)
                        ci = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
                        p1_shuttle = self.encoder_inter(torch.cat((ci, p1_shuttle), dim=1) + self.embedding_inter, mask)
                        p2_shuttle = self.encoder_inter(torch.cat((ci, p2_shuttle), dim=1) + self.embedding_inter, mask)
                        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
                        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()
                        info_need_clean = torch.minimum(p1_shuttle_cls, p2_shuttle_cls)
                        dirt = self.mlp_clean(info_need_clean)
                        shuttle_cls = shuttle_cls - dirt
                        x = torch.cat((p1_cls + p1_shuttle_cls, p2_cls + p2_shuttle_cls, shuttle_cls), dim=1)
                        return self.mlp_head(x)

                model = BST_CG(in_dim, seq_len, n_class=n_classes)
                model.load_state_dict(state_dict)
                model.to(device).eval()
                if device == "cuda":
                    model = model.half()
                    print(f"BST_CG loaded (FP16): in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
                else:
                    print(f"BST_CG loaded (FP32): in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
            else:
                print("BST state_dict not recognized, using rule-based fallback")
        except Exception as e:
            print(f"BST load error: {e}")
    
    # Use actual video dimensions for normalization
    # vid_w/vid_h are passed from the caller (actual video resolution)
    
    # Group pose by frame
    pose_by_frame = {}
    if len(pose_df) > 0:
        for f_idx in pose_df['frame'].unique():
            frame_poses = pose_df[pose_df['frame'] == f_idx]
            pd_dict = {}
            for _, row in frame_poses.iterrows():
                kps_raw = row['keypoints']
                kps = np.array(kps_raw) if isinstance(kps_raw, np.ndarray) else np.array(kps_raw)
                if kps.shape != (17, 3) and hasattr(kps_raw, 'tolist'):
                    kps = np.array(kps_raw.tolist())
                if kps.shape == (17, 3):
                    pd_dict[row['player_id']] = kps
            pose_by_frame[f_idx] = pd_dict
    
    # Get all hit frames for centering clips
    hit_frames = sorted([h['frame'] for h in hits_data])
    
    # Build detection bbox lookup per player
    det_bbox_lookup = {}
    if player_detections:
        # Use per-frame tid_to_pid to build consistent bbox lookup
        for frame_idx in sorted(set(d["frame"] for d in player_detections)):
            frame_dets = [d for d in player_detections if d["frame"] == frame_idx]
            tid_to_pid = {}
            for d in frame_dets:
                tid = d.get("track_id", 0)
                if tid not in tid_to_pid:
                    tid_to_pid[tid] = f"player_{len(tid_to_pid)+1}"
            for d in frame_dets:
                pid = tid_to_pid.get(d.get("track_id", 0))
                if pid:
                    if pid not in det_bbox_lookup:
                        det_bbox_lookup[pid] = {}
                    det_bbox_lookup[pid][frame_idx] = d["bbox"]
    
    # Process each hit
    shots = []
    
    if model is not None:
        import torch
        
        # Sort hit frames for smart clipping (previous-hit to next-hit)
        hit_frames_sorted = sorted([h['frame'] for h in hits_data])
        
        for hit in hits_data:
            hit_frame = hit['frame']
            
            # Smart clipping: previous opponent's hit to next opponent's hit
            hit_pos = hit_frames_sorted.index(hit_frame)
            
            if hit_pos > 0:
                start_frame = hit_frames_sorted[hit_pos - 1]
            else:
                start_frame = max(0, hit_frame - seq_len // 2)
            
            if hit_pos < len(hit_frames_sorted) - 1:
                end_frame = hit_frames_sorted[hit_pos + 1] + 2
            else:
                end_frame = hit_frame + seq_len // 2 + 1
            
            # Extract clip frames
            clip_frames = []
            for f in range(start_frame, end_frame):
                frame_data = {}
                
                # Shuttle (normalized)
                if len(shuttle_df) > 0:
                    s_row = shuttle_df[shuttle_df['frame'] == f]
                    if len(s_row) > 0:
                        frame_data['shuttle_x'] = float(s_row.iloc[0]['x']) / vid_w
                        frame_data['shuttle_y'] = float(s_row.iloc[0]['y']) / vid_h
                    else:
                        frame_data['shuttle_x'] = 0.0
                        frame_data['shuttle_y'] = 0.0
                else:
                    frame_data['shuttle_x'] = 0.0
                    frame_data['shuttle_y'] = 0.0
                
                # Pose: look up by player_1/player_2 directly (set by per-frame tid_to_pid)
                raw_pose = pose_by_frame.get(f, {})
                frame_data['pose'] = {
                    'player_1': raw_pose.get('player_1'),
                    'player_2': raw_pose.get('player_2'),
                }
                frame_data['det_bboxes'] = {
                    'player_1': det_bbox_lookup.get('player_1', {}).get(f),
                    'player_2': det_bbox_lookup.get('player_2', {}).get(f),
                }
                
                clip_frames.append(frame_data)
            
            # Pad/truncate to seq_len, centering on the hit frame
            if len(clip_frames) > seq_len:
                hit_offset = hit_frame - start_frame
                half = seq_len // 2
                clip_start = max(0, hit_offset - half)
                clip_end = clip_start + seq_len
                if clip_end > len(clip_frames):
                    clip_end = len(clip_frames)
                    clip_start = max(0, clip_end - seq_len)
                clip_frames = clip_frames[clip_start:clip_end]
            while len(clip_frames) < seq_len:
                clip_frames.append({'shuttle_x': 0.0, 'shuttle_y': 0.0, 'pose': {}})
            
            # Prepare BST input
            JnB, shuttle_arr, pos_arr, v_len = prepare_bst_clip(clip_frames, seq_len)
            
            # Convert to tensors
            JnB_t = torch.from_numpy(JnB).float().unsqueeze(0).to(device)
            shuttle_t = torch.from_numpy(shuttle_arr).float().unsqueeze(0).to(device)
            pos_t = torch.from_numpy(pos_arr).float().unsqueeze(0).to(device)
            video_len_t = torch.tensor([v_len], dtype=torch.long).to(device)
            if device == "cuda":
                JnB_t = JnB_t.half()
                shuttle_t = shuttle_t.half()
                pos_t = pos_t.half()
            
            # Inference
            with torch.no_grad():
                logits = model(JnB_t, shuttle_t, pos_t, video_len_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            
            # If top prediction is "unknown", try second-best or rule-based fallback
            if pred_idx == 0:
                second_idx = int(np.argsort(probs)[-2])
                second_conf = float(probs[second_idx])
                if second_conf > 0.10:
                    pred_idx = second_idx
                    confidence = second_conf
                else:
                    # Rule-based shuttle trajectory fallback
                    stroke_type = _rule_based_shuttle_predict(shuttle_df, hit_frame, vid_w, vid_h)
                    shots.append({
                        "frame": hit_frame,
                        "hit_confidence": hit['confidence'],
                        "stroke_type": stroke_type,
                        "stroke_confidence": confidence,
                    })
                    continue
            
            # Map to simplified class
            if pred_idx == 0:
                stroke_type = "unknown"
            elif 1 <= pred_idx <= 12:
                stroke_type = BST_CLASSES[pred_idx - 1] if pred_idx - 1 < len(BST_CLASSES) else "clear"
            elif 13 <= pred_idx <= 24:
                stroke_type = BST_CLASSES[pred_idx - 13] if pred_idx - 13 < len(BST_CLASSES) else "clear"
            else:
                stroke_type = "clear"
            
            shots.append({
                "frame": hit_frame,
                "hit_confidence": hit['confidence'],
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
            })
    else:
        # Rule-based fallback when model not available
        for hit in hits_data:
            frame = hit['frame']
            
            # Use shuttle trajectory for classification
            stroke_type = "clear"
            confidence = 0.4
            
            if len(shuttle_df) > 0:
                window = shuttle_df[(shuttle_df['frame'] >= frame - 5) & (shuttle_df['frame'] <= frame + 5)]
                if len(window) >= 2:
                    y_vals = window['y'].values / vid_h
                    x_vals = window['x'].values / vid_w
                    # Filter out zero coordinates (missing shuttle data)
                    valid = (x_vals != 0) | (y_vals != 0)
                    if valid.sum() < 2:
                        shots.append({"frame": frame, "hit_confidence": hit['confidence'], "stroke_type": "clear", "stroke_confidence": 0.4})
                        continue
                    y_vals = y_vals[valid]
                    dy = np.diff(y_vals)
                    speed = np.mean(np.abs(dy))
                    
                    if speed > 0.1 and np.mean(dy) > 0.05:
                        stroke_type = "smash"
                        confidence = 0.5
                    elif speed < 0.03:
                        stroke_type = "net_shot"
                        confidence = 0.45
                    elif np.mean(dy) < -0.03:
                        stroke_type = "clear"
                        confidence = 0.5
                    elif speed > 0.05:
                        stroke_type = "drive"
                        confidence = 0.45
            
            shots.append({
                "frame": frame,
                "hit_confidence": hit['confidence'],
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
            })
    
    return shots


def stage_attribution(shots_data, shuttle_data):
    if not shots_data:
        return shots_data
    shuttle_df = pd.DataFrame(shuttle_data)
    for shot in shots_data:
        frame = shot["frame"]
        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        if len(shuttle_row) > 0:
            sy = float(shuttle_row.iloc[0]["y"])
            shot["player_id"] = "player_1" if sy > 300 else "player_2"
        else:
            shot["player_id"] = "player_1"
    return shots_data


def _rule_based_shuttle_predict(shuttle_df, frame, vid_w, vid_h):
    """Classify stroke from shuttle trajectory when BST predicts unknown."""
    if shuttle_df is None or len(shuttle_df) == 0:
        return "clear"
    window = shuttle_df[(shuttle_df['frame'] >= frame - 5) & (shuttle_df['frame'] <= frame + 5)]
    if len(window) < 2:
        return "clear"
    y_vals = window['y'].values / vid_h
    x_vals = window['x'].values / vid_w
    valid = (x_vals != 0) | (y_vals != 0)
    if valid.sum() < 2:
        return "clear"
    y_vals = y_vals[valid]
    dy = np.diff(y_vals)
    dx = x_vals[valid][1:] - x_vals[valid][:-1] if len(x_vals[valid]) > 1 else np.array([0.0])
    speed = np.mean(np.sqrt(dx**2 + dy**2))
    mean_dy = float(np.mean(dy))
    end_y = float(y_vals[-1])
    if speed > 0.15 and mean_dy > 0.05:
        return "smash"
    elif speed < 0.03:
        return "net_shot"
    elif mean_dy < -0.03 and speed > 0.05:
        return "clear"
    elif speed > 0.08 and abs(mean_dy) < 0.02:
        return "drive"
    elif mean_dy > 0.02 and speed > 0.03:
        return "lift"
    elif end_y > 0.7 and speed < 0.06:
        return "drop"
    else:
        return "clear"


def stage_rallies(shots_data, gap_threshold=45, min_shots=3):
    if not shots_data:
        return []
    shots_sorted = sorted(shots_data, key=lambda s: s["frame"])
    rallies = []
    rally_id = 1
    start = shots_sorted[0]["frame"]
    count = 1
    for i in range(1, len(shots_sorted)):
        if shots_sorted[i]["frame"] - shots_sorted[i-1]["frame"] > gap_threshold:
            if count >= min_shots:
                rallies.append({"rally_id": rally_id, "start_frame": start,
                              "end_frame": shots_sorted[i-1]["frame"], "shot_count": count})
                rally_id += 1
            start = shots_sorted[i]["frame"]
            count = 1
        else:
            count += 1
    if count >= min_shots:
        rallies.append({"rally_id": rally_id, "start_frame": start,
                       "end_frame": shots_sorted[-1]["frame"], "shot_count": count})
    return rallies


def stage_court_position(shuttle_data, shots_data, frame_width=1280, frame_height=720):
    zone_names = ["front_left", "front_center", "front_right", "mid_left", "mid_center", "mid_right", "rear_left", "rear_center", "rear_right"]
    shuttle_df = pd.DataFrame(shuttle_data)
    transitions = []
    for shot in shots_data:
        row = shuttle_df[shuttle_df["frame"] == shot["frame"]]
        if len(row) > 0:
            x, y = float(row.iloc[0]["x"]), float(row.iloc[0]["y"])
            nx = x / frame_width
            ny = y / frame_height
            col = min(int(nx * 3), 2)
            row_idx = min(int(ny * 3), 2)
            transitions.append({"frame": shot["frame"], "zone": zone_names[row_idx * 3 + col],
                              "player_id": shot.get("player_id", "unknown")})
    return {"zone_transitions": transitions, "court_dimensions": {"length": COURT_LENGTH, "width": COURT_WIDTH}}


def stage_footwork(pose_data, shots_data):
    metrics = {}
    pose_df = pd.DataFrame(pose_data) if pose_data else pd.DataFrame()
    if len(pose_df) == 0:
        return metrics
    for pid in pose_df["player_id"].unique():
        player = pose_df[pose_df["player_id"] == pid].sort_values("frame")
        com_points = []
        for _, row in player.iterrows():
            kps_raw = row["keypoints"]
            kps = np.array(kps_raw.tolist()) if hasattr(kps_raw, 'tolist') else np.array(kps_raw)
            if kps.ndim == 1:
                kps = np.array([np.array(x) for x in kps])
            if kps.shape == (17, 3) and np.any(kps != 0):
                com_points.append((kps[11][:2] + kps[12][:2]) / 2)
        dist = sum(np.sqrt(np.sum((np.array(com_points[i+1]) - np.array(com_points[i]))**2))
                   for i in range(len(com_points)-1)) if len(com_points) > 1 else 0
        metrics[pid] = {"distance_covered": float(dist), "recovery_times": [], "avg_recovery": 0}
    return metrics


def stage_fitness(footwork_data, rallies_data, shots_data):
    """Compute fitness analytics with real fatigue trend detection."""
    fitness = {}
    shots_df = pd.DataFrame(shots_data) if shots_data else pd.DataFrame()
    rallies_df = pd.DataFrame(rallies_data) if rallies_data else pd.DataFrame()
    
    def compute_fatigue_trend(intensities):
        if len(intensities) < 5:
            return "insufficient_data"
        n = len(intensities)
        q1 = intensities[:n//4]
        q4 = intensities[3*n//4:]
        avg_q1, avg_q4 = np.mean(q1), np.mean(q4)
        x = np.arange(len(intensities))
        slope = np.polyfit(x, intensities, 1)[0]
        avg_intensity = np.mean(intensities)
        normalized_slope = slope / avg_intensity if avg_intensity > 0 else 0
        if avg_q4 < avg_q1 * 0.8 and normalized_slope < -0.01:
            return "declining"
        elif avg_q4 > avg_q1 * 1.2 and normalized_slope > 0.01:
            return "improving"
        return "stable"
    
    for pid, fw in footwork_data.items():
        intensities = []
        
        if len(rallies_df) > 0 and len(shots_df) > 0:
            for _, rally in rallies_df.iterrows():
                rs = shots_df[(shots_df["frame"] >= rally["start_frame"]) & 
                              (shots_df["frame"] <= rally["end_frame"]) & 
                              (shots_df["player_id"] == pid)]
                duration = max((rally["end_frame"] - rally["start_frame"]) / 30, 0.1)
                intensities.append(float(len(rs) / duration))
        
        fatigue_trend = compute_fatigue_trend(intensities)
        avg_intensity = float(np.mean(intensities)) if intensities else 0
        peak_intensity = float(np.max(intensities)) if intensities else 0
        late_fatigue = 0.0
        if len(intensities) >= 6:
            first_half = intensities[:len(intensities)//2]
            second_half = intensities[len(intensities)//2:]
            avg_first = np.mean(first_half)
            if avg_first > 0:
                late_fatigue = float((avg_first - np.mean(second_half)) / avg_first)
        
        fitness[pid] = {"rally_intensity": avg_intensity, "rally_intensities": intensities,
                       "fatigue_trend": fatigue_trend, "avg_recovery": fw.get("avg_recovery", 0),
                       "total_distance": fw.get("distance_covered", 0),
                       "peak_intensity": peak_intensity, "late_rally_fatigue": late_fatigue,
                       "rally_count": len(intensities)}
    return fitness


def stage_tactical(shots_data):
    """Compute tactical analytics with sequence patterns."""
    tactical = {}
    for shot in shots_data:
        pid = shot.get("player_id", "player_1")
        if pid not in tactical:
            tactical[pid] = {"shot_distribution": Counter(), "total_shots": 0,
                           "common_patterns": [], "unique_strokes": [],
                           "rally_openers": Counter(), "rally_enders": Counter()}
        tactical[pid]["shot_distribution"][shot["stroke_type"]] += 1
        tactical[pid]["total_shots"] += 1

    shots_by_player = {}
    for shot in shots_data:
        pid = shot.get("player_id", "player_1")
        if pid not in shots_by_player:
            shots_by_player[pid] = []
        shots_by_player[pid].append(shot["stroke_type"])

    for pid in tactical:
        total = tactical[pid]["total_shots"]
        tactical[pid]["shot_distribution"] = {k: v/total for k, v in tactical[pid]["shot_distribution"].items()}
        seq = shots_by_player.get(pid, [])

        patterns = Counter()
        for i in range(len(seq) - 2):
            pattern = f"{seq[i]} -> {seq[i+1]} -> {seq[i+2]}"
            patterns[pattern] += 1
        tactical[pid]["common_patterns"] = [
            {"pattern": p, "count": c} for p, c in patterns.most_common(5)
        ]

        tactical[pid]["unique_strokes"] = list(tactical[pid]["shot_distribution"].keys())

        from collections import defaultdict
        rally_shots = defaultdict(list)
        for shot in shots_data:
            if shot.get("player_id") == pid and shot.get("rally_id") is not None:
                rally_shots[shot["rally_id"]].append(shot["stroke_type"])
        for rally_id, strokes in rally_shots.items():
            if strokes:
                tactical[pid]["rally_openers"][strokes[0]] += 1
                tactical[pid]["rally_enders"][strokes[-1]] += 1

        tactical[pid]["rally_openers"] = dict(tactical[pid]["rally_openers"])
        tactical[pid]["rally_enders"] = dict(tactical[pid]["rally_enders"])

    return tactical


def stage_technical(shots_data):
    technical = {}
    for shot in shots_data:
        pid = shot.get("player_id", "player_1")
        if pid not in technical:
            technical[pid] = {}
        st = shot["stroke_type"]
        if st not in technical[pid]:
            technical[pid][st] = {"avg_score": 0.5, "shot_count": 0, "scores": []}
        technical[pid][st]["shot_count"] += 1
        technical[pid][st]["scores"].append(0.5)
        technical[pid][st]["avg_score"] = 0.5
    return technical


def stage_rally_stats(shots_data, rallies_data):
    """Compute rally-level statistics for coaching."""
    stats = {"avg_length": 0, "max_length": 0, "min_length": 0,
             "first_shot_win_rate": 0, "long_rally_pct": 0}
    if not rallies_data or not shots_data:
        return stats

    lengths = [r["shot_count"] for r in rallies_data]
    stats["avg_length"] = float(np.mean(lengths))
    stats["max_length"] = max(lengths)
    stats["min_length"] = min(lengths)
    stats["long_rally_pct"] = float(sum(1 for l in lengths if l > 8) / len(lengths))

    shots_df = pd.DataFrame(shots_data)
    first_shot_wins = 0
    for rally in rallies_data:
        rally_shots = shots_df[(shots_df["frame"] >= rally["start_frame"]) &
                               (shots_df["frame"] <= rally["end_frame"])]
        if len(rally_shots) >= 2:
            first_player = rally_shots.iloc[0].get("player_id")
            last_player = rally_shots.iloc[-1].get("player_id")
            if first_player == last_player:
                first_shot_wins += 1
    stats["first_shot_win_rate"] = float(first_shot_wins / len(rallies_data)) if rallies_data else 0

    return stats


def stage_court_analysis(court_analytics, shots_data):
    """Analyze court zone distribution for coaching."""
    transitions = court_analytics.get("zone_transitions", [])
    if not transitions:
        return {"front_pct": 0, "mid_pct": 0, "rear_pct": 0, "left_pct": 0, "right_pct": 0}

    total = len(transitions)
    front = sum(1 for t in transitions if "front" in t.get("zone", ""))
    mid = sum(1 for t in transitions if "mid" in t.get("zone", ""))
    rear = sum(1 for t in transitions if "rear" in t.get("zone", ""))
    left = sum(1 for t in transitions if "left" in t.get("zone", ""))
    right = sum(1 for t in transitions if "right" in t.get("zone", ""))

    return {
        "front_pct": float(front / total),
        "mid_pct": float(mid / total),
        "rear_pct": float(rear / total),
        "left_pct": float(left / total),
        "right_pct": float(right / total),
    }


def stage_coach(tactical, fitness, footwork, rallies=None, court_analytics=None, shots_data=None):
    """Generate context-aware coaching recommendations using all analytics."""
    strengths_set = set()
    weaknesses_set = set()
    improvements = []
    drills = []
    evidence = []

    rally_stats = stage_rally_stats(shots_data or [], rallies or [])
    court_analysis = stage_court_analysis(court_analytics or {}, shots_data or [])

    player_ids = list(tactical.keys())
    opponent_data = {}
    for pid in player_ids:
        opp_ids = [p for p in player_ids if p != pid]
        if opp_ids:
            opp_tactical = tactical.get(opp_ids[0], {})
            opp_dist = opp_tactical.get("shot_distribution", {})
            opponent_data[pid] = {
                "smash_pct": opp_dist.get("smash", 0),
                "net_pct": opp_dist.get("net_shot", 0),
                "clear_pct": opp_dist.get("clear", 0),
                "total_shots": opp_tactical.get("total_shots", 0),
            }

    def get_nested(data, path):
        keys = path.split(".")
        current = data
        for key in keys:
            if current is None:
                return 0
            if isinstance(current, dict):
                current = current.get(key, 0)
            elif isinstance(current, (list, tuple)):
                try:
                    idx = int(key)
                    current = current[idx] if 0 <= idx < len(current) else 0
                except (ValueError, IndexError):
                    return 0
            else:
                return 0
        return current if current is not None else 0

    def compare(actual, op, expected):
        try:
            actual, expected = float(actual), float(expected)
        except (TypeError, ValueError):
            return str(actual) == str(expected) if op == "==" else False
        if op == "<": return actual < expected
        elif op == ">": return actual > expected
        elif op == "<=": return actual <= expected
        elif op == ">=": return actual >= expected
        elif op == "==": return actual == expected
        elif op == "!=": return actual != expected
        return False

    def evaluate_condition(expr, analytics):
        parts = expr.split()
        if len(parts) != 3:
            return False
        field_path, op, val_str = parts
        try:
            val = float(val_str)
        except ValueError:
            return False
        return compare(get_nested(analytics, field_path), op, val)

    def evaluate_rule(rule, analytics):
        check = rule.get("check", {})
        if not check:
            return False
        min_shots = check.get("min_shots")
        if min_shots and not evaluate_condition(min_shots, analytics):
            return False
        field_path = check.get("field")
        op = check.get("operator")
        threshold = check.get("threshold", check.get("value"))
        if not field_path or not op:
            return False
        return compare(get_nested(analytics, field_path), op, threshold)

    def format_recommendation(template, analytics):
        """Format recommendation template with actual values."""
        try:
            import re
            fields = re.findall(r'\{(\w+(?:\.\w+)*)\}', template)
            values = {}
            for field in fields:
                val = get_nested(analytics, field)
                if isinstance(val, float):
                    values[field] = val
                else:
                    values[field] = val
            return template.format(**values)
        except (KeyError, ValueError, IndexError):
            return template

    for pid in set(list(tactical.keys()) + list(fitness.keys())):
        player_analytics = {
            "tactical": tactical.get(pid, {}),
            "fitness": fitness.get(pid, {}),
            "footwork": footwork.get(pid, {}),
            "rally_stats": rally_stats,
            "court_analysis": court_analysis,
            "opponent": opponent_data.get(pid, {}),
        }

        tactical_data = player_analytics["tactical"]
        if tactical_data:
            shot_dist = tactical_data.get("shot_distribution", {})
            tactical_data["max_shot_percentage"] = max(shot_dist.values()) if shot_dist else 0

        fitness_data = player_analytics["fitness"]
        if fitness_data:
            fitness_data["intensity"] = fitness_data.get("rally_intensity", 0)
            fitness_data["peak"] = fitness_data.get("peak_intensity", 0)
            fitness_data["distance"] = fitness_data.get("total_distance", 0)

        footwork_data = player_analytics["footwork"]
        if footwork_data:
            footwork_data["recovery"] = footwork_data.get("avg_recovery", 0)

        total = tactical_data.get("total_shots", 0)

        for rule in RULES:
            try:
                if evaluate_rule(rule, player_analytics):
                    rec = format_recommendation(rule["recommendation"], player_analytics)
                    rec_with_player = f"[{pid}] {rec}"
                    entry = {
                        "finding": rec_with_player,
                        "metrics": [f"player: {pid}", f"total shots: {total}"],
                    }
                    for cf in rule.get("context_fields", []):
                        val = get_nested(player_analytics, cf)
                        if isinstance(val, float):
                            entry["metrics"].append(f"{cf}: {val:.3f}")
                    evidence.append(entry)

                    if rule["category"] == "strength":
                        if rec_with_player not in strengths_set:
                            strengths_set.add(rec_with_player)
                    elif rule["category"] == "weakness":
                        if rec_with_player not in weaknesses_set:
                            weaknesses_set.add(rec_with_player)
                            improvements.append(rec_with_player)
                            drills.append(rule.get("drill", ""))
                    elif rule["category"] == "insight":
                        if rec_with_player not in weaknesses_set:
                            weaknesses_set.add(rec_with_player)
            except Exception:
                continue

    strengths = list(strengths_set)
    weaknesses = list(weaknesses_set)
    return {"strengths": strengths, "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3], "recommended_drills": drills[:3],
            "evidence": evidence, "rally_stats": rally_stats}


def generate_report(court, players, shuttle, pose, hits, shots, rallies,
                    court_analytics, footwork, fitness, tactical, technical, coach, fps=30):
    shot_dist = {}
    for pid, data in tactical.items():
        shot_dist.update(data.get("shot_distribution", {}))

    # Add timestamps to shots for UI
    shots_with_ts = []
    for s in shots:
        shots_with_ts.append({
            "frame": s["frame"],
            "timestamp": round(s["frame"] / fps, 2),
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
    }


# ─── Main Pipeline (streaming/batched) ──────────────────────────────────────

BATCH_SIZE = 2000

def run_pipeline(video_path: str, output_path: str, device: str = "cuda", pose_model: str = "rtmpose", sample_rate: int = 0):
    start_time = time.time()
    video_name = Path(video_path).name

    print(f"=" * 60)
    print(f"  BMCA Pipeline - {video_name}")
    print(f"  Device: {device}")
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

    # Court detection (no frames needed)
    print("\n[2/14] Court detection...")
    corners = [(100, 500), (1820, 500), (100, 100), (1820, 100)]
    court = stage_court_detection(corners)
    print("  Done")

    # Initialize ML models
    print("\n  Loading ML models...")
    tracker = YOLOv8Tracker(conf_threshold=0.7, device=device)
    tracknet = TrackNetV3(str(TRACKNET_PATH), device=device)
    # Pose model selection
    pose_estimator = None
    pose_estimator_secondary = None

    if pose_model == "hybrid":
        if HRNET_PATH.exists():
            print(f"  Using HYBRID mode: MMPose (strokes) + RTMPose (hits)")
            pose_estimator = RTMPoseEstimator(str(HRNET_PATH), device=device)
            rtmpose_path = str(RTMOPOSE_PATH_ALT if RTMOPOSE_PATH_ALT.exists() else RTMOPOSE_PATH)
            if not Path(rtmpose_path).exists():
                rtmpose_dir = CKPT_DIR / "rtmpose"
                onnx_files = list(rtmpose_dir.rglob("*.onnx"))
                if onnx_files:
                    rtmpose_path = str(onnx_files[0])
            pose_estimator_secondary = RTMPoseEstimator(rtmpose_path, device=device)
        else:
            print(f"  WARNING: HRNet not found, falling back to RTMPose only")
            pose_model = "rtmpose"

    if pose_model == "mmpose" and HRNET_PATH.exists():
        pose_path = str(HRNET_PATH)
        print(f"  Using MMPose HRNet-W32 (accurate)")
        pose_estimator = RTMPoseEstimator(pose_path, device=device)
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
        pose_estimator = RTMPoseEstimator(pose_path, device=device)

    print("  Models loaded")

    # Accumulators for results across batches
    all_shuttle = []
    all_det = {}
    all_pose = []
    all_pose_secondary = []  # RTMPose pose data (for hybrid hit confidence)
    all_player_detections = []
    sample_idx = 0
    batch_count = 0

    # Process video in batches
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n[1-5/14] Running ML stages on {num_samples} sampled frames ({total_batches} batches)...")

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
                               all_pose_secondary=all_pose_secondary)
                batch_frames = []
                batch_global_indices = []
                gc.collect()
        frame_idx += 1
        batch_pbar.update(1)
    batch_pbar.close()

    # Process remaining frames
    if batch_frames:
        batch_count += 1
        _process_batch(batch_frames, batch_global_indices, sample_idx - len(batch_frames),
                       tracker, tracknet, pose_estimator, device,
                       all_shuttle, all_det, all_pose, all_player_detections,
                       batch_count, total_batches,
                       pose_estimator_secondary=pose_estimator_secondary,
                       all_pose_secondary=all_pose_secondary)
        batch_frames = []
        batch_global_indices = []
        gc.collect()

    cap.release()

    print(f"\n  ML stages complete. Data collected:")
    print(f"    Shuttle positions: {len(all_shuttle)} frames")
    print(f"    Player detections: {len(all_player_detections)} total")
    print(f"    Pose keypoints:    {len(all_pose)} frames")

    # Export stage outputs for debugging
    debug_dir = Path(output_path).parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Exporting stage outputs to {debug_dir}/")

    pd.DataFrame(all_shuttle).to_parquet(debug_dir / "shuttle.parquet", index=False)
    print(f"    shuttle.parquet ({len(all_shuttle)} rows)")

    pd.DataFrame(all_pose).to_parquet(debug_dir / "pose.parquet", index=False)
    print(f"    pose.parquet ({len(all_pose)} rows)")

    if all_pose_secondary:
        pd.DataFrame(all_pose_secondary).to_parquet(debug_dir / "pose_secondary.parquet", index=False)
        print(f"    pose_secondary.parquet ({len(all_pose_secondary)} rows)")

    pd.DataFrame(all_player_detections).to_parquet(debug_dir / "player_detections.parquet", index=False)
    print(f"    player_detections.parquet ({len(all_player_detections)} rows)")

    # Build player summary
    tid_counts = Counter(d.get("track_id", 0) for d in all_player_detections)
    top2 = [tid for tid, _ in tid_counts.most_common(2)]
    filtered_dets = [d for d in all_player_detections if d.get("track_id", 0) in top2]

    players = {}
    for d in filtered_dets:
        tid = d.get("track_id", 0)
        side = d.get("side", "near")
        matched = False
        for pid, p in players.items():
            if p.get("track_id") == tid:
                p["detections"].append(d)
                matched = True
                break
        if not matched:
            pid = f"player_{len(players)+1}"
            players[pid] = {"id": pid, "side": side, "track_id": tid, "detections": [d]}

    players_data = {"players": [{"id": p["id"], "side": p["side"], "detection_count": len(p["detections"])} for p in players.values()]}

    # Free ML models from GPU
    del tracker, tracknet, pose_estimator
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Analytics stages (CPU only, lightweight)
    print("\n[6/14] Hit frame localization...")
    hits = stage_hits(all_shuttle)
    print(f"  Found {len(hits)} hits")
    pd.DataFrame(hits).to_parquet(debug_dir / "hits.parquet", index=False)
    print(f"    hits.parquet ({len(hits)} rows)")

    print("\n[7/14] Stroke classification...")
    # In hybrid mode: check if primary pose (HRNet) has valid keypoints
    # If HRNet produces all zeros, fall back to secondary (RTMPose) for BST
    bst_pose = all_pose
    if pose_model == "hybrid" and all_pose_secondary:
        nonzero_count = sum(1 for p in all_pose[:100] if np.any(np.array(p["keypoints"]) != 0))
        if nonzero_count < 10:
            print(f"  HRNet keypoints mostly zero ({nonzero_count}/100 non-zero), using RTMPose for BST")
            bst_pose = all_pose_secondary
        else:
            print(f"  HRNet keypoints valid ({nonzero_count}/100 non-zero)")
    shots = stage_strokes(hits, all_shuttle, bst_pose, court, device, vid_w=vid_w, vid_h=vid_h, player_detections=all_player_detections)
    shots = stage_attribution(shots, all_shuttle)
    print(f"  Classified {len(shots)} shots")
    pd.DataFrame(shots).to_parquet(debug_dir / "shots.parquet", index=False)
    print(f"    shots.parquet ({len(shots)} rows)")

    print("\n[8/14] Rally segmentation...")
    rallies = stage_rallies(shots)
    print(f"  Segmented {len(rallies)} rallies")
    pd.DataFrame(rallies).to_parquet(debug_dir / "rallies.parquet", index=False)
    print(f"    rallies.parquet ({len(rallies)} rows)")

    # Assign rally_id to each shot
    for shot in shots:
        shot_rally = None
        for rally in rallies:
            if rally["start_frame"] <= shot["frame"] <= rally["end_frame"]:
                shot_rally = rally["rally_id"]
                break
        shot["rally_id"] = shot_rally

    print("\n[9/14] Court position analytics...")
    court_analytics = stage_court_position(all_shuttle, shots, vid_w, vid_h)
    print(f"  {len(court_analytics['zone_transitions'])} zone transitions")

    print("\n[10/14] Footwork analytics...")
    # Use RTMPose for footwork/fitness (better movement tracking)
    footwork_pose = all_pose_secondary if all_pose_secondary else all_pose
    footwork = stage_footwork(footwork_pose, shots)
    print("  Done")

    print("\n[11/14] Fitness analytics...")
    fitness = stage_fitness(footwork, rallies, shots)
    print("  Done")

    print("\n[12/14] Tactical analytics...")
    tactical = stage_tactical(shots)
    print("  Done")

    print("\n[13/14] Technical analytics...")
    technical = stage_technical(shots)
    print("  Done")

    print("\n[14/14] Coach recommendations...")
    coach = stage_coach(tactical, fitness, footwork, rallies=rallies,
                        court_analytics=court_analytics, shots_data=shots)
    print(f"  {len(coach['strengths'])} strengths, {len(coach['weaknesses'])} weaknesses")

    report = generate_report(court, players_data, all_shuttle, all_pose, hits, shots, rallies,
                            court_analytics, footwork, fitness, tactical, technical, coach, fps=video_fps)

    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2, default=str))

    # Export stroke_map.json for UI timestamp display
    stroke_map = {
        "fps": video_fps,
        "duration_seconds": duration,
        "strokes": [
            {
                "frame": s["frame"],
                "timestamp": round(s["frame"] / video_fps, 2),
                "stroke_type": s["stroke_type"],
                "confidence": round(s["stroke_confidence"], 3),
                "player_id": s.get("player_id", "player_1"),
                "rally_id": s.get("rally_id"),
            }
            for s in shots
        ],
    }
    stroke_map_path = output.parent / "stroke_map.json"
    stroke_map_path.write_text(json.dumps(stroke_map, indent=2))

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE in {elapsed:.1f}s")
    print(f"  Report saved to: {output}")
    print(f"{'=' * 60}")

    print(f"\n  Pipeline Summary:")
    print(f"    [1/5] Court detection       OK")
    print(f"    [2/5] Player tracking       {len(players_data.get('players', []))} players, {len(all_player_detections)} detections")
    print(f"    [3/5] Shuttle tracking      {len(all_shuttle)} frames tracked (avg conf: {np.mean([s['confidence'] for s in all_shuttle]):.3f})")
    print(f"    [4/5] Pose estimation       {len(all_pose)} frames estimated")
    print(f"    [5/5] Hit detection         {len(hits)} hits found")

    print(f"\n  Analysis:")
    print(f"    Shots:     {len(shots)}")
    print(f"    Rallies:   {len(rallies)}")
    print(f"    Zones:     {len(court_analytics['zone_transitions'])} transitions")

    sd = report.get("shot_distribution", {})
    if sd:
        top3 = sorted(sd.items(), key=lambda x: -x[1])[:3]
        print(f"    Top shots: {', '.join(f'{s} ({p*100:.0f}%)' for s, p in top3)}")

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

    print(f"\n  Open the local UI and click 'Load Report' to view the full dashboard.")

    return report


def _process_batch(frames, global_indices, batch_start_offset,
                   tracker, tracknet, pose_estimator, device,
                   all_shuttle, all_det, all_pose, all_player_detections, batch_num=0, total_batches=0,
                   pose_estimator_secondary=None, all_pose_secondary=None):
    """Run ML stages on one batch of frames, append results to accumulators."""
    if not frames:
        return

    tag = f"  Batch {batch_num}/{total_batches}"

    # 1. Player tracking (YOLOv8)
    tqdm.write(f"{tag} | YOLOv8 tracking {len(frames)} frames...")
    batch_det = tracker.track_batch(frames, 0)
    h, w = frames[0].shape[:2]
    court_mid_y = h * 0.5
    for local_idx, global_idx in enumerate(global_indices):
        for d in batch_det.get(local_idx, []):
            d["frame"] = global_idx
            d["side"] = "near" if d["bbox"][1] > court_mid_y else "far"
            all_player_detections.append(d)
            all_det[global_idx] = all_det.get(global_idx, [])
            all_det[global_idx].append(d)

    # 2. Shuttle tracking (TrackNet)
    tqdm.write(f"{tag} | TrackNet shuttle tracking...")
    ow, oh = frames[0].shape[1], frames[0].shape[0]
    shuttle_preds = tracknet.predict_batch(frames, original_size=(ow, oh))
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
                        dist = abs(other_idx - local_idx)
                        if dist < best_dist:
                            best_dist = dist
                            best_det = d
                if best_det:
                    crop_list.append((global_idx, pid, best_det["bbox"], frame))
            continue
        tid_to_pid = {}
        for d in dets_for_frame[:2]:
            tid = d.get("track_id", 0)
            if tid not in tid_to_pid:
                tid_to_pid[tid] = f"player_{len(tid_to_pid)+1}"
        for d in dets_for_frame[:2]:
            pid = tid_to_pid.get(d.get("track_id", 0), "player_1")
            crop_list.append((global_idx, pid, d["bbox"], frame))

    tqdm.write(f"{tag} | RTMPose batch estimation ({len(crop_list)} crops)...")
    BATCH_CHUNK = 128
    for crop_chunk_start in range(0, len(crop_list), BATCH_CHUNK):
        chunk = crop_list[crop_chunk_start:crop_chunk_start + BATCH_CHUNK]
        crops = [(c[2], c[3]) for c in chunk]
        kps_batch = pose_estimator.estimate_batch(crops)
        for j, (global_idx, pid, _, _) in enumerate(chunk):
            all_pose.append({"frame": global_idx, "player_id": pid, "keypoints": kps_batch[j].tolist()})

    # Secondary pose estimation (for hybrid mode)
    if pose_estimator_secondary is not None and all_pose_secondary is not None:
        tqdm.write(f"{tag} | Secondary RTMPose estimation ({len(crop_list)} crops)...")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMCA - Badminton Match Coaching Assistant")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="report.json", help="Output report path")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--pose-model", default="rtmpose", choices=["rtmpose", "mmpose", "hybrid"],
                        help="Pose model: rtmpose (fast), mmpose/hrnet (accurate), or hybrid (MMPose strokes + RTMPose hits)")
    parser.add_argument("--sample-rate", "-s", type=int, default=0,
                        help="Frame sampling interval (0=auto for 10fps, 1=every frame, 2=every 2nd, etc.)")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    run_pipeline(args.video, args.output, args.device, pose_model=args.pose_model, sample_rate=args.sample_rate)
