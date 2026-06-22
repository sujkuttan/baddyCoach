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
from collections import deque
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

# ─── Shared module imports ──────────────────────────────────────────────────
# Add backend to path for shared modules (unification with backend pipeline)
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.pipeline.shared.court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, COURT_MODEL, COURT_ASPECT_RATIO,
    _detect_court_color_line, _correct_court_points, _validate_court_geometry,
    compute_homography, image_to_court, HomographySmoother, make_undistorter,
    foot_midpoint_from_pose, foot_point_from_bbox,
)
from app.pipeline.shared.utils import (
    _rule_based_shuttle_predict, _evaluate_shot,
    _infer_end_reason, _is_rally_ending_shot,
    stage_rally_stats,
)
from app.pipeline.shared.core import STROKE_CLASSES, _get_gpu_batch_config

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)

TRACKNET_PATH = CKPT_DIR / "TrackNet_best.pt"
YOLOV8_MODEL = "yolov8s.pt"
RTMOPOSE_PATH = CKPT_DIR / "rtmpose" / "rtmpose-m_8xb64-270e_coco-256x192.onnx"
COURT_KP_MODEL_PATH = CKPT_DIR / "court_kpRCNN.pth"
RTMOPOSE_PATH_ALT = CKPT_DIR / "rtmpose" / "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx"
BST_PATH = CKPT_DIR / "bst" / "bst_CG_JnB_bone_merged.pt"
HRNET_PATH = CKPT_DIR / "mmpose" / "hrnet_w32_coco_256x192.onnx"


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
     "recommendation": "Court coverage is limited ({fitness.total_distance:.0f} m). Work on movement to reach more shots.",
     "category": "weakness", "drill": "6-corner footwork: shadow movement to all court positions.",
     "context_fields": ["fitness.total_distance"]},

    {"name": "distance_high",
     "check": {"field": "fitness.total_distance", "operator": ">", "threshold": 300000, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Excellent court coverage ({fitness.total_distance:.0f} m) — you cover the full court effectively.",
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

    # Court keypoint model (SoloShuttlePose)
    if not COURT_KP_MODEL_PATH.exists():
        try:
            import gdown
            print("  Downloading court keypoint model...")
            gdown.download(id="1FGKyX-NudJGXvfsmKEpjiQYojDAWONdy", output=str(COURT_KP_MODEL_PATH), quiet=False)
        except Exception as e:
            print(f"  Court KP model download failed: {e}")

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
    def __init__(self, model_path: str, device: str = "cuda", chunk_size: int = 16):
        import torch
        import torch.nn as nn

        self.device = device
        self.model = None
        self.input_height = 288
        self.input_width = 512
        self._tracknet_chunk = chunk_size

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
        results = [{"x": 0, "y": 0, "confidence": 0} for _ in range(len(preprocessed))]

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
                results[chunk_start + j] = {
                    "x": float(x_idx * ow / self.input_width),
                    "y": float(y_idx * oh / self.input_height),
                    "confidence": float(hm.max()),
                }
            del tensor, out, heatmaps, batch, windows
        return results


class CourtKeypointDetector:
    """Court keypoint detector using Keypoint R-CNN (SoloShuttlePose).
    
    Detects 6 court keypoints per frame:
    - KP 0: far-left corner   (court: 0, 0)
    - KP 1: far-right corner  (court: 0, 5.18)
    - KP 2: net-left          (court: 6.7, 0)   — unreliable, often duplicates KP0
    - KP 3: net-right         (court: 6.7, 5.18)
    - KP 4: near-left corner  (court: 13.4, 0)
    - KP 5: near-right corner (court: 13.4, 5.18)
    
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
    def __init__(self, conf_threshold=0.3, device="cuda", yolo_chunk=200, yolo_batch=16):
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
        return all_det


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
            # Badminton court aspect ratio is ~2.59 (13.4/5.18)
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


def _prepare_stroke_classification(artifacts, all_shuttle, all_pose, all_player_detections, court, vid_w, vid_h, gpu_cfg, pose_model, all_pose_secondary=None, bst_batch_size=32):
    """Run stroke classification using colab's BST implementation (GPU-efficient).

    This is kept in colab because the backend's StrokeClassificationStage uses a
    different BST model path and may not load correctly. The colab BST loading
    follows the original paper's approach.
    """
    from app.pipeline.hits import HitFrameLocalizationStage
    from app.pipeline.base import StageConfig

    config = StageConfig(gpu_enabled=True)

    hits_result = HitFrameLocalizationStage().run(artifacts, config)
    if hits_result.status == "error":
        print(f"  Hit detection failed: {hits_result.error}")
        return [], []
    hits = hits_result.metadata.get("hits", [])
    print(f"  Found {len(hits)} hits")

    if not hits:
        return [], []

    shuttle_df = artifacts.get_parquet("shuttle")
    pose_df = artifacts.get_parquet("pose")

    hits_data = pd.DataFrame(hits)
    if pose_df is not None and len(pose_df) > 0:
        from collections import defaultdict
        by_player = defaultdict(list)
        for _, row in pose_df.iterrows():
            by_player[row.get("player_id", "")].append(row)

    # BST configuration
    SEQ_LEN = 30
    BST_CLASSES = [
        "net_shot", "block", "smash", "lift", "clear", "drive",
        "drop", "push", "rush", "cross_court", "short_serve", "long_serve"
    ]

    BONE_PAIRS = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
        (3, 5), (4, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def create_bones(joints):
        bones = []
        for start, end in BONE_PAIRS:
            start_j = joints[:, :, start, :]
            end_j = joints[:, :, end, :]
            bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
            bones.append(bone)
        return np.stack(bones, axis=-2)

    def normalize_joints_bstdiag(coords, det_bbox=None):
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
        n_frames = len(clip_frames)
        joints = np.zeros((seq_len, 2, 17, 2), dtype=np.float32)
        shuttle = np.zeros((seq_len, 2), dtype=np.float32)
        pos = np.zeros((seq_len, 2, 2), dtype=np.float32)

        for t, frame in enumerate(clip_frames[:seq_len]):
            if 'shuttle_x' in frame and 'shuttle_y' in frame:
                shuttle[t] = [frame['shuttle_x'], frame['shuttle_y']]

            det_bboxes = frame.get('det_bboxes', {})
            for p_idx, pid in enumerate(['player_2', 'player_1']):
                if pid in frame.get('pose', {}):
                    kps = frame['pose'][pid]
                    if kps is not None and kps.shape == (17, 3):
                        coords = kps[:, :2]
                        det_bbox = det_bboxes.get(pid)
                        joints[t, p_idx] = normalize_joints_bstdiag(coords, det_bbox=det_bbox)
                        feet_y = max(coords[15, 1], coords[16, 1])
                        feet_x = (coords[15, 0] + coords[16, 0]) / 2
                        pos[t, p_idx] = [feet_x / vid_w, feet_y / vid_h]

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
    import torch
    import torch.nn as nn
    import math

    bst_path = str(BST_PATH) if BST_PATH.exists() else None
    model = None
    seq_len = SEQ_LEN
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if bst_path:
        try:
            checkpoint = torch.load(bst_path, map_location=device, weights_only=False)

            state_dict = checkpoint if isinstance(checkpoint, dict) and 'model' not in checkpoint else None
            if state_dict is None and isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model'] if isinstance(checkpoint['model'], dict) else None

            if state_dict and any('tcn_pose' in k for k in list(state_dict.keys())[:10]):
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
                print(f"BST_CG loaded (FP32): in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
            else:
                print("BST state_dict not recognized, using rule-based fallback")
        except Exception as e:
            print(f"BST load error: {e}")
    
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
    
    # Build detection bbox lookup per player (side-based assignment, matches pose data)
    det_bbox_lookup = {}
    if player_detections:
        for d in player_detections:
            side = d.get("side", "near")
            pid = "player_1" if side == "near" else "player_2"
            frame = d.get("frame")
            if frame is not None:
                if pid not in det_bbox_lookup:
                    det_bbox_lookup[pid] = {}
                det_bbox_lookup[pid][frame] = d["bbox"]
    
    # Process each hit
    shots = []
    
    if model is not None:
        import torch
        
        hit_frames_sorted = sorted([h['frame'] for h in hits_data])
        
        all_clips = []
        for hit in hits_data:
            hit_frame = hit['frame']
            hit_pos = hit_frames_sorted.index(hit_frame)
            start_frame = hit_frames_sorted[hit_pos - 1] if hit_pos > 0 else max(0, hit_frame - seq_len // 2)
            end_frame = hit_frames_sorted[hit_pos + 1] + 2 if hit_pos < len(hit_frames_sorted) - 1 else hit_frame + seq_len // 2 + 1
            
            clip_frames = []
            for f in range(start_frame, end_frame):
                frame_data = {}
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
                raw_pose = pose_by_frame.get(f, {})
                frame_data['pose'] = {'player_1': raw_pose.get('player_1'), 'player_2': raw_pose.get('player_2')}
                frame_data['det_bboxes'] = {
                    'player_1': det_bbox_lookup.get('player_1', {}).get(f),
                    'player_2': det_bbox_lookup.get('player_2', {}).get(f),
                }
                clip_frames.append(frame_data)
            
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
            
            JnB, shuttle_arr, pos_arr, v_len = prepare_bst_clip(clip_frames, seq_len)
            all_clips.append((hit, JnB, shuttle_arr, pos_arr, v_len))
        
        for batch_start in range(0, len(all_clips), bst_batch_size):
            batch = all_clips[batch_start:batch_start + bst_batch_size]
            JnB_batch = torch.from_numpy(np.stack([c[1] for c in batch])).float().to(device)
            shuttle_batch = torch.from_numpy(np.stack([c[2] for c in batch])).float().to(device)
            pos_batch = torch.from_numpy(np.stack([c[3] for c in batch])).float().to(device)
            vlen_batch = torch.tensor([c[4] for c in batch], dtype=torch.long).to(device)
            if device == "cuda":
                JnB_batch = JnB_batch.half()
                shuttle_batch = shuttle_batch.half()
                pos_batch = pos_batch.half()
            
            with torch.no_grad():
                logits_batch = model(JnB_batch, shuttle_batch, pos_batch, vlen_batch)
                probs_batch = torch.softmax(logits_batch.float(), dim=1).cpu().numpy()
            
            for j, (hit, *_) in enumerate(batch):
                hit_frame = hit['frame']
                probs = probs_batch[j]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                
                if pred_idx == 0:
                    second_idx = int(np.argsort(probs)[-2])
                    second_conf = float(probs[second_idx])
                    if second_conf > 0.05:
                        pred_idx = second_idx
                        confidence = second_conf
                    else:
                        stroke_type = _rule_based_shuttle_predict(shuttle_df, hit_frame, vid_w, vid_h)
                        shots.append({"frame": hit_frame, "hit_confidence": hit['confidence'],
                                      "stroke_type": stroke_type, "stroke_confidence": confidence})
                        continue
                
                if pred_idx == 0:
                    stroke_type = "unknown"
                elif 1 <= pred_idx <= 12:
                    stroke_type = BST_CLASSES[pred_idx - 1] if pred_idx - 1 < len(BST_CLASSES) else "clear"
                elif 13 <= pred_idx <= 24:
                    stroke_type = BST_CLASSES[pred_idx - 13] if pred_idx - 13 < len(BST_CLASSES) else "clear"
                else:
                    stroke_type = "clear"
                
                shots.append({"frame": hit_frame, "hit_confidence": hit['confidence'],
                              "stroke_type": stroke_type, "stroke_confidence": confidence})
        
        del JnB_batch, shuttle_batch, pos_batch, vlen_batch
        if device == "cuda":
            torch.cuda.empty_cache()
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

    # Post-classification smoothing: if a shot has low confidence and
    # its neighbors agree on a different class, adopt the neighbors' class.
    # Requires >= 3 neighbors to agree to prevent cascade from small samples.
    if len(shots) > 2:
        for i in range(len(shots)):
            if shots[i]["stroke_confidence"] >= 0.25 or shots[i]["stroke_type"] == "unknown":
                continue
            neighbors = []
            for j in range(max(0, i - 2), min(len(shots), i + 3)):
                if j != i and shots[j]["stroke_type"] != "unknown":
                    neighbors.append(shots[j]["stroke_type"])
            if neighbors:
                from collections import Counter
                majority = Counter(neighbors).most_common(1)[0]
                if majority[0] != shots[i]["stroke_type"] and majority[1] >= 3:
                    shots[i]["stroke_type"] = majority[0]
                    shots[i]["stroke_confidence"] = 0.3

    return shots


# ─── Shuttle-Coach Integration ──────────────────────────────────────────────

def _shuttle_coach_recovery_time(positions_df, shots_df, player_ids):
    """Compute recovery time metric for each player."""
    import numpy as np
    results = []
    # Skip if required columns are missing
    if "court_x" not in positions_df.columns or "court_y" not in positions_df.columns:
        return results
    if "player_id" not in positions_df.columns:
        return results
    # Use ts if available, otherwise derive from frame
    ts_col = "ts" if "ts" in positions_df.columns else "frame"
    for pid in player_ids:
        pos = positions_df[positions_df["player_id"] == pid].dropna(subset=["court_x", "court_y"]).sort_values(ts_col)
        if len(pos) < 10:
            continue
        base = np.array([pos["court_x"].median(), pos["court_y"].median()])
        player_shots = shots_df[shots_df["player_id"] == pid].sort_values("start_ts")
        recov = []
        for _, s in player_shots.iterrows():
            shot_ts = s.get("start_ts", s.get("frame", 0) / 30.0)
            after = pos[pos[ts_col] >= shot_ts].head(60)
            if after.empty:
                continue
            d = np.linalg.norm(after[["court_x", "court_y"]].to_numpy() - base, axis=1)
            back = np.argmax(d < 1.0) if (d < 1.0).any() else len(d) - 1
            recov.append(after[ts_col].iloc[back] - shot_ts)
        if recov:
            results.append({
                "metric_id": "movement.recovery_time",
                "player_id": pid,
                "value": float(np.mean(recov)),
                "unit": "s",
                "sample_size": len(recov),
                "confidence": min(1.0, len(recov) / 30),
                "context": {"median": float(np.median(recov))}
            })
    return results


def _shuttle_coach_shot_mix(shots_df, player_ids):
    """Compute shot distribution for each player."""
    results = []
    for pid in player_ids:
        s = shots_df[shots_df["player_id"] == pid]
        if s.empty:
            continue
        mix = (s["stroke_type"].value_counts(normalize=True) * 100).round(1).to_dict()
        results.append({
            "metric_id": "shots.mix",
            "player_id": pid,
            "value": mix,
            "unit": "%",
            "sample_size": len(s),
            "confidence": float(s["stroke_confidence"].mean()) if "stroke_confidence" in s.columns else 1.0,
            "context": {}
        })
    return results


def _shuttle_coach_error_location(rallies_df, player_ids):
    """Compute error breakdown for each player."""
    results = []
    for pid in player_ids:
        if "winner_player_id" not in rallies_df.columns:
            continue
        lost = rallies_df[(rallies_df["winner_player_id"].notna()) & (rallies_df["winner_player_id"] != pid)]
        if "end_reason" in rallies_df.columns and not lost.empty:
            reasons = (lost["end_reason"].value_counts(normalize=True) * 100).round(1).to_dict()
        else:
            reasons = {}
        results.append({
            "metric_id": "errors.location_reason",
            "player_id": pid,
            "value": reasons,
            "unit": "%",
            "sample_size": int(len(lost)),
            "confidence": 1.0,
            "context": {}
        })
    return results


def _shuttle_coach_derive_findings(metrics):
    """Derive coaching findings from metrics."""
    findings = []
    
    # Group by metric_id
    by_id = {}
    for m in metrics:
        by_id.setdefault(m["metric_id"], []).append(m)
    
    # Slow recovery
    for m in by_id.get("movement.recovery_time", []):
        if m["value"] > 0.8 and m["sample_size"] >= 15:
            severity = min(1.0, (m["value"] - 0.8) / 0.8)
            findings.append({
                "code": "slow_recovery",
                "player_id": m["player_id"],
                "severity": severity,
                "headline": "Slow recovery to base position",
                "detail": f"Average recovery {m['value']:.2f}s (median {m['context'].get('median', 0):.2f}s) over {m['sample_size']} shots. Returning to base faster would reduce time spent out of position.",
                "evidence": [m["metric_id"]],
                "drill": "Split-step practice: bounce on toes, explode to shuttle on opponent's hit."
            })
    
    # High unforced errors
    for m in by_id.get("errors.location_reason", []):
        if isinstance(m["value"], dict):
            unforced = m["value"].get("unforced_error", 0)
            if unforced > 20 and m["sample_size"] >= 8:
                severity = min(1.0, unforced / 50)
                findings.append({
                    "code": "high_unforced",
                    "player_id": m["player_id"],
                    "severity": severity,
                    "headline": "High unforced error rate",
                    "detail": f"{unforced:.0f}% of lost points are unforced errors ({m['sample_size']} lost rallies). Shot tolerance is the highest-leverage area to improve.",
                    "evidence": [m["metric_id"]],
                    "drill": "Consistency drills: rally with partner, focus on keeping shuttle in play."
                })
            
            # High net errors
            net_err = m["value"].get("net", 0)
            if net_err > 8 and m["sample_size"] >= 8:
                severity = min(1.0, net_err / 20)
                findings.append({
                    "code": "high_net_errors",
                    "player_id": m["player_id"],
                    "severity": severity,
                    "headline": "Frequent net errors",
                    "detail": f"{net_err:.0f}% of lost points ended at the net ({m['sample_size']} lost rallies). Tighten net shots and improve clearance.",
                    "evidence": [m["metric_id"]],
                    "drill": "Net clearance drills: practice lifting just over the tape from mid-court."
                })
    
    # Shot variety
    for m in by_id.get("shots.mix", []):
        if isinstance(m["value"], dict) and m["value"]:
            max_shot = max(m["value"].values())
            max_type = max(m["value"], key=m["value"].get)
            if max_shot > 40 and m["sample_size"] >= 15:
                severity = (max_shot - 40) / 40
                findings.append({
                    "code": "low_variety",
                    "player_id": m["player_id"],
                    "severity": severity,
                    "headline": f"Predictable shot selection ({max_type}: {max_shot:.0f}%)",
                    "detail": f"Dominant stroke accounts for {max_shot:.0f}% of shots. Opponents can read your patterns.",
                    "evidence": [m["metric_id"]],
                    "drill": "Pattern-breaking drill: after 2 identical shots, forced switch to a different stroke."
                })
            elif max_shot < 25 and m["sample_size"] >= 15:
                findings.append({
                    "code": "good_variety",
                    "player_id": m["player_id"],
                    "severity": 0.1,
                    "headline": "Good shot variety",
                    "detail": f"No single stroke dominates (max {max_shot:.0f}%). This keeps opponents guessing.",
                    "evidence": [m["metric_id"]],
                    "drill": ""
                })
    
    return findings


def _shuttle_coach_to_ui_format(findings, metrics):
    """Map shuttle-coach findings to UI CoachPanel format."""
    strengths = []
    weaknesses = []
    improvements = []
    drills = []
    evidence = []
    
    # Sort by severity
    findings = sorted(findings, key=lambda f: f.get("severity", 0), reverse=True)
    
    for f in findings:
        severity = f.get("severity", 0)
        headline = f.get("headline", "")
        detail = f.get("detail", "")
        drill = f.get("drill", "")
        player_id = f.get("player_id", "")
        
        # Add player prefix for clarity
        player_prefix = f"[{'Near' if player_id == 'player_1' else 'Far'}] " if player_id else ""
        finding_text = f"{player_prefix}{headline} — {detail}"
        
        if severity < 0.3:
            strengths.append(finding_text)
        else:
            weaknesses.append(finding_text)
            improvements.append(finding_text)
            if drill:
                drills.append(drill)
        
        # Add evidence
        evidence.append({
            "finding": finding_text,
            "metrics": [f"{m['metric_id']}: {m['value']}" for m in metrics if m["metric_id"] in f.get("evidence", [])]
        })
    
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "top_3_improvements": improvements[:3],
        "recommended_drills": drills[:3],
        "evidence": evidence[:10]  # Limit evidence to prevent huge reports
    }


def stage_shuttle_coach(debug_dir, shots, rallies, player_detections, shuttle_data):
    """Run shuttle-coach analysis on exported parquet files."""
    import pandas as pd
    
    player_ids = sorted(set(s.get("player_id", "player_1") for s in shots))
    
    # Create DataFrames
    shots_df = pd.DataFrame(shots)
    rallies_df = pd.DataFrame(rallies)
    positions_df = pd.DataFrame(player_detections)
    
    # Ensure required columns exist
    if "start_ts" not in shots_df.columns:
        shots_df["start_ts"] = shots_df["frame"] / 30.0
    if "court_x" not in positions_df.columns:
        positions_df["court_x"] = np.nan
        positions_df["court_y"] = np.nan
    
    # Convert side to player_id if needed (Colab uses 'side', backend uses 'player_id')
    if "player_id" not in positions_df.columns and "side" in positions_df.columns:
        side_map = {"near": "player_1", "far": "player_2"}
        positions_df["player_id"] = positions_df["side"].map(side_map).fillna("player_1")
    
    # Compute metrics
    metrics = []
    metrics.extend(_shuttle_coach_recovery_time(positions_df, shots_df, player_ids))
    metrics.extend(_shuttle_coach_shot_mix(shots_df, player_ids))
    metrics.extend(_shuttle_coach_error_location(rallies_df, player_ids))
    
    # Derive findings
    findings = _shuttle_coach_derive_findings(metrics)
    
    # Map to UI format
    ui_format = _shuttle_coach_to_ui_format(findings, metrics)
    
    return {
        "metrics": metrics,
        "findings": findings,
        "ui": ui_format
    }


# ─── Main Pipeline ───────────────────────────────────────────────────────────

BATCH_SIZE = 2000


def _generate_report(court, players_data, shots, rallies, coach,
                     tactical, fitness, footwork, technical, court_analytics, fps=30):
    """Build the final report dict from all analytics."""
    shot_dist = {}
    for pid, data in tactical.items():
        shot_dist.update(data.get("shot_distribution", {}))

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
    from app.coach.engine import CoachEngine
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
            vram_gb = props.total_mem / (1024 ** 3)
            print(f"  GPU: {props.name} ({vram_gb:.1f} GB)")
        else:
            print("  GPU: CUDA requested but not available, using CPU")
    except Exception:
        print("  GPU: detection failed")
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
    tracknet = TrackNetV3(str(TRACKNET_PATH), device=device, chunk_size=gpu_cfg["tracknet_chunk"])
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
                               all_pose_secondary=all_pose_secondary, corners=corners)
                batch_frames = []
                batch_global_indices = []
                gc.collect()
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
                       all_pose_secondary=all_pose_secondary, corners=corners)
        gc.collect()

    cap.release()

    print(f"\n  ML stages complete:")
    print(f"    Shuttle: {len(all_shuttle)} frames")
    print(f"    Players: {len(all_player_detections)} detections")
    print(f"    Pose:    {len(all_pose)} frames")

    # Build player summary
    players = {"player_1": {"id": "player_1", "side": "near", "detections": []},
               "player_2": {"id": "player_2", "side": "far", "detections": []}}
    for d in all_player_detections:
        side = d.get("side", "near")
        pid = "player_1" if side == "near" else "player_2"
        players[pid]["detections"].append(d)
    players_data = {"players": [{"id": p["id"], "side": p["side"], "detection_count": len(p["detections"])} for p in players.values()]}

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
        config = StageConfig(gpu_enabled=False)

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
        shots_result = StrokeClassificationStage().run(store, config)
        shots_df = store.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            print("  Backend stroke classification produced no shots, using colab BST fallback")
            shots = []
        else:
            shots = shots_df.to_dict("records")
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

        court_analytics = store.get("court_analytics") or {}
        footwork = store.get("footwork_analytics") or {}
        fitness = store.get("fitness_analytics") or {}
        tactical = store.get("tactical_analytics") or {}
        technical = store.get("technical_analytics") or {}
        print(f"  Court: {len(court_analytics.get('zone_transitions', []))} transitions")

        # ── Coach recommendations (backend engine) ──
        print("\n[5/5] Coach recommendations...")
        engine = CoachEngine()
        all_players = set(list(tactical.keys()) + list(fitness.keys()))
        if not all_players:
            all_players = {"player_1"}

        coach = {"strengths": [], "weaknesses": [], "top_3_improvements": [],
                 "recommended_drills": [], "evidence": [], "rally_stats": None}

        for pid in sorted(all_players):
            player_analytics = {
                "tactical_analytics": tactical,
                "fitness_analytics": fitness,
                "footwork_analytics": footwork,
                "court_analytics": court_analytics,
                "_rallies_df": rallies_df,
                "_shots_df": shots_df,
            }
            result = engine.generate(player_analytics, pid)
            for key in coach:
                if key in result:
                    if isinstance(coach[key], list):
                        coach[key].extend(result[key])
                    else:
                        coach[key] = result[key]

        rally_stats = stage_rally_stats(shots, rallies)
        coach["rally_stats"] = rally_stats
        print(f"  {len(coach['strengths'])} strengths, {len(coach['weaknesses'])} weaknesses")

        # ── Shuttle-coach advanced analytics ──
        print("\n  Shuttle-coach analytics...")
        try:
            shuttle_coach_result = stage_shuttle_coach(debug_dir, shots, rallies, all_player_detections, all_shuttle)
            sc_ui = shuttle_coach_result["ui"]
            for w in sc_ui.get("weaknesses", []):
                if w not in coach["weaknesses"]:
                    coach["weaknesses"].append(w)
            for s in sc_ui.get("strengths", []):
                if s not in coach["strengths"]:
                    coach["strengths"].append(s)
            for d in sc_ui.get("recommended_drills", []):
                if d and d not in coach["recommended_drills"]:
                    coach["recommended_drills"].append(d)
            for e in sc_ui.get("evidence", []):
                if e not in coach["evidence"]:
                    coach["evidence"].append(e)
            coach["top_3_improvements"] = coach["weaknesses"][:3]
            print(f"  Shuttle-coach: {len(shuttle_coach_result['findings'])} findings")
        except Exception as e:
            print(f"  Shuttle-coach skipped: {e}")

    # ── Build and save report ──
    report = _generate_report(court, players_data, shots, rallies, coach,
                              tactical, fitness, footwork, technical, court_analytics, fps=video_fps)

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
                   corners=None):
    """Run ML stages on one batch of frames, append results to accumulators."""
    if not frames:
        return

    tag = f"  Batch {batch_num}/{total_batches}"

    # 1. Player tracking (YOLOv8)
    tqdm.write(f"{tag} | YOLOv8 tracking {len(frames)} frames...")
    batch_det = tracker.track_batch(frames, 0)
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
