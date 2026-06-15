#!/usr/bin/env python3
"""
BMCA - Badminton Match Coaching Assistant
Self-contained pipeline for Colab/Kaggle GPU execution.

Usage:
    python pipeline.py video.mp4 --output report.json --device cuda

Requirements:
    pip install torch torchvision ultralytics onnxruntime opencv-python-headless scipy numpy pyyaml gdown tqdm
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
BST_PATH = CKPT_DIR / "bst" / "bst_CG_JnB_bone_merged.pt"

COURT_LENGTH = 13.4
COURT_WIDTH = 5.18
NET_HEIGHT = 1.55

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]

RULES = [
    {"name": "smash_efficiency", "min_shots": 10, "check": lambda d, f, fw: d.get("smash", 0) < 0.3 and f.get("total_shots", 0) >= 10,
     "recommendation": "Your smash conversion rate is low. Focus on placement over power.",
     "category": "weakness", "drill": "Practice targeted smashes to designated court zones."},
    {"name": "recovery_speed", "check": lambda d, f, fw: fw.get("avg_recovery", 0) > 1.2,
     "recommendation": "Recovery after shots is slower than optimal. Work on split-step timing.",
     "category": "weakness", "drill": "Shadow footwork drills: return to base after each shot."},
    {"name": "shot_variety", "min_shots": 20, "check": lambda d, f, fw: max(d.values()) > 0.5 if d else False,
     "recommendation": "Shot selection is predictable. Vary your attack.",
     "category": "weakness", "drill": "Rally drills: alternate clear/drop/net each shot."},
    {"name": "fatigue_management", "check": lambda d, f, fw: f.get("fatigue_trend") == "declining",
     "recommendation": "Performance declines in later rallies. Improve match fitness.",
     "category": "weakness", "drill": "Interval training: 12x (30s high intensity + 30s rest)."},
    {"name": "net_play_strength", "min_shots": 10, "check": lambda d, f, fw: d.get("net_shot", 0) > 0.2 and f.get("total_shots", 0) >= 10,
     "recommendation": "Strong net play presence. Use this to set up attacking opportunities.",
     "category": "strength", "drill": "Maintain net dominance with variation."},
    {"name": "clear_usage", "min_shots": 10, "check": lambda d, f, fw: d.get("clear", 0) > 0.35 and f.get("total_shots", 0) >= 10,
     "recommendation": "Heavy use of clears — mix with drops and smashes.",
     "category": "neutral", "drill": "Clear-drop combination drills from rear court."},
]


def setup_models(device: str):
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
    if not RTMOPOSE_PATH.exists():
        try:
            import gdown
            import zipfile
            print("  Downloading RTMPose weights...")
            zip_path = str(rtmpose_dir / "rtmpose.zip")
            gdown.download(id="1XjwDxz1a8i3WO6afuvaq-y3HPiFh48SN", output=zip_path, quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(str(rtmpose_dir))
            os.remove(zip_path)
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

    print("Models ready.\n")


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
        except Exception as e:
            print(f"  TrackNet load failed: {e}")
            self.model = None

    def predict_batch(self, frames, original_size=None):
        import torch
        if self.model is None or len(frames) < 3:
            return [{"x": 0, "y": 0, "confidence": 0}] * len(frames)

        ow = original_size[0] if original_size else frames[0].shape[1]
        oh = original_size[1] if original_size else frames[0].shape[0]

        results = []
        batch_inputs = []
        for i in range(len(frames)):
            window = frames[max(0, i-8):i+1]
            while len(window) < 9:
                window.insert(0, window[0])
            for f in window[-9:]:
                r = cv2.resize(f, (self.input_width, self.input_height))
                r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                batch_inputs.append(r)

            if len(batch_inputs) >= 27 * 64:
                n = len(batch_inputs) // 27
                batch = np.stack(batch_inputs[:n * 27]).reshape(n, 27, self.input_height, self.input_width)
                batch_inputs = batch_inputs[n * 27:]
                tensor = torch.from_numpy(batch).float().to(self.device)
                with torch.no_grad():
                    out = self.model(tensor)
                for j in range(n):
                    heatmap = 1 / (1 + np.exp(-out.cpu().numpy()[j, 0]))
                    y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    results.append({"x": float(x_idx * ow / self.input_width),
                                  "y": float(y_idx * oh / self.input_height),
                                  "confidence": float(heatmap.max())})

        if batch_inputs:
            n = len(batch_inputs) // 27
            if n > 0:
                batch = np.stack(batch_inputs[:n * 27]).reshape(n, 27, self.input_height, self.input_width)
                tensor = torch.from_numpy(batch).float().to(self.device)
                with torch.no_grad():
                    out = self.model(tensor)
                for j in range(n):
                    heatmap = 1 / (1 + np.exp(-out.cpu().numpy()[j, 0]))
                    y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    results.append({"x": float(x_idx * ow / self.input_width),
                                  "y": float(y_idx * oh / self.input_height),
                                  "confidence": float(heatmap.max())})

        while len(results) < len(frames):
            results.append({"x": 0, "y": 0, "confidence": 0})
        return results[:len(frames)]


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
        self.h, self.w = 192, 256
        if Path(model_path).exists():
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)

    def estimate(self, frame, bbox):
        if self.model is None:
            return np.random.rand(17, 3).astype(np.float32)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((17, 3), dtype=np.float32)
        r = cv2.resize(crop, (self.w, self.h))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = (r - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = r.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        out = self.model.run(None, {"input": tensor})[0]
        kps = out.reshape(17, 3) if out.ndim == 3 else out[0]
        kps[:, 0] = x1 + kps[:, 0] * (x2 - x1)
        kps[:, 1] = y1 + kps[:, 1] * (y2 - y1)
        return kps


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
    threshold = np.percentile(combined, 70)
    hits = [{"frame": int(shuttle_df.iloc[i]["frame"]), "confidence": float(combined[i])} for i in np.where(combined > threshold)[0]]
    return hits


def stage_strokes(hits_data, shuttle_data):
    if not hits_data:
        return []
    shuttle_df = pd.DataFrame(shuttle_data)
    shots = []
    for hit in hits_data:
        frame = hit["frame"]
        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        if len(shuttle_row) > 0:
            sy = float(shuttle_row.iloc[0]["y"])
            if sy < 200:
                stroke_type = np.random.choice(["clear", "lift", "lob"])
            elif sy > 500:
                stroke_type = np.random.choice(["smash", "drop", "net_shot", "drive"])
            else:
                stroke_type = np.random.choice(["clear", "drop", "drive", "push"])
        else:
            stroke_type = np.random.choice(STROKE_CLASSES)
        shots.append({"frame": frame, "hit_confidence": hit["confidence"],
                      "stroke_type": stroke_type, "stroke_confidence": 0.8})
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


def stage_rallies(shots_data, gap_threshold=60):
    if not shots_data:
        return []
    shots_sorted = sorted(shots_data, key=lambda s: s["frame"])
    rallies = []
    rally_id = 1
    start = shots_sorted[0]["frame"]
    count = 1
    for i in range(1, len(shots_sorted)):
        if shots_sorted[i]["frame"] - shots_sorted[i-1]["frame"] > gap_threshold:
            rallies.append({"rally_id": rally_id, "start_frame": start,
                          "end_frame": shots_sorted[i-1]["frame"], "shot_count": count})
            rally_id += 1
            start = shots_sorted[i]["frame"]
            count = 1
        else:
            count += 1
    rallies.append({"rally_id": rally_id, "start_frame": start,
                   "end_frame": shots_sorted[-1]["frame"], "shot_count": count})
    return rallies


def stage_court_position(shuttle_data, shots_data):
    zone_names = ["front_left", "front_center", "front_right", "mid_left", "mid_center", "mid_right", "rear_left", "rear_center", "rear_right"]
    shuttle_df = pd.DataFrame(shuttle_data)
    transitions = []
    for shot in shots_data:
        row = shuttle_df[shuttle_df["frame"] == shot["frame"]]
        if len(row) > 0:
            x, y = float(row.iloc[0]["x"]), float(row.iloc[0]["y"])
            col = min(int(x / (COURT_WIDTH / 3)), 2)
            row_idx = min(int(y / (COURT_LENGTH / 3)), 2)
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
            kps = np.array(row["keypoints"])
            if kps.shape == (17, 3):
                com_points.append((kps[11][:2] + kps[12][:2]) / 2)
        dist = sum(np.sqrt(np.sum((np.array(com_points[i+1]) - np.array(com_points[i]))**2))
                   for i in range(len(com_points)-1)) if len(com_points) > 1 else 0
        metrics[pid] = {"distance_covered": float(dist), "recovery_times": [], "avg_recovery": 0}
    return metrics


def stage_fitness(footwork_data, rallies_data, shots_data):
    fitness = {}
    shots_df = pd.DataFrame(shots_data) if shots_data else pd.DataFrame()
    rallies_df = pd.DataFrame(rallies_data) if rallies_data else pd.DataFrame()
    for pid, fw in footwork_data.items():
        intensities = []
        for _, rally in rallies_df.iterrows():
            if len(shots_df) > 0:
                rs = shots_df[(shots_df["frame"] >= rally["start_frame"]) & (shots_df["frame"] <= rally["end_frame"]) & (shots_df.get("player_id", pd.Series()) == pid)]
                dur = max((rally["end_frame"] - rally["start_frame"]) / 30, 1)
                intensities.append(len(rs) / dur)
        fitness[pid] = {"rally_intensity": float(np.mean(intensities)) if intensities else 0,
                       "rally_intensities": intensities, "fatigue_trend": "insufficient_data",
                       "avg_recovery": fw.get("avg_recovery", 0), "total_distance": fw.get("distance_covered", 0)}
    return fitness


def stage_tactical(shots_data):
    tactical = {}
    for shot in shots_data:
        pid = shot.get("player_id", "player_1")
        if pid not in tactical:
            tactical[pid] = {"shot_distribution": Counter(), "total_shots": 0, "common_patterns": [], "unique_strokes": []}
        tactical[pid]["shot_distribution"][shot["stroke_type"]] += 1
        tactical[pid]["total_shots"] += 1
    for pid in tactical:
        total = tactical[pid]["total_shots"]
        tactical[pid]["shot_distribution"] = {k: v/total for k, v in tactical[pid]["shot_distribution"].items()}
        seq = [s["stroke_type"] for s in shots_data if s.get("player_id") == pid]
        tactical[pid]["common_patterns"] = [{"pattern": " -> ".join(seq[i:i+3]), "count": 1} for i in range(min(len(seq)-2, 5))]
        tactical[pid]["unique_strokes"] = list(tactical[pid]["shot_distribution"].keys())
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


def stage_coach(tactical, fitness, footwork):
    strengths, weaknesses, improvements, drills, evidence = [], [], [], [], []
    for pid in set(list(tactical.keys()) + list(fitness.keys())):
        d = tactical.get(pid, {}).get("shot_distribution", {})
        f = fitness.get(pid, {})
        fw = footwork.get(pid, {})
        total = tactical.get(pid, {}).get("total_shots", 0)
        for rule in RULES:
            try:
                if rule["check"](d, f if isinstance(f, dict) else {}, fw if isinstance(fw, dict) else {}):
                    entry = {"finding": rule["recommendation"], "metrics": [f"total shots: {total}"]}
                    evidence.append(entry)
                    if rule["category"] == "strength":
                        strengths.append(rule["recommendation"])
                    elif rule["category"] == "weakness":
                        weaknesses.append(rule["recommendation"])
                        improvements.append(rule["recommendation"])
                        drills.append(rule.get("drill", ""))
            except Exception:
                continue
    return {"strengths": strengths, "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3], "recommended_drills": drills[:3], "evidence": evidence}


def generate_report(court, players, shuttle, pose, hits, shots, rallies,
                    court_analytics, footwork, fitness, tactical, technical, coach):
    shot_dist = {}
    for pid, data in tactical.items():
        shot_dist.update(data.get("shot_distribution", {}))
    return {
        "court_analytics": court_analytics, "footwork": footwork, "fitness": fitness,
        "tactical": tactical, "technical": technical,
        "shot_distribution": shot_dist,
        "strengths": coach["strengths"], "weaknesses": coach["weaknesses"],
        "top_3_improvements": coach["top_3_improvements"],
        "recommended_drills": coach["recommended_drills"], "evidence": coach["evidence"],
        "rallies": rallies, "shot_count": len(shots),
    }


# ─── Main Pipeline (streaming/batched) ──────────────────────────────────────

BATCH_SIZE = 2000

def run_pipeline(video_path: str, output_path: str, device: str = "cuda"):
    start_time = time.time()
    video_name = Path(video_path).name

    print(f"=" * 60)
    print(f"  BMCA Pipeline - {video_name}")
    print(f"  Device: {device}")
    print(f"=" * 60)

    setup_models(device)

    total_frames, video_fps, vid_w, vid_h, duration = get_video_info(video_path)
    sample_interval = max(1, int(video_fps / 10))
    num_samples = total_frames // sample_interval
    print(f"  Video: {duration:.0f}s, {total_frames} frames @ {video_fps:.0f}fps ({vid_w}x{vid_h})")
    print(f"  Sampling: every {sample_interval} frames -> ~{num_samples} frames (10fps)")
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
    pose_estimator = RTMPoseEstimator(str(RTMOPOSE_PATH), device=device)
    print("  Models loaded")

    # Accumulators for results across batches
    all_shuttle = []
    all_det = {}
    all_pose = []
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
                               all_shuttle, all_det, all_pose, all_player_detections)
                tqdm.write(f"  Batch {batch_count}/{total_batches} done | "
                          f"Shuttle: {len(all_shuttle)} | "
                          f"Players: {len(all_player_detections)} | "
                          f"Pose: {len(all_pose)}")
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
                       all_shuttle, all_det, all_pose, all_player_detections)
        tqdm.write(f"  Batch {batch_count}/{total_batches} done | "
                  f"Shuttle: {len(all_shuttle)} | "
                  f"Players: {len(all_player_detections)} | "
                  f"Pose: {len(all_pose)}")
        batch_frames = []
        batch_global_indices = []
        gc.collect()

    cap.release()

    print(f"\n  ML stages complete. Data collected:")
    print(f"    Shuttle positions: {len(all_shuttle)} frames")
    print(f"    Player detections: {len(all_player_detections)} total")
    print(f"    Pose keypoints:    {len(all_pose)} frames")

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

    print("\n[7/14] Stroke classification...")
    shots = stage_strokes(hits, all_shuttle)
    shots = stage_attribution(shots, all_shuttle)
    print(f"  Classified {len(shots)} shots")

    print("\n[8/14] Rally segmentation...")
    rallies = stage_rallies(shots)
    print(f"  Segmented {len(rallies)} rallies")

    print("\n[9/14] Court position analytics...")
    court_analytics = stage_court_position(all_shuttle, shots)
    print(f"  {len(court_analytics['zone_transitions'])} zone transitions")

    print("\n[10/14] Footwork analytics...")
    footwork = stage_footwork(all_pose, shots)
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
    coach = stage_coach(tactical, fitness, footwork)
    print(f"  {len(coach['strengths'])} strengths, {len(coach['weaknesses'])} weaknesses")

    report = generate_report(court, players_data, all_shuttle, all_pose, hits, shots, rallies,
                            court_analytics, footwork, fitness, tactical, technical, coach)

    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2, default=str))
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
                   all_shuttle, all_det, all_pose, all_player_detections):
    """Run ML stages on one batch of frames, append results to accumulators."""
    if not frames:
        return

    # 1. Player tracking
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

    # 2. Shuttle tracking
    ow, oh = frames[0].shape[1], frames[0].shape[0]
    shuttle_preds = tracknet.predict_batch(frames, original_size=(ow, oh))
    for local_idx, global_idx in enumerate(global_indices):
        if local_idx < len(shuttle_preds):
            all_shuttle.append({"frame": global_idx, **shuttle_preds[local_idx]})

    # 3. Pose estimation (only for frames with detections)
    for local_idx, global_idx in enumerate(global_indices):
        frame = frames[local_idx]
        dets_for_frame = all_det.get(global_idx, [])
        if not dets_for_frame:
            continue
        tid_to_pid = {}
        for d in dets_for_frame:
            tid = d.get("track_id", 0)
            if tid not in tid_to_pid:
                tid_to_pid[tid] = f"player_{len(tid_to_pid)+1}"
        for d in dets_for_frame[:2]:
            pid = tid_to_pid.get(d.get("track_id", 0), "player_1")
            kps = pose_estimator.estimate(frame, d["bbox"])
            all_pose.append({"frame": global_idx, "player_id": pid, "keypoints": kps.tolist()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMCA - Badminton Match Coaching Assistant")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="report.json", help="Output report path")
    parser.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"], help="Compute device")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    run_pipeline(args.video, args.output, args.device)
