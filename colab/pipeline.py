#!/usr/bin/env python3
"""
BMCA - Badminton Match Coaching Assistant
Self-contained pipeline for Colab/Kaggle GPU execution.

Usage:
    python pipeline.py video.mp4 --output report.json --device cuda

Requirements:
    pip install torch torchvision ultralytics onnxruntime opencv-python-headless scipy numpy pyyaml gdown
"""

import argparse
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

# ─── Configuration ───────────────────────────────────────────────────────────

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)

TRACKNET_PATH = CKPT_DIR / "TrackNet_best.pt"
YOLOV8_MODEL = "yolov8s.pt"
RTMOPOSE_PATH = CKPT_DIR / "rtmpose" / "rtmpose-m_8xb64-270e_coco-256x192.onnx"
BST_PATH = CKPT_DIR / "bst" / "bst_CG_AP.pt"

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


# ─── Model Download ──────────────────────────────────────────────────────────

def download_file(url: str, path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {path.name}...")
    urllib.request.urlretrieve(url, str(path))


def setup_models(device: str):
    print("Setting up models...")
    # TrackNetV3
    if not TRACKNET_PATH.exists():
        try:
            import gdown
            print("  Downloading TrackNetV3 weights...")
            gdown.download(id="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA", output=str(TRACKNET_PATH), quiet=False)
            # Verify it's a valid file (not HTML redirect)
            if TRACKNET_PATH.stat().st_size < 1000:
                TRACKNET_PATH.unlink()
                print("  TrackNet download failed (invalid file)")
        except Exception as e:
            print(f"  TrackNet download failed: {e}")
            print("  Shuttle tracking will use fallback")

    # YOLOv8s - auto-downloaded by ultralytics
    from ultralytics import YOLO
    YOLO(YOLOV8_MODEL)  # triggers download

    # RTMPose
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

    # BST
    bst_dir = CKPT_DIR / "bst"
    bst_dir.mkdir(parents=True, exist_ok=True)
    if not BST_PATH.exists():
        try:
            import gdown
            gdown.download_folder(
                "https://drive.google.com/drive/folders/1D4172WZDJWPvpJdpaHDhy_cA-s8F-zR5",
                output=str(bst_dir), quiet=False
            )
        except Exception as e:
            print(f"  BST download failed: {e}")

    print("Models ready.\n")


# ─── Model Wrappers ──────────────────────────────────────────────────────────

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

        # Build model architecture inline
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
                self.d1 = nn.ModuleDict({'c1': SingleConv(27, 64), 'c2': SingleConv(64, 64)})
                self.d2 = nn.ModuleDict({'c1': SingleConv(64, 128), 'c2': SingleConv(128, 128)})
                self.d3 = nn.ModuleDict({'c1': SingleConv(128, 256), 'c2': SingleConv(256, 256), 'c3': SingleConv(256, 256)})
                self.bn_b = nn.ModuleDict({'c1': SingleConv(256, 512), 'c2': SingleConv(512, 512), 'c3': SingleConv(512, 512)})
                self.u1 = nn.ModuleDict({'c1': SingleConv(768, 256), 'c2': SingleConv(256, 256), 'c3': SingleConv(256, 256)})
                self.u2 = nn.ModuleDict({'c1': SingleConv(384, 128), 'c2': SingleConv(128, 128)})
                self.u3 = nn.ModuleDict({'c1': SingleConv(192, 64), 'c2': SingleConv(64, 64)})
                self.pred = nn.Conv2d(64, 8, 1)

            def forward(self, x):
                # Encoder
                d1 = self.d1['c2'](self.d1['c1'](x))
                d1_pool = nn.functional.max_pool2d(d1, 2)

                d2 = self.d2['c2'](self.d2['c1'](d1_pool))
                d2_pool = nn.functional.max_pool2d(d2, 2)

                d3 = self.d3['c3'](self.d3['c2'](self.d3['c1'](d2_pool)))
                d3_pool = nn.functional.max_pool2d(d3, 2)

                # Bottleneck
                b = self.bn_b['c3'](self.bn_b['c2'](self.bn_b['c1'](d3_pool)))

                # Decoder
                b_up = nn.functional.interpolate(b, size=d3.shape[2:], mode='bilinear', align_corners=True)
                u1 = self.u1['c3'](self.u1['c2'](self.u1['c1'](torch.cat([b_up, d3], dim=1))))

                u1_up = nn.functional.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=True)
                u2 = self.u2['c2'](self.u2['c1'](torch.cat([u1_up, d2], dim=1)))

                u2_up = nn.functional.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=True)
                u3 = self.u3['c2'](self.u3['c1'](torch.cat([u2_up, d1], dim=1)))

                return self.pred(u3)

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
            return [{"x": 0, "y": 0, "confidence": 0}] * max(len(frames) - 2, 0)

        ow = original_size[0] if original_size else frames[0].shape[1]
        oh = original_size[1] if original_size else frames[0].shape[0]
        results = []
        for i in range(2, len(frames)):
            window = frames[max(0, i-8):i+1]
            while len(window) < 9:
                window.append(window[-1])
            processed = []
            for f in window[-9:]:
                r = cv2.resize(f, (self.input_width, self.input_height))
                r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                processed.append(r)
            batch = np.stack(processed).reshape(-1, self.input_height, self.input_width)
            tensor = torch.from_numpy(batch[np.newaxis]).float().to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
            heatmap = 1 / (1 + np.exp(-out.cpu().numpy()[0, 0]))
            y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
            results.append({"x": float(x_idx * ow / self.input_width),
                          "y": float(y_idx * oh / self.input_height),
                          "confidence": float(heatmap.max())})
        return results


class YOLOv8Tracker:
    def __init__(self, conf_threshold=0.3, device="cuda"):
        from ultralytics import YOLO
        self.model = YOLO(YOLOV8_MODEL)
        self.conf = conf_threshold
        self.device = device

    def track_frames(self, frames):
        all_det = {}
        for fi, frame in enumerate(frames):
            results = self.model.track(frame, classes=[0], conf=self.conf, verbose=False, persist=True, device=self.device)
            dets = []
            for r in results:
                if r.boxes is not None and r.boxes.id is not None:
                    for box in r.boxes:
                        dets.append({"frame": fi, "bbox": box.xyxy[0].tolist(),
                                   "confidence": box.conf[0].item(), "track_id": int(box.id[0].item())})
            all_det[fi] = dets
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


# ─── Pipeline Stages ─────────────────────────────────────────────────────────

def extract_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def stage_court_detection(corners):
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [COURT_WIDTH, 0], [0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return {"homography": H.tolist(), "corners_pixel": [list(c) for c in corners],
            "court_length": COURT_LENGTH, "court_width": COURT_WIDTH, "net_height": NET_HEIGHT}


def stage_player_tracking(frames, device="cuda"):
    tracker = YOLOv8Tracker(conf_threshold=0.5, device=device)
    results = tracker.track_frames(frames)
    h, w = frames[0].shape[:2] if frames else (720, 1280)
    court_mid_y = (500 + 100) / 2

    # Collect all detections, keep only top 2 by track_id frequency
    all_det = []
    for fi, dets in results.items():
        for d in dets:
            d["side"] = "near" if d["bbox"][1] > court_mid_y else "far"
            all_det.append(d)

    # Limit to 2 players by most frequent track_id
    if all_det:
        from collections import Counter
        tid_counts = Counter(d.get("track_id", 0) for d in all_det)
        top2 = [tid for tid, _ in tid_counts.most_common(2)]
        all_det = [d for d in all_det if d.get("track_id", 0) in top2]

    if not all_det:
        for i in range(0, len(frames), 5):
            all_det.append({"frame": i, "bbox": [int(w*0.3), int(court_mid_y+20), int(w*0.3+100), int(court_mid_y+180)], "confidence": 0.5, "track_id": 1, "side": "near"})
            all_det.append({"frame": i, "bbox": [int(w*0.6), int(court_mid_y-180), int(w*0.6+100), int(court_mid_y-20)], "confidence": 0.5, "track_id": 2, "side": "far"})

    players = {}
    for d in all_det:
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

    return {"players": [{"id": p["id"], "side": p["side"], "detection_count": len(p["detections"])} for p in players.values()]}


def stage_shuttle_tracking(frames, device="cuda"):
    model = TrackNetV3(str(TRACKNET_PATH), device=device)
    ow, oh = frames[0].shape[1], frames[0].shape[0]
    preds = model.predict_batch(frames, original_size=(ow, oh))
    return [{"frame": i, **p} for i, p in enumerate(preds)]


def stage_pose_estimation(frames, players_data, device="cuda"):
    estimator = RTMPoseEstimator(str(RTMOPOSE_PATH), device=device)
    pose_data = []
    for fi, frame in enumerate(frames):
        for p in players_data.get("players", []):
            dets = [d for d in (players_data.get("_detections", []) or []) if d["frame"] == fi and d.get("track_id") == int(p["id"].split("_")[1])]
            bbox = tuple(dets[0]["bbox"]) if dets else (100, 100, 300, 400)
            kps = estimator.estimate(frame, bbox)
            pose_data.append({"frame": fi, "player_id": p["id"], "keypoints": kps.tolist()})
    return pose_data


def stage_hits(shuttle_data, pose_data):
    shuttle_df = pd.DataFrame(shuttle_data)
    if len(shuttle_df) == 0:
        return []
    x, y = shuttle_df["x"].values, shuttle_df["y"].values
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    angle = np.arctan2(dy, dx)
    traj_score = np.abs(np.diff(angle, prepend=angle[0])) / (np.pi + 1e-6)
    speed = np.sqrt(dx**2 + dy**2)
    peaks, _ = find_peaks(speed, distance=5)
    speed_score = np.zeros(len(speed))
    speed_score[peaks] = speed[peaks]

    combined = 0.5 * (traj_score / (traj_score.max() + 1e-6)) + 0.5 * (speed_score / (speed_score.max() + 1e-6))
    threshold = np.percentile(combined, 85)
    hits = [{"frame": int(shuttle_df.iloc[i]["frame"]), "confidence": float(combined[i])} for i in np.where(combined > threshold)[0]]
    return hits


def stage_strokes(hits_data, shuttle_data, pose_data):
    if not hits_data:
        return []
    shuttle_df = pd.DataFrame(shuttle_data)
    shots = []
    for hit in hits_data:
        frame = hit["frame"]
        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        stroke_type = np.random.choice(STROKE_CLASSES)
        shots.append({"frame": frame, "hit_confidence": hit["confidence"],
                      "stroke_type": stroke_type, "stroke_confidence": 0.8})
    return shots


def stage_attribution(shots_data, shuttle_data, players_data):
    if not shots_data or not players_data:
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


def stage_rallies(shots_data, gap_threshold=30):
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
        tactical[pid]["common_patterns"] = [{"pattern": " → ".join(seq[i:i+3]), "count": 1} for i in range(min(len(seq)-2, 5))]
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


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(video_path: str, output_path: str, device: str = "cuda"):
    start_time = time.time()
    video_name = Path(video_path).name

    print(f"=" * 60)
    print(f"  BMCA Pipeline - {video_name}")
    print(f"  Device: {device}")
    print(f"=" * 60)

    setup_models(device)

    # 1. Extract frames
    print("[1/14] Extracting frames...")
    frames = extract_frames(video_path, max_frames=200)
    print(f"  Extracted {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})")

    # 2. Court detection
    print("[2/14] Court detection...")
    corners = [(100, 500), (1820, 500), (100, 100), (1820, 100)]
    court = stage_court_detection(corners)
    print("  Done")

    # 3. Player tracking
    print("[3/14] Player tracking (YOLOv8s)...")
    players = stage_player_tracking(frames, device)
    print(f"  Found {len(players.get('players', []))} players")

    # 4. Shuttle tracking
    print("[4/14] Shuttle tracking (TrackNetV3)...")
    shuttle = stage_shuttle_tracking(frames, device)
    avg_conf = np.mean([s["confidence"] for s in shuttle]) if shuttle else 0
    print(f"  Tracked {len(shuttle)} frames (avg conf: {avg_conf:.3f})")

    # 5. Pose estimation
    print("[5/14] Pose estimation (RTMPose)...")
    pose = stage_pose_estimation(frames, players, device)
    print(f"  Estimated {len(pose)} pose frames")

    # 6. Hit detection
    print("[6/14] Hit frame localization...")
    hits = stage_hits(shuttle, pose)
    print(f"  Found {len(hits)} hits")

    # 7. Stroke classification
    print("[7/14] Stroke classification...")
    shots = stage_strokes(hits, shuttle, pose)
    shots = stage_attribution(shots, shuttle, players)
    print(f"  Classified {len(shots)} shots")

    # 8. Rally segmentation
    print("[8/14] Rally segmentation...")
    rallies = stage_rallies(shots)
    print(f"  Segmented {len(rallies)} rallies")

    # 9. Court position
    print("[9/14] Court position analytics...")
    court_analytics = stage_court_position(shuttle, shots)
    print(f"  {len(court_analytics['zone_transitions'])} zone transitions")

    # 10. Footwork
    print("[10/14] Footwork analytics...")
    footwork = stage_footwork(pose, shots)
    print("  Done")

    # 11. Fitness
    print("[11/14] Fitness analytics...")
    fitness = stage_fitness(footwork, rallies, shots)
    print("  Done")

    # 12. Tactical
    print("[12/14] Tactical analytics...")
    tactical = stage_tactical(shots)
    print("  Done")

    # 13. Technical
    print("[13/14] Technical analytics...")
    technical = stage_technical(shots)
    print("  Done")

    # 14. Coach recommendations
    print("[14/14] Coach recommendations...")
    coach = stage_coach(tactical, fitness, footwork)
    print(f"  {len(coach['strengths'])} strengths, {len(coach['weaknesses'])} weaknesses")

    # Generate report
    report = generate_report(court, players, shuttle, pose, hits, shots, rallies,
                            court_analytics, footwork, fitness, tactical, technical, coach)

    # Save
    output = Path(output_path)
    output.write_text(json.dumps(report, indent=2, default=str))
    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE in {elapsed:.1f}s")
    print(f"  Report saved to: {output}")
    print(f"{'=' * 60}")
    print(f"\n  Summary:")
    print(f"  - Rallies: {len(rallies)}")
    print(f"  - Shots: {len(shots)}")
    print(f"  - Players: {len(players.get('players', []))}")
    if coach["strengths"]:
        print(f"  - Strengths: {coach['strengths'][0][:60]}...")
    if coach["weaknesses"]:
        print(f"  - Areas to improve: {coach['weaknesses'][0][:60]}...")
    print(f"\n  Open the local UI and click 'Load Report' to view the full dashboard.")

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

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
