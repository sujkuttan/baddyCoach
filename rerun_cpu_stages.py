#!/usr/bin/env python3
"""Re-run CPU pipeline stages with debug_level=3 using saved parquet data."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.pipeline.base import StageConfig
from app.pipeline.strokes import StrokeClassificationStage
from app.pipeline.attribution import PlayerAttributionStage
from app.pipeline.rallies import RallySegmentationStage
from app.storage.artifacts import ArtifactStore

DEBUG_DIR = Path("results/mmpose_results/debug")
OUTPUT_DIR = Path("results/rerun_with_fixes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load parquet data from previous run
hits_df = pd.read_parquet(DEBUG_DIR / "hits.parquet")
shuttle_df = pd.read_parquet(DEBUG_DIR / "shuttle.parquet")
pose_df = pd.read_parquet(DEBUG_DIR / "pose.parquet")
player_dets_df = pd.read_parquet(DEBUG_DIR / "player_detections.parquet")

print(f"Hits: {len(hits_df)}")
print(f"Shuttle: {len(shuttle_df)}")
print(f"Pose: {len(pose_df)}")
print(f"Player detections: {len(player_dets_df)}")

# Infer video resolution from shuttle max
vid_w = max(float(shuttle_df["x"].max()) * 2, 1280)
vid_h = max(float(shuttle_df["y"].max()) * 2, 720)

# Build players data with full detections for bbox interpolation
players_data = {"players": []}
for side_label, pid in [("near", "player_1"), ("far", "player_2")]:
    p_side_df = player_dets_df[player_dets_df["side"] == side_label]
    detections = []
    for _, row in p_side_df.iterrows():
        detections.append({
            "frame": int(row["frame"]),
            "bbox": list(row["bbox"]),
            "confidence": float(row["confidence"]),
            "track_id": int(row["track_id"]),
        })
    players_data["players"].append({
        "id": pid,
        "side": side_label,
        "detection_count": len(detections),
        "detections": detections,
    })

# Build court data (defaults, homography not available)
court_data = {
    "court_length": 13.4,
    "court_width": 6.1,
    "valid": True,
    "homography": None,
}

with tempfile.TemporaryDirectory() as tmpdir:
    store = ArtifactStore(Path(tmpdir))

    store.set("court", court_data)
    store.set("video_resolution", {"width": vid_w, "height": vid_h})
    store.set("players", players_data)
    store.set_parquet("shuttle", shuttle_df)
    store.set_parquet("pose", pose_df)
    store.set_parquet("hits", hits_df)

    config = StageConfig(gpu_enabled=False, debug_level=3)
    config.extra["bst_batch"] = 128

    # Stroke Classification
    print("\n=== Stroke Classification ===")
    result = StrokeClassificationStage().run(store, config)
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)
    shots_df = store.get_parquet("shots")
    if shots_df is not None:
        print(f"Shots: {len(shots_df)}")
        shots_df.to_parquet(OUTPUT_DIR / "shots.parquet", index=False)

    # Player Attribution
    print("\n=== Player Attribution ===")
    result = PlayerAttributionStage().run(store, config)
    print(f"Status: {result.status}")
    shots_df = store.get_parquet("shots")
    if shots_df is not None:
        shots_df.to_parquet(OUTPUT_DIR / "shots.parquet", index=False)

    # Rally Segmentation
    print("\n=== Rally Segmentation ===")
    result = RallySegmentationStage().run(store, config)
    print(f"Status: {result.status}")
    rallies_df = store.get_parquet("rallies")
    if rallies_df is not None:
        rallies_df.to_parquet(OUTPUT_DIR / "rallies.parquet", index=False)

    # Save debug outputs
    for f in Path(tmpdir).iterdir():
        if f.suffix == ".parquet" and f.name not in ("shuttle.parquet", "pose.parquet", "hits.parquet"):
            import shutil
            shutil.copy(f, OUTPUT_DIR / f.name)
            print(f"Debug: {f.name}")

print(f"\nOutputs saved to {OUTPUT_DIR}")

# Quick analysis
if shots_df is not None:
    bst = shots_df[~shots_df["is_rule_based"]]
    rb = shots_df[shots_df["is_rule_based"]]
    print(f"\n=== Results ===")
    print(f"Total shots: {len(shots_df)}")
    print(f"BST shots: {len(bst)} ({len(bst)/len(shots_df)*100:.0f}%)")
    print(f"Rule-based: {len(rb)} ({len(rb)/len(shots_df)*100:.0f}%)")
    print(f"BST types: {bst['stroke_type'].value_counts().to_dict()}")
    print(f"Rule-based types: {rb['stroke_type'].value_counts().to_dict()}")
    print(f"Mean confidence BST: {bst['stroke_confidence'].mean():.3f}")
    print(f"Mean confidence RB: {rb['stroke_confidence'].mean():.3f}")

    if "clip_n_frames" in shots_df.columns:
        print(f"\nClip stats:")
        print(f"  Missing bbox: mean={shots_df['clip_n_missing_bbox'].mean():.1f}, "
              f"median={shots_df['clip_n_missing_bbox'].median():.1f}")
        print(f"  Missing pose: mean={shots_df['clip_n_missing_pose'].mean():.1f}, "
              f"median={shots_df['clip_n_missing_pose'].median():.1f}")

if rallies_df is not None:
    print(f"\nRallies: {len(rallies_df)}")
    print(f"End reasons: {rallies_df['end_reason'].value_counts().to_dict()}")
