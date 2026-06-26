import pandas as pd
import numpy as np

pose_df = pd.read_parquet("pose.parquet")

f = 278
window = pose_df[(pose_df["frame"] >= f - 15) & (pose_df["frame"] <= f + 15)]

print("Player 2 keypoints across the window:")
p2_window = window[window["player_id"] == "player_2"].sort_values("frame")
for _, row in p2_window.iterrows():
    raw = row["keypoints"]
    kps = np.array(raw.tolist()) if hasattr(raw, "tolist") else np.array(raw)
    print(f"Frame {row['frame']} shape: {kps.shape}")
    if kps.ndim == 2 and kps.shape[0] >= 17:
        com = (kps[11, :2] + kps[12, :2]) / 2
        print(f"  COM = {com}")
