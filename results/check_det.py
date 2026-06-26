import pandas as pd

det_df = pd.read_parquet("player_detections.parquet")

f = 278
window = det_df[(det_df["frame"] >= f - 15) & (det_df["frame"] <= f + 15)]

print("Detections across the window:")
window = window.sort_values(["frame", "track_id"])
for _, row in window.iterrows():
    print(f"Frame {row['frame']}: track_id={row['track_id']}, side={row['side']}, bbox={row['bbox']}")
