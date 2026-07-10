import csv, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

def load_labels(path: str, time_multiplier: float = 1.0, fps: float = 30.0):
    """Load manual labels CSV.
    
    Args:
        path: Path to CSV with columns: Time,Player,Stroke,Rally
        time_multiplier: Multiply Time column by this to get seconds.
                         E.g., 1.0 if Time is already seconds,
                               60.0 if Time is decimal minutes.
        fps: Video frame rate for converting seconds to frames.
    """
    labels = []
    rally_id_counter = 0
    current_rally = None
    
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_raw = float(row["Time"].strip())
            t_sec = t_raw * time_multiplier
            rally_cell = row["Rally"].strip()
            
            if rally_cell.lower() in ("rally end", "end"):
                pass  # still count the shot; rally boundary is implicit
            elif rally_cell.isdigit():
                current_rally = int(rally_cell)
            
            labels.append({
                "time_s": t_sec,
                "frame": int(round(t_sec * fps)),
                "player": row["Player"].strip(),
                "stroke": row["Stroke"].strip(),
                "rally": current_rally,
            })
    return labels


def load_pipeline_shots(parquet_path: str, fps: float = 30.0):
    shots = pd.read_parquet(parquet_path)
    shots = shots.sort_values("frame").reset_index(drop=True)
    shots["time_s"] = shots["frame"] / fps
    return shots


def match_labels_to_shots(
    labels: list,
    shots: pd.DataFrame,
    radius_frames: int = 15,
) -> dict:
    """Match each label to the nearest pipeline shot within radius_frames.
    
    Each pipeline shot can match at most one label (greedy nearest).
    Returns dict with matches, unmatched_labels, false_positive_shots.
    """
    shots_unused = set(shots.index)
    matches = []

    for lab in labels:
        best_idx = None
        best_dist = float("inf")
        for idx in shots_unused:
            dist = abs(shots.loc[idx, "frame"] - lab["frame"])
            if dist < best_dist and dist <= radius_frames:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            matches.append({
                "label": lab,
                "shot_idx": best_idx,
                "shot": shots.loc[best_idx].to_dict(),
                "frame_error": int(best_dist),
            })
            shots_unused.remove(best_idx)
        else:
            matches.append({
                "label": lab,
                "shot_idx": None,
                "shot": None,
                "frame_error": None,
            })

    false_positives = [
        {"idx": idx, "shot": shots.loc[idx].to_dict()}
        for idx in sorted(shots_unused)
    ]
    return {
        "matches": matches,
        "false_positives": false_positives,
        "n_labels": len(labels),
        "n_pipeline": len(shots),
        "n_matched": sum(1 for m in matches if m["shot_idx"] is not None),
        "n_missed": sum(1 for m in matches if m["shot_idx"] is None),
        "n_false_positives": len(false_positives),
    }


# ── Stroke taxonomy mapping ──
STROKE_SIMILARITY = {
    "lift": ["lift", "defensive_lift", "clear"],
    "clear": ["clear", "lift", "defensive_lift"],
    "drop": ["drop", "net_shot"],
    "net shot": ["net_shot", "drop", "push"],
    "netshot": ["net_shot", "drop", "push"],
    "drive": ["drive", "push", "rush"],
    "smash": ["smash", "rush", "drive"],
    "serve": ["short_serve", "long_serve", "serve"],
    "push": ["push", "drive", "net_shot"],
    "rush": ["rush", "drive", "smash", "push"],
    "block": ["block", "drive", "net_shot"],
    "defensive lift": ["lift", "defensive_lift", "clear"],
    "soft lift or push": ["soft_lift_or_push", "lift", "push"],
    "cross court": ["cross_court", "drive", "clear"],
}

def normalize_stroke(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "")


def stroke_matches(pipeline_stroke: str, label_stroke: str) -> str:
    """Check if pipeline stroke matches label stroke.
    Returns 'exact', 'similar', or 'wrong'.
    """
    p_norm = normalize_stroke(pipeline_stroke)
    l_norm = normalize_stroke(label_stroke)
    
    if p_norm == l_norm:
        return "exact"
    
    # Check similarity map
    for key, alternatives in STROKE_SIMILARITY.items():
        key_norm = normalize_stroke(key)
        alt_norm = [normalize_stroke(a) for a in alternatives]
        if l_norm == key_norm and p_norm in alt_norm:
            return "similar"
        if p_norm == key_norm and l_norm in alt_norm:
            return "similar"
    
    return "wrong"


def compute_metrics(result: dict) -> dict:
    matches = result["matches"]
    
    stroke_results = []
    player_results = []
    frame_errors = []
    
    for m in matches:
        if m["shot"] is None:
            stroke_results.append("missed")
            player_results.append("missed")
            continue
        
        s_match = stroke_matches(m["shot"]["stroke_type"], m["label"]["stroke"])
        stroke_results.append(s_match)
        
        p_match = "correct" if m["shot"]["side"] == m["label"]["player"].lower() else "wrong"
        player_results.append(p_match)
        
        frame_errors.append(m["frame_error"])
    
    n_matched = result["n_matched"]
    stroke_exact = sum(1 for s in stroke_results if s == "exact")
    stroke_similar = sum(1 for s in stroke_results if s == "similar")
    stroke_correct = stroke_exact + stroke_similar
    player_correct = sum(1 for p in player_results if p == "correct")
    
    return {
        "n_labels": result["n_labels"],
        "n_pipeline": result["n_pipeline"],
        "n_matched": n_matched,
        "n_missed": result["n_missed"],
        "n_false_positives": result["n_false_positives"],
        "stroke_exact": stroke_exact,
        "stroke_similar": stroke_similar,
        "stroke_accuracy": stroke_correct / n_matched * 100 if n_matched else 0,
        "stroke_accuracy_strict": stroke_exact / n_matched * 100 if n_matched else 0,
        "player_accuracy": player_correct / n_matched * 100 if n_matched else 0,
        "mean_frame_error": np.mean(frame_errors) if frame_errors else 0,
        "median_frame_error": np.median(frame_errors) if frame_errors else 0,
        "recall": n_matched / result["n_labels"] * 100 if result["n_labels"] else 0,
        "precision": n_matched / result["n_pipeline"] * 100 if result["n_pipeline"] else 0,
    }


def print_report(result: dict, metrics: dict, show_details: bool = True):
    print("=" * 70)
    print("  LABEL EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n  Coverage:")
    print(f"    Labels:          {metrics['n_labels']}")
    print(f"    Pipeline shots:  {metrics['n_pipeline']}")
    print(f"    Matched:         {metrics['n_matched']} ({metrics['recall']:.0f}% recall)")
    print(f"    Missed:          {metrics['n_missed']} (label has no nearby pipeline shot)")
    print(f"    False positives: {metrics['n_false_positives']} (pipeline shot with no nearby label)")
    
    print(f"\n  Stroke Accuracy:")
    print(f"    Exact match:     {metrics['stroke_exact']}/{metrics['n_matched']} ({metrics['stroke_accuracy_strict']:.0f}%)")
    print(f"    Similar match:   {metrics['stroke_similar']}/{metrics['n_matched']} (+{metrics['stroke_similar']/metrics['n_matched']*100:.0f}%)")
    print(f"    Combined:        {metrics['stroke_accuracy']:.0f}%")
    
    print(f"\n  Player Attribution:")
    print(f"    Correct:         {metrics['player_accuracy']:.0f}%")
    
    print(f"\n  Temporal Alignment:")
    print(f"    Mean frame error: {metrics['mean_frame_error']:.1f} frames")
    print(f"    Median frame err: {metrics['median_frame_error']:.0f} frames")
    
    if not show_details:
        return
    
    # Per-shot breakdown
    print(f"\n{'─' * 95}")
    print(f"{'#':>3s} {'Label Time':>10s} {'Frame':>6s} {'Player':>7s} {'Stroke':>18s} {'Pipe Time':>10s} {'Frame':>6s} {'Player':>7s} {'Stroke':>20s} {'Match':>10s} {'Err':>4s}")
    print(f"{'─' * 95}")
    
    for i, m in enumerate(result["matches"]):
        label = m["label"]
        shot = m["shot"]
        fe = m["frame_error"]
        
        if shot is None:
            s_match = "MISSED"
        else:
            s_match = stroke_matches(shot["stroke_type"], label["stroke"])
            p_correct = "✓" if shot["side"] == label["player"].lower() else "✗"
            s_match_display = {
                "exact": f"✓{s_match.upper():>6s}",
                "similar": f"~{s_match.upper():>6s}",
                "wrong": f"✗{s_match.upper():>6s}",
            }.get(s_match, f" {s_match:>7s}")
        
        pipe_time = f"{shot['time_s']:.2f}s" if shot is not None else "—"
        pipe_frame = f"{int(shot['frame'])}" if shot is not None else "—"
        pipe_player = f"{shot['side']}" if shot is not None else "—"
        pipe_stroke = f"{shot['stroke_type']}" if shot is not None else "—"
        pipe_conf = f"{shot['stroke_confidence']:.2f}" if shot is not None else "—"
        fe_str = f"{fe}" if fe is not None else "—"
        
        print(f"{i+1:3d}  {label['time_s']:8.1f}s  {label['frame']:5d}  {label['player']:>6s}  {label['stroke']:>18s}  {pipe_time:>9s}  {pipe_frame:>5s}  {pipe_player:>6s}  {pipe_stroke:>20s}  {s_match_display:>10s}  {fe_str:>3s}  {pipe_conf}")
    
    # False positives
    if result["false_positives"]:
        print(f"\n  False positives ({len(result['false_positives'])}):")
        for fp in result["false_positives"][:10]:
            s = fp["shot"]
            print(f"    t={s['time_s']:7.2f}s  frame={s['frame']:4d}  {s['side']:5s}  {s['stroke_type']:20s}  conf={s['stroke_confidence']:.2f}")
        if len(result["false_positives"]) > 10:
            print(f"    ... and {len(result['false_positives']) - 10} more")


def summarize_bst_input_quality(shots: pd.DataFrame) -> dict:
    """Summarize coverage and strict accuracy for manually labeled shots."""
    labeled = shots.dropna(subset=["true_stroke"]).copy()
    eligible = labeled[labeled["bst_input_eligible"].fillna(False)]
    correct = labeled["stroke_type"] == labeled["true_stroke"]
    accepted_correct = eligible["stroke_type"] == eligible["true_stroke"]
    reason_counts = {}
    for reasons in labeled.get("bst_input_quality_reasons", pd.Series(dtype=object)):
        if isinstance(reasons, np.ndarray):
            reasons = reasons.tolist()
        for reason in reasons if isinstance(reasons, list) else []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total_labeled": len(labeled),
        "eligible_labeled": len(eligible),
        "coverage": len(eligible) / max(1, len(labeled)),
        "accepted_accuracy": float(accepted_correct.mean()) if len(eligible) else 0.0,
        "overall_accuracy": float(correct.mean()) if len(labeled) else 0.0,
        "reason_counts": reason_counts,
    }


def evaluate_enriched_csv(csv_path: str, max_frame_diff: int = 15) -> dict:
    """Evaluate using already-matched enriched CSV (labels_enriched.csv format)."""
    df = pd.read_csv(csv_path)
    df = df[df["label_status"] == "labeled"].copy()
    
    shots = load_pipeline_shots("results/hybrid_results/debug/shots.parquet")
    
    # Build label list from enriched CSV
    labels = []
    for _, row in df.iterrows():
        # Find actual shot by matching shot_frame
        lf = int(row["shot_frame"]) if "shot_frame" in row and not pd.isna(row.get("shot_frame")) else int(row["label_frame"])
        labels.append({
            "time_s": lf / 30.0,
            "frame": lf,
            "player": str(row["side"]).strip(),
            "stroke": str(row["true_stroke"]).strip(),
        })
    
    result = match_labels_to_shots(labels, shots, radius_frames=max_frame_diff)
    # Override frame diff with the pre-computed value from the enriched CSV
    if "frame_diff" in df.columns:
        for i, m in enumerate(result["matches"]):
            if i < len(df) and m["shot_idx"] is not None:
                m["frame_error"] = int(df.iloc[i]["frame_diff"])
    
    metrics = compute_metrics(result)
    if "bst_input_eligible" in shots.columns:
        matched_rows = []
        for match in result["matches"]:
            if match["shot"] is None:
                continue
            row = dict(match["shot"])
            row["true_stroke"] = match["label"]["stroke"]
            matched_rows.append(row)
        metrics["bst_input_quality"] = summarize_bst_input_quality(pd.DataFrame(matched_rows))
    # Override with enriched frame diffs
    if "frame_diff" in df.columns:
        frame_diffs = [int(d) for d in df["frame_diff"] if not pd.isna(d)]
        if frame_diffs:
            metrics["mean_frame_error"] = np.mean(frame_diffs)
            metrics["median_frame_error"] = np.median(frame_diffs)
    
    return result, metrics


def main():
    """Usage: python evaluate_labels.py [labels.csv|labels_enriched.csv] [time_multiplier] [radius_frames]"""
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "labels_enriched.csv"
    time_mult = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    print(f"  Labels file: {labels_path}")
    
    # Detect enriched format (has label_status column)
    df_check = pd.read_csv(labels_path, nrows=1)
    is_enriched = "label_status" in df_check.columns and "true_class_id" in df_check.columns
    
    if is_enriched:
        print(f"  Detected enriched CSV format (pre-matched labels)")
        result, metrics = evaluate_enriched_csv(labels_path, max_frame_diff=radius)
    else:
        print(f"  Time multiplier: {time_mult}x")
        print(f"  Match radius: {radius} frames ({radius/30:.1f}s @30fps)")
        labels = load_labels(labels_path, time_multiplier=time_mult)
        shots = load_pipeline_shots("results/hybrid_results/debug/shots.parquet")
        result = match_labels_to_shots(labels, shots, radius_frames=radius)
        metrics = compute_metrics(result)
    
    print_report(result, metrics, show_details=True)
    if "bst_input_quality" in metrics:
        quality = metrics["bst_input_quality"]
        print("\n  BST Input Quality:")
        print(f"    Coverage:          {quality['coverage']:.1%}")
        print(f"    Accepted accuracy: {quality['accepted_accuracy']:.1%}")
        print(f"    Overall accuracy:  {quality['overall_accuracy']:.1%}")
        print(f"    Abstention reasons: {quality['reason_counts']}")
    
    print(f"\n{'─' * 70}")
    print(f"  Interpretation notes:")
    print(f"  - Stroke 'similar' uses a similarity map (e.g., lift↔clear, net_shot↔drop)")
    print(f"  - Frame error > 15 frames (~0.5s) counts as 'missed'")
    print(f"  - False positives may be pipeline over-detecting or unlabeled shots")
    print(f"  - Missed labels may be pipeline under-detecting or tracking failures")


if __name__ == "__main__":
    main()
