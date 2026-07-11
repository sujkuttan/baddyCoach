import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.config.settings import settings


FEATURE_COLUMNS = [
    "ownership_trajectory_near",
    "ownership_trajectory_far",
    "ownership_court_side_near",
    "ownership_court_side_far",
    "ownership_proximity_near",
    "ownership_proximity_far",
    "ownership_motion_near",
    "ownership_motion_far",
    "ownership_pose_near",
    "ownership_pose_far",
]


def compute_owner_metrics(matched: pd.DataFrame) -> dict[str, float | dict[str, int]]:
    assigned = matched[matched["pred_side"].isin(["near", "far"])].copy()
    coverage = len(assigned) / len(matched) if len(matched) else 0.0
    assigned_accuracy = float((assigned["pred_side"] == assigned["label_side"]).mean()) if len(assigned) else 0.0
    overall_accuracy = float((matched["pred_side"] == matched["label_side"]).mean()) if len(matched) else 0.0
    source_breakdown = (
        matched.groupby("owner_source")["pred_side"].count().to_dict()
        if "owner_source" in matched.columns
        else {}
    )
    return {
        "coverage": coverage,
        "assigned_accuracy": assigned_accuracy,
        "overall_accuracy": overall_accuracy,
        "abstention_rate": 1.0 - coverage,
        "source_breakdown": source_breakdown,
    }


def recommend_deploy(
    baseline: dict,
    candidate: dict,
    min_accuracy_lift: float,
    min_coverage_lift: float,
) -> dict[str, float | bool]:
    accuracy_lift = candidate["assigned_accuracy"] - baseline["assigned_accuracy"]
    coverage_lift = candidate["coverage"] - baseline["coverage"]
    return {
        "deploy": accuracy_lift >= min_accuracy_lift and coverage_lift >= min_coverage_lift,
        "accuracy_lift": accuracy_lift,
        "coverage_lift": coverage_lift,
    }


def load_and_match(shots_path: str, labels_path: str, match_tolerance: int) -> pd.DataFrame:
    shots = pd.read_parquet(shots_path).sort_values("frame").reset_index(drop=True)
    labels = pd.read_csv(labels_path).sort_values("frame").reset_index(drop=True)

    unmatched = set(shots.index)
    rows = []
    for _, label in labels.iterrows():
        best_idx = None
        best_dist = None
        for idx in unmatched:
            dist = abs(int(shots.loc[idx, "frame"]) - int(label["frame"]))
            if dist <= match_tolerance and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            continue
        unmatched.remove(best_idx)
        shot = shots.loc[best_idx]
        row = {"label_side": label["side"], "pred_side": shot.get("side", "unknown"), "owner_source": shot.get("owner_source", "unknown")}
        for col in FEATURE_COLUMNS:
            row[col] = shot.get(col)
        row["rally_id"] = shot.get("rally_id")
        rows.append(row)
    return pd.DataFrame(rows)


def run_leave_one_rally_out_calibration(matched: pd.DataFrame, feature_columns: list[str]) -> dict[str, float | dict[str, int]]:
    if matched.empty:
        return {"coverage": 0.0, "assigned_accuracy": 0.0, "overall_accuracy": 0.0, "abstention_rate": 1.0, "source_breakdown": {}}

    usable = matched.dropna(subset=feature_columns).copy()
    usable = usable[usable["label_side"].isin(["near", "far"])].copy()
    if usable.empty or usable["label_side"].nunique() < 2:
        baseline = matched.copy()
        baseline["pred_side"] = baseline["pred_side"].fillna("unknown")
        return compute_owner_metrics(baseline)

    rally_ids = [rid for rid in usable["rally_id"].dropna().unique().tolist()]
    if not rally_ids:
        rally_ids = [0]
        usable["rally_id"] = 0

    preds = []
    for rally_id in rally_ids:
        train = usable[usable["rally_id"] != rally_id]
        test = usable[usable["rally_id"] == rally_id]
        if train.empty or test.empty or train["label_side"].nunique() < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(train[feature_columns], train["label_side"])
        pred = model.predict(test[feature_columns])
        fold = test[["label_side", "owner_source"]].copy()
        fold["pred_side"] = pred
        preds.append(fold)

    if not preds:
        fallback = usable[["label_side", "owner_source"]].copy()
        fallback["pred_side"] = matched["pred_side"].iloc[: len(fallback)].tolist()
        return compute_owner_metrics(fallback)

    predicted = pd.concat(preds, ignore_index=True)
    return compute_owner_metrics(predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-tolerance", type=int, default=settings.ownership_calibration_match_tolerance_frames)
    args = parser.parse_args()

    matched = load_and_match(args.shots, args.labels, args.match_tolerance)
    baseline = compute_owner_metrics(matched.rename(columns={"pred_side": "pred_side", "label_side": "label_side"}))
    candidate = run_leave_one_rally_out_calibration(matched, FEATURE_COLUMNS)
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "recommendation": recommend_deploy(
            baseline,
            candidate,
            settings.ownership_calibration_min_accuracy_lift,
            settings.ownership_calibration_min_coverage_lift,
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
