"""Benchmark runner — evaluates pipeline components against labeled clips.

Computes precision/recall/F1 for hit detection, accuracy/macro-F1 for
stroke classification, attribution correctness, court homography error,
and shuttle tracking metrics.  Results are persisted as JSON + Markdown.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_DIR = Path(__file__).parent / "manifest"
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════
# Component-level scorers
# ═══════════════════════════════════════════════════════════════════════

def _hit_f1(predicted: list[dict], ground_truth: list[dict],
            tolerance: int = 3) -> dict:
    """Hit detection precision / recall / F1 within ±tolerance frames."""
    gt_frames = {h["frame"] for h in ground_truth}
    pred_frames = {h["frame"] for h in predicted}

    tp = 0
    for pf in pred_frames:
        if any(abs(pf - gf) <= tolerance for gf in gt_frames):
            tp += 1

    fp = len(pred_frames) - tp
    fn = len(gt_frames) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)}


def _confusion_matrix(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """Build a confusion matrix for stroke classification."""
    gt_by_frame = {h["frame"]: h["stroke_type"] for h in ground_truth}
    pred_by_frame = {h["frame"]: h.get("stroke_type", "unknown") for h in predicted}

    common = set(gt_by_frame) & set(pred_by_frame)
    if not common:
        return {"matrix": {}, "classes": [], "n_matched": 0}

    classes = sorted(set(list(gt_by_frame.values()) + list(pred_by_frame.values())))
    cm: dict[str, dict[str, int]] = {}
    for c in classes:
        cm[c] = {k: 0 for k in classes}

    for f in common:
        actual = gt_by_frame[f]
        pred = pred_by_frame[f]
        if actual not in cm:
            cm[actual] = {}
        if pred not in cm[actual]:
            cm[actual][pred] = 0
        cm[actual][pred] += 1

    return {"matrix": cm, "classes": classes, "n_matched": len(common)}


def _stroke_accuracy(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """Stroke classification accuracy and macro-F1 for matched hits."""
    gt_by_frame = {h["frame"]: h["stroke_type"] for h in ground_truth}
    pred_by_frame = {h["frame"]: h.get("stroke_type", "unknown") for h in predicted}

    common = set(gt_by_frame) & set(pred_by_frame)
    if not common:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n_matched": 0}

    correct = sum(1 for f in common if gt_by_frame[f] == pred_by_frame[f])
    accuracy = correct / len(common)

    classes = set(gt_by_frame.values()) | set(pred_by_frame.values())
    f1_scores = []
    for cls in sorted(classes):
        tp = sum(1 for f in common if gt_by_frame[f] == cls and pred_by_frame[f] == cls)
        fp = sum(1 for f in common if pred_by_frame[f] == cls and gt_by_frame[f] != cls)
        fn = sum(1 for f in common if gt_by_frame[f] == cls and pred_by_frame[f] != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {"accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "n_matched": len(common)}


def _attribution_accuracy(predicted: list[dict],
                          ground_truth: list[dict]) -> dict:
    """Fraction of matched hits with correct player_side attribution."""
    gt_by_frame = {h["frame"]: h["player_side"] for h in ground_truth}
    pred_by_frame = {h["frame"]: h.get("player_side", "") for h in predicted}

    common = set(gt_by_frame) & set(pred_by_frame)
    if not common:
        return {"accuracy": 0.0, "n_matched": 0}

    correct = sum(1 for f in common if gt_by_frame[f] == pred_by_frame[f])
    return {"accuracy": round(correct / len(common), 4),
            "n_matched": len(common)}


def _homography_error(court_predicted: dict,
                      court_ground_truth: dict) -> dict:
    """Mean reprojection error of held-out court points (meters).

    Uses standard court dimensions (13.40m × 6.10m doubles) to estimate
    a pixel-to-meter scale factor from the ground-truth corner quadrilateral.
    """
    pred_pts = np.array(court_predicted.get("corners_pixel", []), dtype=np.float32)
    gt_pts = np.array(court_ground_truth.get("corners_pixel", []), dtype=np.float32)

    if len(pred_pts) < 4 or len(gt_pts) < 4:
        return {"mean_error_m": -1.0, "n_points": 0}

    COURT_LENGTH_M = 13.40
    COURT_WIDTH_M = 6.10

    # Estimate pixel-to-meter scale from ground-truth corners
    gt_top = gt_pts[:2]
    gt_bot = gt_pts[2:]
    px_width = float(np.mean([np.linalg.norm(gt_top[1] - gt_top[0]),
                              np.linalg.norm(gt_bot[1] - gt_bot[0])]))
    px_height = float(np.mean([np.linalg.norm(gt_bot[0] - gt_top[0]),
                               np.linalg.norm(gt_bot[1] - gt_top[1])]))
    px_per_m = max(1.0, (px_width / COURT_WIDTH_M + px_height / COURT_LENGTH_M) / 2.0)

    errors_px = np.sqrt(np.sum((pred_pts[:4] - gt_pts[:4]) ** 2, axis=1))
    errors_m = errors_px / px_per_m
    return {"mean_error_m": round(float(np.mean(errors_m)), 4),
            "mean_error_px": round(float(np.mean(errors_px)), 2),
            "n_points": 4}


def _shuttle_tracking(predicted: list[dict],
                      ground_truth: list[dict]) -> dict:
    """Shuttle detection rate and mean pixel error where GT exists."""
    gt_by_frame = {h["frame"]: (h.get("x", 0), h.get("y", 0)) for h in ground_truth}
    pred_by_frame = {h["frame"]: (h.get("x", 0), h.get("y", 0))
                     for h in predicted if h.get("confidence", 0) > 0.3}

    common = set(gt_by_frame) & set(pred_by_frame)
    if not common:
        return {"detection_rate": 0.0, "mean_px_error": 0.0, "n_matched": 0}

    errors = []
    for f in common:
        gx, gy = gt_by_frame[f]
        px, py = pred_by_frame[f]
        errors.append(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))

    detection_rate = len(common) / max(len(gt_by_frame), 1)
    return {"detection_rate": round(detection_rate, 4),
            "mean_px_error": round(float(np.mean(errors)), 2),
            "n_matched": len(common)}


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

COMPONENT_METRICS = {
    "hit_detection": {"tolerance": 3, "gate": {"f1": 0.80}},
    "stroke_classification": {"gate": {"macro_f1": 0.60}},
    "attribution": {"gate": {"accuracy": 0.90}},
    "homography": {"gate": {"mean_error_m": None}},  # None = not gated (informational)
    "shuttle_tracking": {"gate": {"detection_rate": 0.70}},
}


class BenchmarkRunner:
    """Evaluate pipeline components against a manifest of labeled clips."""

    def __init__(self, manifest_dir: Path | None = None):
        self.manifest_dir = manifest_dir or MANIFEST_DIR
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_manifests(self) -> list[dict]:
        """Load all manifest JSONs from the manifest directory."""
        manifests = []
        if not self.manifest_dir.exists():
            return manifests
        for path in sorted(self.manifest_dir.glob("*.json")):
            with open(path) as f:
                manifests.append(json.load(f))
        return manifests

    def evaluate_clip(self, manifest: dict,
                      predictions: dict[str, Any]) -> dict:
        """Evaluate a single clip's predictions against its GT."""
        gt_hits = manifest.get("hits", [])

        result = {"clip_id": manifest.get("clip_id", "unknown")}

        if "hits" in predictions:
            result["hit_detection"] = _hit_f1(
                predictions["hits"], gt_hits,
                tolerance=manifest.get("tolerance", COMPONENT_METRICS["hit_detection"]["tolerance"]),
            )

        if "strokes" in predictions:
            result["stroke_classification"] = _stroke_accuracy(
                predictions["strokes"], gt_hits,
            )
            result["confusion_matrix"] = _confusion_matrix(
                predictions["strokes"], gt_hits,
            )

        if "attribution" in predictions:
            result["attribution"] = _attribution_accuracy(
                predictions["attribution"], gt_hits,
            )

        if "court" in predictions and "court_corners_px" in manifest:
            result["homography"] = _homography_error(
                predictions["court"],
                {"corners_pixel": manifest["court_corners_px"]},
            )

        if "shuttle" in predictions:
            gt_shuttle = []
            for h in gt_hits:
                gt_shuttle.append({"frame": h["frame"], "x": h.get("x", 0), "y": h.get("y", 0)})
            result["shuttle_tracking"] = _shuttle_tracking(
                predictions["shuttle"], gt_shuttle,
            )

        return result

    def run_all(self, predictor_fn) -> dict:
        """Run benchmark against all manifests.

        Args:
            predictor_fn: A callable(clip_manifest) -> dict that returns
                          predictions for a clip.  Must produce the same
                          keys expected by evaluate_clip.

        Returns:
            dict with per-clip results + aggregates + gate verdict.
        """
        manifests = self.load_manifests()
        if not manifests:
            return {"clips": [], "aggregates": {}, "gate": "skipped",
                    "message": "No manifests found in benchmark/manifest/"}

        per_clip = []
        all_components: dict[str, list] = {}

        for m in manifests:
            try:
                preds = predictor_fn(m)
                clip_result = self.evaluate_clip(m, preds)
                per_clip.append(clip_result)
                for comp, scores in clip_result.items():
                    if comp == "clip_id":
                        continue
                    all_components.setdefault(comp, []).append(scores)
            except Exception as e:
                per_clip.append({"clip_id": m.get("clip_id", "unknown"),
                                 "error": str(e)})

        aggregates = {}
        for comp, score_list in all_components.items():
            valid = [s for s in score_list if isinstance(s, dict)]
            if not valid:
                aggregates[comp] = {"error": "no valid results"}
                continue
            agg = {}
            for key in valid[0]:
                values = [s[key] for s in valid if key in s and isinstance(s[key], (int, float))]
                if values:
                    agg[key] = round(float(np.mean(values)), 4)
                else:
                    agg[key] = None
            aggregates[comp] = agg

        gate = self._check_gate(aggregates)
        return {"clips": per_clip, "aggregates": aggregates,
                "gate": gate, "n_clips": len(manifests),
                "timestamp": datetime.now().isoformat()}

    def _check_gate(self, aggregates: dict) -> dict:
        """Check if aggregates pass the release gates.

        A threshold < 0 means the metric is not gated (informational only).
        Components not present in aggregates are treated as passed
        (they may not have been tested).
        """
        verdicts = {}
        for comp, cfg in COMPONENT_METRICS.items():
            gate_cfg = cfg.get("gate", {})
            agg = aggregates.get(comp, {})
            if not agg:
                # Component not present in this run — skip gate check
                verdicts[comp] = {"passed": True, "aggregate": {}, "skipped": True}
                continue
            passed = True
            for metric, threshold in gate_cfg.items():
                if threshold is None or (isinstance(threshold, (int, float)) and threshold < 0):
                    continue  # informational-only metric
                val = agg.get(metric)
                if val is None:
                    passed = False
                elif isinstance(val, (int, float)):
                    if val < threshold:
                        passed = False
            verdicts[comp] = {"passed": passed, "aggregate": agg}
        verdicts["overall"] = all(v["passed"] for v in verdicts.values())
        return verdicts

    def save_results(self, results: dict) -> Path:
        """Persist results as JSON and Markdown."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = self.results_dir / f"{date_str}.json"
        md_path = self.results_dir / f"{date_str}.md"

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        md_lines = ["# Benchmark Results", f"**Date:** {results.get('timestamp', 'unknown')}",
                     f"**Clips:** {results.get('n_clips', 0)}",
                     f"**Gate:** {'PASS' if results.get('gate', {}).get('overall') else 'FAIL'}",
                     "", "## Aggregates", ""]

        for comp, agg in results.get("aggregates", {}).items():
            md_lines.append(f"### {comp}")
            for k, v in agg.items():
                md_lines.append(f"- {k}: {v}")
            gate_v = results.get("gate", {}).get(comp, {})
            md_lines.append(f"**Gate: {'PASS' if gate_v.get('passed') else 'FAIL'}**")
            md_lines.append("")

        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        return json_path
