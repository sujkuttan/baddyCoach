#!/usr/bin/env python3
"""CLI benchmark runner.

Usage:
    python scripts/run_benchmark.py                                  # run all manifests
    python scripts/run_benchmark.py --manifest manifest/rally_001.json  # single manifest
    python scripts/run_benchmark.py --list                             # list available manifests

Pipeline predictions are loaded from the corresponding ``results/<clip_id>_pred.json``
files produced by running the pipeline manually on each clip.
"""

import argparse
import json
from pathlib import Path


def _load_predictions(clip_id: str, results_dir: Path) -> dict | None:
    """Load pre-computed predictions for a clip."""
    pred_path = results_dir / f"{clip_id}_pred.json"
    if pred_path.exists():
        with open(pred_path) as f:
            return json.load(f)
    return None


def _predictor_fn(manifest: dict,
                  results_dir: Path = Path("benchmarks/results")) -> dict:
    """Default predictor: load pre-computed predictions from results dir."""
    clip_id = manifest.get("clip_id", "unknown")
    preds = _load_predictions(clip_id, results_dir)
    if preds is None:
        raise FileNotFoundError(f"No predictions found for {clip_id} "
                                f"(expected {results_dir}/{clip_id}_pred.json)")
    return preds


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluations")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to a single manifest file")
    parser.add_argument("--list", action="store_true",
                        help="List available manifests and exit")
    args = parser.parse_args()

    from benchmarks import BenchmarkRunner

    runner = BenchmarkRunner()

    if args.list:
        manifests = runner.load_manifests()
        if not manifests:
            print("No manifests found.")
        else:
            print("Available manifests:")
            for m in manifests:
                print(f"  {m.get('clip_id', 'unknown')}: {m.get('video', '?')}")
        return

    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        results = {"clips": [runner.evaluate_clip(manifest, _predictor_fn(manifest))],
                   "n_clips": 1, "timestamp": __import__("datetime").datetime.now().isoformat()}
    else:
        results = runner.run_all(_predictor_fn)

    path = runner.save_results(results)
    print(f"Results saved to {path}")
    print(f"Gate: {'PASS' if results.get('gate', {}).get('overall') else 'FAIL'}")

    for comp, agg in results.get("aggregates", {}).items():
        print(f"  {comp}: {agg}")


if __name__ == "__main__":
    main()
