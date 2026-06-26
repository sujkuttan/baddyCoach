"""Build reference percentile tables from many pipeline sessions.

Aggregates technique features from ``data/jobs/*/technical_analytics.json``
across all sessions and writes reference tier files to ``data/reference/``.

Usage:
    python scripts/build_reference.py [--tier intermediate] [--jobs-dir data/jobs]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect_features(jobs_dir: Path) -> dict:
    """Scan all job dirs and collect per-stroke feature p50 values.

    Returns:
        Dict[stroke_type, Dict[feature_name, list[p50_values]]]
    """
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    if not jobs_dir.exists():
        print(f"Jobs directory not found: {jobs_dir}")
        sys.exit(1)

    job_dirs = sorted(jobs_dir.iterdir())
    if not job_dirs:
        print(f"No job directories found in {jobs_dir}")
        return collected

    found = 0
    for job_dir in job_dirs:
        if not job_dir.is_dir():
            continue
        tech_file = job_dir / "technical_analytics.json"
        if not tech_file.exists():
            continue

        try:
            with open(tech_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {job_dir.name}: {e}")
            continue

        features_keys = [k for k in data if isinstance(k, str) and k.endswith("_features")]
        if not features_keys:
            continue

        for fk in features_keys:
            player_features = data[fk]
            if not isinstance(player_features, dict):
                continue
            for stroke_type, feat_map in player_features.items():
                if not isinstance(feat_map, dict):
                    continue
                for feat_name, feat_stats in feat_map.items():
                    if isinstance(feat_stats, dict) and "p50" in feat_stats:
                        val = feat_stats["p50"]
                        if isinstance(val, (int, float)):
                            collected[stroke_type][feat_name].append(float(val))

        found += 1
        print(f"  {job_dir.name}: {len(features_keys)} player(s) processed")

    print(f"\nScanned {found} jobs with technique features")
    return collected


def build_tier(collected: dict, tier: str, output_dir: Path, min_samples: int = 3):
    """Compute percentile tables and write reference JSON."""
    reference = {}

    for stroke_type in sorted(collected):
        ref_stroke = {}
        for feat_name, values in collected[stroke_type].items():
            if len(values) < min_samples:
                continue
            arr = np.array(values)
            reference.setdefault(stroke_type, {})[feat_name] = {
                "p10": round(float(np.percentile(arr, 10)), 2),
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
            }

    if not reference:
        print(f"WARNING: No reference data computed for tier '{tier}' "
              f"(need >= {min_samples} samples per feature)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{tier}.json"
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Wrote {out_path} ({len(reference)} stroke types, "
          f"{sum(len(v) for v in reference.values())} features)")


def main():
    parser = argparse.ArgumentParser(description="Build reference percentile tables")
    parser.add_argument("--tier", default="intermediate",
                        help="Reference tier name (default: intermediate)")
    parser.add_argument("--jobs-dir", default="data/jobs",
                        help="Pipeline job output directory (default: data/jobs)")
    parser.add_argument("--output-dir", default="data/reference",
                        help="Output directory for reference files (default: data/reference)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Minimum samples per feature (default: 3)")
    args = parser.parse_args()

    jobs_dir = Path(args.jobs_dir)
    output_dir = Path(args.output_dir)

    print(f"Collecting features from {jobs_dir} ...")
    collected = collect_features(jobs_dir)

    if not collected:
        no_data_msg = (
            "No technique features found. Ensure pipeline has been run "
            "and technical_analytics.json exists with player_X_features keys."
        )
        print(no_data_msg)
        sys.exit(0)

    total_pairs = sum(len(feats) for feats in collected.values())
    print(f"Collected {total_pairs} (stroke, feature) pairs across "
          f"{len(collected)} stroke types")

    build_tier(collected, args.tier, output_dir, args.min_samples)


if __name__ == "__main__":
    main()
