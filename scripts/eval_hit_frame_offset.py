#!/usr/bin/env python3
"""Offline evaluator for the hit-frame calibration offset.

This tool re-measures the systematic lag between the pipeline's detected hit
frames and ground-truth labelled contact frames, so that
``hit_frame_calibration_offset`` (backend/app/config/settings.py) can be
re-fit if the Phase-1 hit detector's behaviour changed.

Pipeline behaviour
------------------
In ``pipeline/hits.py`` the detector finds hit frames from the shuttle
trajectory, then subtracts ``hit_frame_calibration_offset`` (default 8) to
centre the distribution:

    c["frame"] = c["frame"] - calib_offset

The detector's raw output lags the true contact frame (the trajectory
inflection point trails the racket-shuttle contact), so the offset corrects
that lag. This script treats the ``frame`` column in ``hits.parquet`` as the
RAW detected frame (pre-calibration) and reports, for each candidate offset,
``median((raw_frame - offset) - label_frame)``.

If your hits artifact already has calibration subtracted, pass
``--frame-is-calibrated`` and the script will *add back* the currently
configured offset (``--current-offset``, default 8) before re-applying each
candidate. This keeps the comparison honest against the label frames.

Matching
--------
Each labelled contact frame is matched to the nearest detected hit frame
within ``--tol`` frames (default 30). Labels with no in-tolerance hit are
reported as unmatched and excluded from the median.

Deferred decision
-----------------
This task (Task 9) SHIPS ONLY THIS SCRIPT. The offset value itself is NOT
changed here -- the decision is deferred until a fresh pipeline run that
reflects Tasks 7-8 (raw-preferred hit detection + y_frac nudge) exists
(Task 16). Without a fresh hits parquet, measuring the offset would reflect
the OLD detector and be meaningless.

Usage
-----
    python scripts/eval_hit_frame_offset.py \
        --labels labels_enriched.csv --hits results/.../debug/hits.parquet

Omitting or failing to find ``--hits`` prints a clear "defer to Task 16"
message and exits 0.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

OFFSET_CANDIDATES = [0, 4, 6, 8, 10]
DEFAULT_TOL = 30


def _load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label_frame" not in df.columns:
        raise SystemExit(
            f"labels CSV {path!r} has no 'label_frame' column "
            f"(found: {df.columns.tolist()})"
        )
    return df


def _load_hits(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "frame" not in df.columns:
        raise SystemExit(
            f"hits parquet {path!r} has no 'frame' column "
            f"(found: {df.columns.tolist()})"
        )
    return df


def evaluate(labels: pd.DataFrame, hits: pd.DataFrame, tol: int,
             current_offset: int, frame_is_calibrated: bool) -> dict:
    """Return per-offset median error plus matching stats."""
    label_frames = labels["label_frame"].to_numpy(dtype=float)
    hit_frames = hits["frame"].to_numpy(dtype=float)

    # If the artifact already had calibration subtracted, undo it so we can
    # re-apply candidate offsets from a common (raw) baseline.
    if frame_is_calibrated:
        hit_frames = hit_frames + current_offset

    n = len(label_frames)
    matched_pred = np.full(n, np.nan)
    matched = np.zeros(n, dtype=bool)

    for i, lf in enumerate(label_frames):
        if not np.isfinite(lf):
            continue
        dist = np.abs(hit_frames - lf)
        if len(dist) == 0:
            continue
        j = int(np.argmin(dist))
        if dist[j] <= tol:
            matched_pred[i] = hit_frames[j]
            matched[i] = True

    n_matched = int(matched.sum())
    raw_errors = matched_pred[matched] - label_frames[matched]  # pred - label

    results = {}
    for o in OFFSET_CANDIDATES:
        adj = raw_errors - o  # (raw_frame - offset) - label
        results[o] = float(np.median(adj)) if adj.size else float("nan")

    return {
        "n_labels": n,
        "n_matched": n_matched,
        "tol": tol,
        "frame_is_calibrated": frame_is_calibrated,
        "current_offset": current_offset,
        "raw_median_error": float(np.median(raw_errors)) if raw_errors.size else float("nan"),
        "per_offset_median_error": results,
        "best_offset": (min(results, key=lambda k: abs(results[k]))
                        if any(np.isfinite(list(results.values()))) else None),
    }


def _print_report(report: dict) -> None:
    print("Hit-frame calibration offset evaluation")
    print("=" * 60)
    print(f"labels:                          {report['n_labels']}")
    print(f"matched (within tol):            {report['n_matched']}")
    print(f"tolerance (frames):              {report['tol']}")
    print(f"hits frame treated as calibrated: {report['frame_is_calibrated']}")
    print(f"currently configured offset:     {report['current_offset']}")
    print(f"median raw error (pred - label): {report['raw_median_error']:.2f}")
    print("-" * 60)
    print("median((pred - offset) - label) per candidate offset:")
    for o in OFFSET_CANDIDATES:
        v = report["per_offset_median_error"][o]
        tag = "  <-- closest to 0" if o == report["best_offset"] else ""
        print(f"  offset {o:>2}: median error = {v:7.2f}{tag}")
    print("=" * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True,
                        help="Path to labels CSV (needs a 'label_frame' column)")
    parser.add_argument("--hits", default=None,
                        help="Path to hits.parquet (needs a 'frame' column). "
                             "If omitted or not found, the offset decision is "
                             "deferred and the script exits 0.")
    parser.add_argument("--tol", type=int, default=DEFAULT_TOL,
                        help="Max frame distance to match a label to a hit "
                             f"(default {DEFAULT_TOL})")
    parser.add_argument("--current-offset", type=int, default=8,
                        help="Configured hit_frame_calibration_offset, used to "
                        "undo calibration when --frame-is-calibrated (default 8)")
    parser.add_argument("--frame-is-calibrated", action="store_true",
                        help="Treat the hits.parquet 'frame' as already "
                             "calibration-subtracted (undo it before re-applying "
                             "candidate offsets).")
    args = parser.parse_args(argv)

    if args.hits is None:
        print(
            "Deferring offset decision: no --hits parquet provided.\n"
            "The offset value must be re-measured against a FRESH pipeline run\n"
            "that reflects Tasks 7-8 (raw-preferred hit detection + y_frac nudge).\n"
            "See Task 16. Leaving hit_frame_calibration_offset unchanged (at "
            f"{args.current_offset})."
        )
        return 0

    try:
        hits = _load_hits(args.hits)
    except FileNotFoundError:
        print(
            f"Deferring offset decision: hits parquet not found at "
            f"{args.hits!r}.\n"
            "The offset value must be re-measured against a FRESH pipeline run\n"
            "that reflects Tasks 7-8 (raw-preferred hit detection + y_frac nudge).\n"
            "See Task 16. Leaving hit_frame_calibration_offset unchanged (at "
            f"{args.current_offset})."
        )
        return 0

    labels = _load_labels(args.labels)
    report = evaluate(labels, hits, args.tol, args.current_offset,
                      args.frame_is_calibrated)
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
