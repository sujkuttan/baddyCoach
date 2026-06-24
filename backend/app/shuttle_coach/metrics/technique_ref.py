"""Technique reference comparison metric.

Compares per-stroke feature aggregates against reference percentile tables
(from ``data/reference/*.json``) to surface deviations and pressure degradation.

Reads ``technique_features`` from the analytics pipeline (technical stage)
and compares against the configured reference tier.
"""

import json
import numpy as np
from pathlib import Path

from app.config.settings import settings
from app.shuttle_coach.metrics.base import Metric, MetricResult, register


REFERENCE_DIR = Path("data/reference")


def _load_reference(tier: str = "intermediate") -> dict:
    """Load reference percentile table for a tier."""
    path = REFERENCE_DIR / f"{tier}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _percentile_vs_ref(current_val: float, ref_feature: dict) -> float:
    """Estimate which percentile the current value falls into vs reference.

    Returns a value 0..1 (0 = worst, 1 = best) based on linear interpolation
    of the reference p10/p50/p90 distribution.
    """
    p10 = ref_feature.get("p10", 0)
    p50 = ref_feature.get("p50", 50)
    p90 = ref_feature.get("p90", 100)

    if current_val <= p10:
        return max(0.0, (current_val - p10) / max(p10 - 0.1, 1)) * 0.1
    elif current_val <= p50:
        return 0.1 + (current_val - p10) / max(p50 - p10, 1) * 0.4
    elif current_val <= p90:
        return 0.5 + (current_val - p50) / max(p90 - p50, 1) * 0.4
    else:
        return 0.9 + min(0.1, (current_val - p90) / max(p90 - 0.1, 1) * 0.1)


@register
class TechniqueReference(Metric):
    """Compare current technique features against reference percentiles.

    Requires ``technique_features`` from the technical analytics stage.
    Falls back through: own-history → tier file → absolute bounds.
    """

    metric_id = "technique.reference"
    requires = {"technique"}

    def compute(self, m) -> list[MetricResult]:
        results = []
        ref = _load_reference(settings.technique_reference_tier)
        if not ref:
            return results

        # Try to load technique_features from the job directory
        from app.config.settings import settings as s
        job_dir = s.job_dir(m.match_id) if m.match_id else None
        features = {}
        if job_dir:
            tech_data = {}
            tech_json = job_dir / "technical_analytics.json"
            if tech_json.exists():
                with open(tech_json) as f:
                    tech_data = json.load(f)
            for key, val in tech_data.items():
                if key.endswith("_features"):
                    features = val
                    break

        if not features:
            return results

        for pid in m.player_ids:
            player_features = features.get(pid, {})
            if not player_features:
                continue

            for stroke, fmap in player_features.items():
                ref_stroke = ref.get(stroke, {})
                if not ref_stroke:
                    continue

                for fname, cur in fmap.items():
                    ref_feat = ref_stroke.get(fname)
                    if ref_feat is None:
                        continue

                    pctl = _percentile_vs_ref(cur.get("p50", 0), ref_feat)
                    n = cur.get("n", 0)

                    results.append(MetricResult(
                        metric_id=self.metric_id,
                        player_id=pid,
                        value={
                            "stroke": stroke,
                            "feature": fname,
                            "current": cur.get("p50", 0),
                            "ref_p50": ref_feat.get("p50", 0),
                            "percentile": round(pctl, 4),
                            "n": n,
                        },
                        unit="percentile",
                        sample_size=n,
                        confidence=min(1.0, n / settings.technique_min_history_sessions),
                        context={"reference": settings.technique_reference_tier,
                                 "stroke": stroke, "feature": fname},
                    ))

        return results


def pressure_degradation(job_dir: Path | str | None,
                         player_ids: list[str] | None = None) -> dict:
    """Compare technique features under pressure vs free play.

    Reads ``shot_events.parquet`` and ``shots.parquet`` from the job
    directory, splits by ``under_pressure``, and computes per-stroke
    performance gap.

    Args:
        job_dir: Pipeline job directory.
        player_ids: Optional list of player ids to analyze.

    Returns:
        Dict keyed by player_id, each containing per-stroke degradation
        values (0 = no degradation, 1 = complete breakdown under pressure).
    """
    import pandas as pd

    if job_dir is None:
        return {}
    job_dir = Path(job_dir)

    shot_events_path = job_dir / "shot_events.parquet"
    shots_path = job_dir / "shots.parquet"
    if not shot_events_path.exists() or not shots_path.exists():
        return {}

    try:
        events = pd.read_parquet(shot_events_path)
        shots = pd.read_parquet(shots_path)
    except Exception:
        return {}

    if "under_pressure" not in events.columns:
        return {}

    merged = events.merge(shots, on="frame", suffixes=("_evt", "_shot"), how="inner")
    pid_col = "player_id_shot" if "player_id_shot" in merged.columns else "player_id_evt"

    if pid_col not in merged.columns:
        return {}

    result = {}
    pids = player_ids or merged[pid_col].unique()
    for pid in pids:
        pdata = merged[merged[pid_col] == pid]
        if len(pdata) < 5:
            continue

        stroke_col = ("stroke_type_shot" if "stroke_type_shot" in merged.columns
                      else "stroke_type_evt")
        strokes = pdata[stroke_col].unique() if stroke_col in pdata.columns else ["unknown"]

        per_stroke = {}
        for stroke in strokes:
            sdata = pdata[pdata[stroke_col] == stroke]
            free = sdata[sdata["under_pressure"] == False]
            pressed = sdata[sdata["under_pressure"] == True]
            if len(free) < 2 or len(pressed) < 2:
                continue

            conf_col = ("stroke_confidence_shot" if "stroke_confidence_shot" in merged.columns
                        else "stroke_confidence")
            if conf_col in sdata.columns:
                free_mean = float(free[conf_col].mean())
                pressed_mean = float(pressed[conf_col].mean())
                degradation = max(0.0, free_mean - pressed_mean)
            else:
                free_mean = pressed_mean = degradation = 0.0

            per_stroke[stroke] = {
                "free_mean": round(free_mean, 4),
                "pressed_mean": round(pressed_mean, 4),
                "degradation": round(degradation, 4),
                "n_free": len(free),
                "n_pressed": len(pressed),
            }

        if per_stroke:
            result[pid] = per_stroke

    return result


def rally_intensity_buckets(job_dir: Path | str | None,
                            player_ids: list[str] | None = None) -> dict:
    """Compute technique features split by rally intensity (short vs long).

    Reads ``shot_events.parquet`` and ``shots.parquet``, classifies each
    rally as short (=5 shots) or long (>5 shots), and returns per-stroke
    aggregate features for each bucket.

    Args:
        job_dir: Pipeline job directory.
        player_ids: Optional list of player ids.

    Returns:
        Dict keyed by player_id -> bucket -> stroke -> feature aggregates.
    """
    import pandas as pd

    if job_dir is None:
        return {}
    job_dir = Path(job_dir)

    shot_events_path = job_dir / "shot_events.parquet"
    shots_path = job_dir / "shots.parquet"
    rallies_path = job_dir / "rallies.parquet"

    if not all(p.exists() for p in [shot_events_path, shots_path, rallies_path]):
        return {}

    try:
        events = pd.read_parquet(shot_events_path)
        shots = pd.read_parquet(shots_path)
        rallies = pd.read_parquet(rallies_path)
    except Exception:
        return {}

    if "rally_id" not in events.columns:
        return {}

    rally_lengths = rallies[["rally_id", "shot_count"]].copy()
    rally_lengths["bucket"] = rally_lengths["shot_count"].apply(
        lambda n: "short" if n <= 5 else "long"
    )

    merged = events.merge(rally_lengths[["rally_id", "bucket"]], on="rally_id", how="left")
    merged = merged.merge(shots, on="frame", suffixes=("_evt", "_shot"), how="inner")

    pid_col = "player_id_shot" if "player_id_shot" in merged.columns else "player_id_evt"
    stroke_col = ("stroke_type_shot" if "stroke_type_shot" in merged.columns
                  else "stroke_type_evt")
    conf_col = ("stroke_confidence_shot" if "stroke_confidence_shot" in merged.columns
                else "stroke_confidence")

    result = {}
    pids = player_ids or (merged[pid_col].unique() if pid_col in merged.columns else [])
    for pid in pids:
        pdata = merged[merged[pid_col] == pid]
        if len(pdata) < 5:
            continue

        buckets = {}
        for bucket in ("short", "long"):
            bdata = pdata[pdata["bucket"] == bucket]
            if len(bdata) < 3:
                continue
            strokes = {}
            for stroke in bdata[stroke_col].unique():
                sdata = bdata[bdata[stroke_col] == stroke]
                if len(sdata) < 2:
                    continue
                entry = {"n": len(sdata)}
                if conf_col in sdata.columns:
                    entry["mean_conf"] = round(float(sdata[conf_col].mean()), 4)
                strokes[stroke] = entry
            if strokes:
                buckets[bucket] = strokes

        if buckets:
            result[pid] = buckets

    return result
