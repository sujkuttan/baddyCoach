"""Causal pattern engine — conditional outcome statistics.

Two metrics:
  - ``patterns.conditional_outcome``: loss/win rates grouped by
    (stroke_type, zone, under_pressure).
  - ``patterns.transition_outcome``: 2-shot transition loss rates
    keyed on (prev_stroke_type -> stroke_type).

Both read ``shot_events.parquet`` produced by ShotContextStage.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from app.config.settings import settings
from app.shuttle_coach.metrics.base import Metric, MetricResult, register


def wilson_lower_bound(p: float, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound for a binomial proportion.

    More robust than a normal approximation for small n.
    """
    if n == 0:
        return 0.0
    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return max(0.0, (centre - adj) / denominator)


def sample_confidence(n: int, match_model, fallback_rate: float = 0.0) -> float:
    """Scale confidence by sample size and fallback rate.

    Returns 0..1.  Needs at least 20 samples for full confidence;
    degraded by the BST fallback rate.
    """
    size_factor = min(1.0, n / 20)
    return round(size_factor * (1 - fallback_rate), 4)


@register
class ConditionalShotOutcome(Metric):
    """Loss/win rates for shots grouped by stroke_type, zone, and pressure.

    Requires ``shot_events.parquet`` with the enriched columns from
    ShotContextStage.  Falls back to shots.parquet with basic columns.
    """

    metric_id = "patterns.conditional_outcome"
    requires = {"tactical"}

    def compute(self, m) -> list[MetricResult]:
        results = []
        ev = self._load_shot_events(m)
        if ev is None or ev.empty:
            return results

        fallback_rate = self._estimate_fallback(ev)
        min_samples = settings.pattern_min_samples
        lookahead_k = settings.pattern_lookahead_k

        for pid in m.player_ids:
            pe = ev[ev["player_id"] == pid].copy()
            if pe.empty:
                continue
            baseline_loss = pe["lost_point"].mean()

            # ── Single-shot patterns ─────────────────────────────
            for (stroke, zone, pressed), g in pe.groupby(
                    ["stroke_type", "zone", "under_pressure"],
                    sort=False):
                n = len(g)
                if n < min_samples:
                    continue

                loss = g["lost_point"].mean()
                win = g["won_point"].mean()

                results.append(MetricResult(
                    metric_id=self.metric_id,
                    player_id=pid,
                    value={
                        "n": n,
                        "loss_rate": round(float(loss), 3),
                        "win_rate": round(float(win), 3),
                        "baseline_loss": round(float(baseline_loss), 3),
                        "wilson_loss_lb": round(float(wilson_lower_bound(loss, n)), 3),
                        "stroke": str(stroke),
                        "zone": str(zone),
                        "pressed": bool(pressed),
                    },
                    unit="rate",
                    sample_size=n,
                    confidence=sample_confidence(n, m, fallback_rate),
                    context={"stroke": str(stroke), "zone": str(zone),
                             "pressed": bool(pressed)},
                ))

            # ── 2-shot transition patterns ───────────────────────
            if "prev_stroke_type" not in pe.columns:
                continue
            valid = pe[pe["prev_stroke_type"].notna()].copy()
            if valid.empty:
                continue
            for (prev_stroke, stroke), g in valid.groupby(
                    ["prev_stroke_type", "stroke_type"], sort=False):
                n = len(g)
                if n < min_samples:
                    continue
                loss = g["lost_point"].mean()
                results.append(MetricResult(
                    metric_id="patterns.transition_outcome",
                    player_id=pid,
                    value={
                        "n": n,
                        "loss_rate": round(float(loss), 3),
                        "baseline_loss": round(float(baseline_loss), 3),
                        "wilson_loss_lb": round(float(wilson_lower_bound(loss, n)), 3),
                        "prev_stroke": str(prev_stroke),
                        "stroke": str(stroke),
                    },
                    unit="rate",
                    sample_size=n,
                    confidence=sample_confidence(n, m, fallback_rate),
                    context={"prev_stroke": str(prev_stroke), "stroke": str(stroke)},
                ))

        return results

    def _load_shot_events(self, m) -> pd.DataFrame | None:
        """Try to load shot_events.parquet, fall back to shots + heuristics."""
        # The MatchModel doesn't carry shot_events directly, so we attempt
        # to load from the job directory via the match_id.
        from app.config.settings import settings as s
        job_dir = s.job_dir(m.match_id) if m.match_id else None
        if job_dir and (job_dir / "shot_events.parquet").exists():
            ev = pd.read_parquet(job_dir / "shot_events.parquet")
            return ev

        # Fallback: enrich basic shots with zone from court_x/court_y
        shots = m.shots
        if shots.empty:
            return None
        ev = shots.copy()
        ev["zone"] = "unknown"
        ev["under_pressure"] = False
        ev["won_point"] = False
        ev["lost_point"] = False
        if "court_x" in ev.columns and "court_y" in ev.columns:
            court_length = settings.court_length
            court_width = settings.court_width
            from app.pipeline.analytics.shot_context import _get_zone_from_court
            for idx, row in ev.iterrows():
                cx = row.get("court_x")
                cy = row.get("court_y")
                if pd.notna(cx) and pd.notna(cy):
                    ev.at[idx, "zone"] = _get_zone_from_court(
                        float(cx), float(cy), court_length, court_width)
        return ev

    @staticmethod
    def _estimate_fallback(ev: pd.DataFrame) -> float:
        if "is_bst_fallback" not in ev.columns:
            return 0.0
        return float(ev["is_bst_fallback"].mean())
