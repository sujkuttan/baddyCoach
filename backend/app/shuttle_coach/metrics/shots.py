import numpy as np
import pandas as pd

from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class ShotMix(Metric):
    metric_id = "shots.mix"
    requires = {"shots"}

    def compute(self, m) -> list[MetricResult]:
        results = []

        for pid in m.player_ids:
            shots = m.shots_of(pid)
            if shots.empty or "shot_type" not in shots.columns:
                continue

            counts = shots["shot_type"].value_counts()
            total = counts.sum()
            if total == 0:
                continue

            pcts = (counts / total * 100.0).round(2)
            mix = {k: float(v) for k, v in pcts.items()}

            results.append(MetricResult(
                metric_id=self.metric_id,
                player_id=pid,
                value=mix,
                unit="percent",
                sample_size=int(total),
                confidence=1.0,
                context={},
            ))
        return results


@register
class ShotEffectiveness(Metric):
    metric_id = "shots.effectiveness"
    requires = {"shots"}

    def compute(self, m) -> list[MetricResult]:
        results = []
        if m.rallies.empty or "winner_player_id" not in m.rallies.columns:
            return results

        winner_rallies = set(m.rallies[m.rallies["winner_player_id"].notna()]["rally_id"].unique())

        for pid in m.player_ids:
            shots = m.shots_of(pid)
            if shots.empty or "shot_type" not in shots.columns:
                continue

            shot_wins: dict[str, int] = {}
            shot_totals: dict[str, int] = {}

            for _, shot in shots.iterrows():
                st = shot["shot_type"]
                rally_id = shot["rally_id"]
                shot_totals[st] = shot_totals.get(st, 0) + 1
                if rally_id in winner_rallies:
                    winner = m.rallies[m.rallies["rally_id"] == rally_id]["winner_player_id"].iloc[0]
                    if winner == pid:
                        shot_wins[st] = shot_wins.get(st, 0) + 1

            if not shot_totals:
                continue

            effectiveness = {
                k: round(shot_wins.get(k, 0) / v, 3)
                for k, v in shot_totals.items()
            }

            results.append(MetricResult(
                metric_id=self.metric_id,
                player_id=pid,
                value=effectiveness,
                unit="rate",
                sample_size=sum(shot_totals.values()),
                confidence=1.0,
                context={},
            ))
        return results
