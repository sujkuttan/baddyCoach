import numpy as np
import pandas as pd

from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class ErrorLocation(Metric):
    metric_id = "errors.location_reason"
    requires = {"errors"}

    def compute(self, m) -> list[MetricResult]:
        results = []
        if m.rallies.empty or "end_reason" not in m.rallies.columns:
            return results

        lost_rallies = m.rallies[m.rallies["end_reason"] != "winner"]

        if lost_rallies.empty:
            return results

        counts = lost_rallies["end_reason"].value_counts()
        total = counts.sum()
        if total == 0:
            return results

        pcts = (counts / total * 100.0).round(2)
        breakdown = {k: float(v) for k, v in pcts.items()}

        results.append(MetricResult(
            metric_id=self.metric_id,
            player_id=None,
            value=breakdown,
            unit="percent",
            sample_size=int(total),
            confidence=1.0,
            context={},
        ))
        return results
