from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class PreparationConsistency(Metric):
    metric_id = "technique.preparation_consistency"
    requires = {"technique"}

    def compute(self, m) -> list[MetricResult]:
        return []
