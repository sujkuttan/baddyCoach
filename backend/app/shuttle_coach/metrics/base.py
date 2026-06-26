from dataclasses import dataclass, asdict
from typing import Any

from app.shuttle_coach.events import MatchModel


REGISTRY: list[type["Metric"]] = []


def register(cls: type["Metric"]) -> type["Metric"]:
    REGISTRY.append(cls)
    return cls


@dataclass
class MetricResult:
    metric_id: str
    player_id: str | None
    value: float | dict
    unit: str
    sample_size: int
    confidence: float
    context: dict[str, Any]

    def to_row(self) -> dict:
        return asdict(self)


class Metric:
    metric_id: str = "base"
    requires: set[str] = set()

    def applicable(self, caps: set[str]) -> bool:
        return self.requires.issubset(caps)

    def compute(self, m: MatchModel) -> list[MetricResult]:
        raise NotImplementedError


def run_metrics(match: MatchModel, caps: set[str]) -> list[MetricResult]:
    results = []
    for cls in REGISTRY:
        metric = cls()
        if metric.applicable(caps):
            results.extend(metric.compute(match))
    return results
