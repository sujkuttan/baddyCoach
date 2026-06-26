import pandas as pd

from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import REGISTRY, run_metrics


def _make_match(**overrides) -> MatchModel:
    tables = {
        "rallies": pd.DataFrame({"rally_id": [1], "winner_player_id": ["p1"], "end_reason": ["winner"]}),
        "shots": pd.DataFrame({"rally_id": [1], "player_id": ["p1"], "shot_type": ["smash"], "hit_frame": [100]}),
        "hits": pd.DataFrame({"rally_id": [1], "frame": [100]}),
        "shuttle": pd.DataFrame({"frame": [100], "court_x": [5.0], "court_y": [3.0]}),
        "player_detections": pd.DataFrame({
            "frame": list(range(90, 120)) + list(range(90, 120)),
            "player_id": ["p1"] * 30 + ["p2"] * 30,
            "court_x": [5.0] * 30 + [8.0] * 30,
            "court_y": [3.0] * 30 + [4.0] * 30,
        }),
    }
    tables.update(overrides)
    return MatchModel.from_tables(tables)


class TestMetricRegistry:
    def test_registry_is_nonempty_list(self):
        assert isinstance(REGISTRY, list)
        assert len(REGISTRY) > 0

    def test_all_registry_entries_are_metric_subclasses(self):
        from app.shuttle_coach.metrics.base import Metric
        for cls in REGISTRY:
            assert issubclass(cls, Metric)

    def test_metric_has_metric_id(self):
        from app.shuttle_coach.metrics.base import Metric
        for cls in REGISTRY:
            m = cls()
            assert isinstance(m.metric_id, str)
            assert "." in m.metric_id


class TestRunMetrics:
    def test_returns_list_of_metric_results(self):
        from app.shuttle_coach.metrics.base import MetricResult
        match = _make_match()
        results = run_metrics(match, {"shots", "errors"})
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, MetricResult)

    def test_filters_by_capability(self):
        match = _make_match()
        results_no_movement = run_metrics(match, {"shots", "errors"})
        metric_ids = {r.metric_id for r in results_no_movement}
        assert "movement.recovery_time" not in metric_ids

        results_with_movement = run_metrics(match, {"shots", "errors", "movement"})
        metric_ids_m = {r.metric_id for r in results_with_movement}
        assert "movement.recovery_time" in metric_ids_m
