import pandas as pd

from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import run_metrics


def _make_match_with_errors() -> MatchModel:
    tables = {
        "rallies": pd.DataFrame({
            "rally_id": [1, 2, 3, 4, 5],
            "winner_player_id": ["p1", "p2", "p1", "p1", "p2"],
            "end_reason": ["winner", "error", "winner", "error", "error"],
        }),
        "shots": pd.DataFrame({
            "rally_id": [1, 2, 3, 4, 5],
            "player_id": ["p2", "p1", "p2", "p2", "p1"],
            "shot_type": ["clear", "smash", "drop", "net_shot", "lift"],
        }),
        "hits": pd.DataFrame({"rally_id": [1, 2, 3, 4, 5], "frame": [10, 20, 30, 40, 50]}),
        "shuttle": pd.DataFrame({
            "frame": [10, 20, 30, 40, 50],
            "court_x": [5.0] * 5,
            "court_y": [3.0] * 5,
        }),
        "player_detections": pd.DataFrame({
            "frame": [10, 10],
            "player_id": ["p1", "p2"],
            "court_x": [5.0, 8.0],
            "court_y": [3.0, 4.0],
        }),
    }
    return MatchModel.from_tables(tables)


class TestErrorLocation:
    def test_computes_error_location(self):
        match = _make_match_with_errors()
        results = run_metrics(match, {"shots", "errors"})
        err_results = [r for r in results if r.metric_id == "errors.location_reason"]
        assert len(err_results) > 0
        for r in err_results:
            assert isinstance(r.value, dict)
            total = sum(r.value.values())
            assert abs(total - 100.0) < 0.01
            assert r.unit == "percent"
            assert r.sample_size == 3
