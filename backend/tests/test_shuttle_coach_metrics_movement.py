import numpy as np
import pandas as pd

from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import run_metrics


def _make_positions(player_id: str, frames: list[int], court_x: list[float], court_y: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "frame": frames,
        "player_id": [player_id] * len(frames),
        "court_x": court_x,
        "court_y": court_y,
    })


def _make_match_with_movement() -> MatchModel:
    frames = list(range(0, 60))
    p1_x = [6.5] * 10 + [8.0] * 5 + [6.5] * 10 + [8.0] * 5 + [6.5] * 10 + [8.0] * 5 + [6.5] * 15
    p1_y = [3.0] * 10 + [3.5] * 5 + [3.0] * 10 + [3.5] * 5 + [3.0] * 10 + [3.5] * 5 + [3.0] * 15
    p2_x = [7.0] * 60
    p2_y = [4.0] * 60

    pos_p1 = _make_positions("p1", frames, p1_x, p1_y)
    pos_p2 = _make_positions("p2", frames, p2_x, p2_y)
    positions = pd.concat([pos_p1, pos_p2], ignore_index=True)

    tables = {
        "rallies": pd.DataFrame({"rally_id": [1]}),
        "shots": pd.DataFrame({
            "rally_id": [1, 1, 1],
            "player_id": ["p1", "p2", "p1"],
            "hit_frame": [5, 15, 25],
        }),
        "hits": pd.DataFrame({"rally_id": [1, 1, 1], "frame": [5, 15, 25]}),
        "shuttle": pd.DataFrame({"frame": [5, 15, 25], "court_x": [5.0] * 3, "court_y": [3.0] * 3}),
        "player_detections": positions,
    }
    return MatchModel.from_tables(tables)


class TestRecoveryTime:
    def test_computes_recovery_time(self):
        match = _make_match_with_movement()
        results = run_metrics(match, {"shots", "errors", "movement"})
        rec_results = [r for r in results if r.metric_id == "movement.recovery_time"]
        assert len(rec_results) == 2
        for r in rec_results:
            assert r.unit == "seconds"
            assert r.sample_size >= 0
            assert isinstance(r.value, float)
            assert r.value >= 0.0


class TestCourtCoverage:
    def test_computes_court_coverage(self):
        match = _make_match_with_movement()
        results = run_metrics(match, {"shots", "errors", "movement"})
        cov_results = [r for r in results if r.metric_id == "movement.court_coverage"]
        assert len(cov_results) == 2
        for r in cov_results:
            assert isinstance(r.value, dict)
            assert len(r.value) == 6
            total = sum(r.value.values())
            assert abs(total - 100.0) < 0.01


class TestDistancePerRally:
    def test_computes_distance_per_rally(self):
        match = _make_match_with_movement()
        results = run_metrics(match, {"shots", "errors", "movement"})
        dist_results = [r for r in results if r.metric_id == "movement.distance_per_rally"]
        assert len(dist_results) == 2
        for r in dist_results:
            assert r.unit == "meters"
            assert isinstance(r.value, float)
            assert r.value >= 0.0
