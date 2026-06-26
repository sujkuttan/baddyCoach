import pandas as pd

from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import run_metrics


def _make_match_with_shots() -> MatchModel:
    tables = {
        "rallies": pd.DataFrame({
            "rally_id": [1, 1, 2, 2],
            "winner_player_id": ["p1", "p1", "p2", "p2"],
            "end_reason": ["winner", "winner", "error", "error"],
        }),
        "shots": pd.DataFrame({
            "rally_id": [1, 1, 1, 2, 2, 2],
            "player_id": ["p1", "p1", "p2", "p1", "p2", "p2"],
            "shot_type": ["smash", "drop", "clear", "net_shot", "smash", "lift"],
        }),
        "hits": pd.DataFrame({"rally_id": [1, 1, 1, 2, 2, 2], "frame": [10, 20, 30, 40, 50, 60]}),
        "shuttle": pd.DataFrame({
            "frame": [10, 20, 30, 40, 50, 60],
            "court_x": [5.0] * 6,
            "court_y": [3.0] * 6,
        }),
        "player_detections": pd.DataFrame({
            "frame": [10, 10],
            "player_id": ["p1", "p2"],
            "court_x": [5.0, 8.0],
            "court_y": [3.0, 4.0],
        }),
    }
    return MatchModel.from_tables(tables)


class TestShotMix:
    def test_computes_shot_mix(self):
        match = _make_match_with_shots()
        results = run_metrics(match, {"shots", "errors"})
        mix_results = [r for r in results if r.metric_id == "shots.mix"]
        assert len(mix_results) > 0
        for r in mix_results:
            assert isinstance(r.value, dict)
            total = sum(r.value.values())
            assert abs(total - 100.0) < 0.1


class TestShotEffectiveness:
    def test_computes_shot_effectiveness(self):
        match = _make_match_with_shots()
        results = run_metrics(match, {"shots", "errors"})
        eff_results = [r for r in results if r.metric_id == "shots.effectiveness"]
        assert len(eff_results) > 0
        for r in eff_results:
            assert isinstance(r.value, dict)
            for v in r.value.values():
                assert 0.0 <= v <= 1.0
