import pandas as pd
from app.shuttle_coach.events import MatchModel


def _make_tables():
    return {
        "rallies": pd.DataFrame({"rally_id": [1, 2]}),
        "shots": pd.DataFrame({
            "rally_id": [1, 1, 2],
            "player_id": ["p1", "p2", "p1"],
            "shot_type": ["smash", "clear", "drop"],
        }),
        "hits": pd.DataFrame({
            "rally_id": [1, 1, 2],
            "frame": [100, 150, 200],
        }),
        "shuttle": pd.DataFrame({
            "frame": [100, 150, 200],
            "x": [640.0, 320.0, 480.0],
            "y": [360.0, 180.0, 240.0],
        }),
        "player_detections": pd.DataFrame({
            "frame": [100, 100, 150, 150, 200, 200],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "x": [0.3, 0.7, 0.4, 0.6, 0.5, 0.5],
        }),
        "pose": pd.DataFrame({
            "frame": [100, 100],
            "player_id": ["p1", "p2"],
        }),
    }


class TestMatchModelFromTables:
    def test_creates_model_from_tables(self):
        tables = _make_tables()
        model = MatchModel.from_tables(tables, match_id="test_match")
        assert model.match_id == "test_match"
        assert len(model.rallies) == 2
        assert len(model.shots) == 3
        assert len(model.hits) == 3
        assert len(model.shuttle) == 3
        assert len(model.positions) == 6
        assert model.pose is not None
        assert len(model.pose) == 2

    def test_player_ids_are_sorted(self):
        tables = _make_tables()
        model = MatchModel.from_tables(tables)
        assert model.player_ids == ["p1", "p2"]

    def test_player_ids_ignore_unknown_owner_rows(self):
        tables = _make_tables()
        tables["shots"] = pd.DataFrame(
            {
                "rally_id": [1, 1, 2],
                "player_id": ["p1", None, "p2"],
                "owner_confident": [True, False, True],
                "shot_type": ["smash", "clear", "drop"],
            }
        )
        model = MatchModel.from_tables(tables)
        assert model.player_ids == ["p1", "p2"]
        assert list(model.shots_of("p1")["shot_type"]) == ["smash"]

    def test_pose_is_none_when_missing(self):
        tables = _make_tables()
        del tables["pose"]
        model = MatchModel.from_tables(tables)
        assert model.pose is None


class TestMatchModelShotsOf:
    def test_filters_shots_by_player(self):
        tables = _make_tables()
        model = MatchModel.from_tables(tables)
        p1_shots = model.shots_of("p1")
        assert len(p1_shots) == 2
        assert all(p1_shots["player_id"] == "p1")

    def test_returns_empty_for_unknown_player(self):
        tables = _make_tables()
        model = MatchModel.from_tables(tables)
        result = model.shots_of("unknown")
        assert len(result) == 0

    def test_filters_unconfident_rows(self):
        tables = _make_tables()
        tables["shots"]["owner_confident"] = [True, False, True]
        model = MatchModel.from_tables(tables)
        p2_shots = model.shots_of("p2")
        assert p2_shots.empty


class TestMatchModelPositionsOf:
    def test_filters_positions_by_player(self):
        tables = _make_tables()
        model = MatchModel.from_tables(tables)
        p2_pos = model.positions_of("p2")
        assert len(p2_pos) == 3
        assert all(p2_pos["player_id"] == "p2")
