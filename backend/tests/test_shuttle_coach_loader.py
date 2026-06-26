import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.shuttle_coach.loader import load_match, capabilities


def _write_parquet(path: Path, name: str, df: pd.DataFrame):
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f"{name}.parquet", index=False)


class TestLoadMatchBackendFormat:
    def test_loads_files_at_root(self, tmp_path: Path):
        _write_parquet(tmp_path, "rallies", pd.DataFrame({"rally_id": [1, 2]}))
        _write_parquet(tmp_path, "shots", pd.DataFrame({
            "rally_id": [1], "player_id": ["p1"]
        }))
        _write_parquet(tmp_path, "hits", pd.DataFrame({
            "rally_id": [1], "frame": [100]
        }))
        _write_parquet(tmp_path, "shuttle", pd.DataFrame({
            "frame": [100], "x": [640.0], "y": [360.0]
        }))
        _write_parquet(tmp_path, "player_detections", pd.DataFrame({
            "frame": [100], "player_id": ["p1"]
        }))
        _write_parquet(tmp_path, "pose", pd.DataFrame({
            "frame": [100], "player_id": ["p1"]
        }))

        tables = load_match(tmp_path)
        assert "rallies" in tables
        assert "shots" in tables
        assert "hits" in tables
        assert "shuttle" in tables
        assert "player_detections" in tables
        assert "pose" in tables
        assert len(tables["rallies"]) == 2


class TestLoadMatchColabFormat:
    def test_loads_files_in_debug_subdirectory(self, tmp_path: Path):
        debug_dir = tmp_path / "debug"
        _write_parquet(debug_dir, "rallies", pd.DataFrame({"rally_id": [1]}))
        _write_parquet(debug_dir, "shots", pd.DataFrame({
            "rally_id": [1], "player_id": ["p1"]
        }))
        _write_parquet(debug_dir, "hits", pd.DataFrame({
            "rally_id": [1], "frame": [50]
        }))
        _write_parquet(debug_dir, "shuttle", pd.DataFrame({
            "frame": [50], "x": [320.0], "y": [180.0]
        }))
        _write_parquet(debug_dir, "player_detections", pd.DataFrame({
            "frame": [50], "player_id": ["p1"]
        }))

        tables = load_match(tmp_path)
        assert "rallies" in tables
        assert len(tables["rallies"]) == 1


class TestLoadMatchAliases:
    def test_applies_column_aliases(self, tmp_path: Path):
        _write_parquet(tmp_path, "rallies", pd.DataFrame({"rally_id": [1]}))
        _write_parquet(tmp_path, "shots", pd.DataFrame({
            "rally_id": [1],
            "player_id": ["p1"],
            "stroke_type": ["smash"],
            "stroke_confidence": [0.95],
            "frame": [100],
        }))
        _write_parquet(tmp_path, "hits", pd.DataFrame({
            "rally_id": [1], "frame": [100]
        }))
        _write_parquet(tmp_path, "shuttle", pd.DataFrame({
            "frame": [100], "x": [640.0], "y": [360.0]
        }))
        _write_parquet(tmp_path, "player_detections", pd.DataFrame({
            "frame": [100], "player_id": ["p1"]
        }))

        tables = load_match(tmp_path)
        assert "shot_type" in tables["shots"].columns
        assert "shot_conf" in tables["shots"].columns
        assert "hit_frame" in tables["shots"].columns
        assert "stroke_type" not in tables["shots"].columns


class TestLoadMatchValidation:
    def test_raises_on_missing_required_columns(self, tmp_path: Path):
        _write_parquet(tmp_path, "rallies", pd.DataFrame({"rally_id": [1]}))
        _write_parquet(tmp_path, "shots", pd.DataFrame({
            "player_id": ["p1"]
        }))
        _write_parquet(tmp_path, "hits", pd.DataFrame({
            "rally_id": [1], "frame": [100]
        }))
        _write_parquet(tmp_path, "shuttle", pd.DataFrame({
            "frame": [100], "x": [640.0], "y": [360.0]
        }))
        _write_parquet(tmp_path, "player_detections", pd.DataFrame({
            "frame": [100], "player_id": ["p1"]
        }))

        with pytest.raises(ValueError, match="rally_id"):
            load_match(tmp_path)


def _make_minimal_tables(**extra_cols):
    """Build minimal valid tables for capability tests."""
    shots_cols = {"rally_id": [1], "player_id": ["p1"]}
    hits_cols = {"rally_id": [1], "frame": [100]}
    shuttle_cols = {"frame": [100]}
    rallies_cols = {"rally_id": [1]}
    detections_cols = {"frame": [100], "player_id": ["p1"]}

    shuttle_cols.update(extra_cols.get("shuttle", {}))
    detections_cols.update(extra_cols.get("player_detections", {}))

    tables = {
        "shots": pd.DataFrame(shots_cols),
        "hits": pd.DataFrame(hits_cols),
        "shuttle": pd.DataFrame(shuttle_cols),
        "rallies": pd.DataFrame(rallies_cols),
        "player_detections": pd.DataFrame(detections_cols),
    }
    if "pose" in extra_cols:
        tables["pose"] = pd.DataFrame(extra_cols["pose"])
    return tables


class TestCapabilities:
    def test_always_includes_shots_and_errors(self):
        caps = capabilities(_make_minimal_tables())
        assert "shots" in caps
        assert "errors" in caps

    def test_includes_movement_with_court_coords(self):
        caps = capabilities(_make_minimal_tables(
            player_detections={"court_x": [0.5], "court_y": [0.5]}
        ))
        assert "movement" in caps

    def test_excludes_movement_without_court_coords(self):
        caps = capabilities(_make_minimal_tables())
        assert "movement" not in caps

    def test_includes_tactical_with_shuttle_court_coords(self):
        caps = capabilities(_make_minimal_tables(
            shuttle={"court_x": [0.5], "court_y": [0.5]}
        ))
        assert "tactical" in caps

    def test_includes_technique_with_pose_table(self):
        caps = capabilities(_make_minimal_tables(pose={"frame": [100]}))
        assert "technique" in caps

    def test_excludes_technique_without_pose_table(self):
        caps = capabilities(_make_minimal_tables())
        assert "technique" not in caps
