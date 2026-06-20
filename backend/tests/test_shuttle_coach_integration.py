import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

from app.shuttle_coach.engine import analyze


def _write_parquet(path: Path, name: str, df: pd.DataFrame):
    df.to_parquet(path / f"{name}.parquet", index=False)


def test_analyze_end_to_end():
    with TemporaryDirectory() as d:
        base = Path(d)

        rallies = pd.DataFrame({
            "rally_id": [1, 2, 3],
            "winner_player_id": ["p1", "p2", "p1"],
            "end_reason": ["winner", "winner", "unforced"],
        })
        shots = pd.DataFrame({
            "rally_id": [1, 1, 2, 2, 3, 3],
            "player_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "shot_type": ["smash", "clear", "drop", "net_shot", "smash", "error"],
            "hit_frame": [100, 105, 200, 205, 300, 305],
        })
        hits = pd.DataFrame({
            "rally_id": [1, 1, 2, 2, 3, 3],
            "frame": [100, 105, 200, 205, 300, 305],
        })
        shuttle = pd.DataFrame({
            "frame": [100, 105, 200, 205, 300, 305],
            "x": [5.0, 8.0, 5.0, 8.0, 5.0, 8.0],
            "y": [3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        })
        detections = pd.DataFrame({
            "frame": list(range(90, 130)) * 2,
            "player_id": ["p1"] * 40 + ["p2"] * 40,
            "court_x": [5.0] * 40 + [8.0] * 40,
            "court_y": [3.0] * 40 + [4.0] * 40,
        })
        pose = pd.DataFrame({
            "frame": [100, 200, 300],
            "player_id": ["p1", "p1", "p1"],
        })

        _write_parquet(base, "rallies", rallies)
        _write_parquet(base, "shots", shots)
        _write_parquet(base, "hits", hits)
        _write_parquet(base, "shuttle", shuttle)
        _write_parquet(base, "player_detections", detections)
        _write_parquet(base, "pose", pose)

        result = analyze(d)

        assert "player_ids" in result
        assert "capabilities" in result
        assert "metrics" in result
        assert "findings" in result
        assert "report_md" in result
        assert "report_json" in result
        assert len(result["player_ids"]) == 2
        assert isinstance(result["report_md"], str)
        assert isinstance(result["report_json"], dict)
        assert len(result["metrics"]) > 0


def test_analyze_minimal():
    with TemporaryDirectory() as d:
        base = Path(d)

        rallies = pd.DataFrame({"rally_id": [1]})
        shots = pd.DataFrame({
            "rally_id": [1, 1],
            "player_id": ["p1", "p2"],
            "shot_type": ["smash", "clear"],
            "hit_frame": [100, 105],
        })
        hits = pd.DataFrame({"rally_id": [1, 1], "frame": [100, 105]})
        shuttle = pd.DataFrame({"frame": [100, 105], "x": [5.0, 8.0], "y": [3.0, 4.0]})
        detections = pd.DataFrame({
            "frame": [100, 105],
            "player_id": ["p1", "p2"],
        })

        _write_parquet(base, "rallies", rallies)
        _write_parquet(base, "shots", shots)
        _write_parquet(base, "hits", hits)
        _write_parquet(base, "shuttle", shuttle)
        _write_parquet(base, "player_detections", detections)

        result = analyze(d)

        assert result["player_ids"] == ["p1", "p2"]
        assert "shots" in result["capabilities"]
        assert "errors" in result["capabilities"]
        assert result["report_md"].startswith("# Coaching Report")
