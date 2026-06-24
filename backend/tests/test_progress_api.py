import pytest
import json
from pathlib import Path

from app.storage.progress import (
    make_snapshot, save_player_session, get_player_history,
    compute_metric_trend, compare_last_n,
)


def test_make_snapshot_contains_expected_keys():
    snap = make_snapshot("job_1", {"tactical_analytics": {"p1": {"total_shots": 10}}})
    assert "job_id" in snap
    assert "timestamp" in snap
    assert "tactical" in snap


def test_save_and_load_session(tmp_path):
    import app.storage.progress as pr
    original = pr.HISTORY_DIR
    pr.HISTORY_DIR = tmp_path / "player_history"
    try:
        save_player_session("test_player", "job_1", {})
        history = get_player_history("test_player")
        assert len(history) == 1
        assert history[0]["job_id"] == "job_1"
    finally:
        pr.HISTORY_DIR = original


def test_compute_metric_trend_insufficient(tmp_path):
    import app.storage.progress as pr
    original = pr.HISTORY_DIR
    pr.HISTORY_DIR = tmp_path / "history"
    pr.HISTORY_DIR.mkdir(exist_ok=True)
    try:
        trend = compute_metric_trend("unknown_player", "fitness.rally_intensity")
        assert trend["direction"] == "insufficient_data"
    finally:
        pr.HISTORY_DIR = original


def test_compare_last_n_empty(tmp_path):
    import app.storage.progress as pr
    original = pr.HISTORY_DIR
    pr.HISTORY_DIR = tmp_path / "history2"
    pr.HISTORY_DIR.mkdir(exist_ok=True)
    try:
        headlines = compare_last_n("empty_player", n=5)
        assert headlines == []
    finally:
        pr.HISTORY_DIR = original
