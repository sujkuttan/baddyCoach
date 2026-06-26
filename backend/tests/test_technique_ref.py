import pytest
import json
from pathlib import Path

from app.shuttle_coach.metrics.technique_ref import (
    _percentile_vs_ref, _load_reference, pressure_degradation,
    rally_intensity_buckets,
)


def test_percentile_below_p10():
    ref = {"p10": 10, "p50": 50, "p90": 90}
    p = _percentile_vs_ref(5, ref)
    assert 0.0 <= p <= 0.1


def test_percentile_at_p50():
    ref = {"p10": 10, "p50": 50, "p90": 90}
    p = _percentile_vs_ref(50, ref)
    assert 0.1 <= p <= 0.5


def test_percentile_above_p90():
    ref = {"p10": 10, "p50": 50, "p90": 90}
    p = _percentile_vs_ref(100, ref)
    assert 0.9 <= p <= 1.0


def test_load_reference_missing():
    ref = _load_reference("nonexistent_tier")
    assert ref == {}


def test_load_reference_from_file(tmp_path):
    data = {"smash": {"elbow_extension": {"p10": 10, "p50": 20, "p90": 30}}}
    p = tmp_path / "data" / "reference"
    p.mkdir(parents=True, exist_ok=True)
    (p / "intermediate.json").write_text(json.dumps(data))
    # Patch REFERENCE_DIR
    import app.shuttle_coach.metrics.technique_ref as tr
    original = tr.REFERENCE_DIR
    tr.REFERENCE_DIR = p
    try:
        ref = tr._load_reference("intermediate")
        assert "smash" in ref
    finally:
        tr.REFERENCE_DIR = original


def test_pressure_degradation_no_data(tmp_path):
    result = pressure_degradation(tmp_path, player_ids=["player_1"])
    assert result == {}


def test_rally_intensity_buckets_no_data(tmp_path):
    result = rally_intensity_buckets(tmp_path, player_ids=["player_1"])
    assert result == {}
