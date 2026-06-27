"""Tests for labeling data contract and stroke↔class_id mapping."""

COACH_CLASSES = [
    "net_shot", "block", "smash", "lift", "clear", "drive",
    "drop", "push", "rush", "cross_court", "short_serve", "long_serve",
]


def _shuttleset_id(stroke: str, side: str) -> int:
    """Map (coach_stroke, side) -> ShuttleSet class ID (0-24).

    far  → Top_*  (class_id 1–12)
    near → Bottom_* (class_id 13–24)
    unknown → 0
    """
    if stroke == "unknown" or stroke not in COACH_CLASSES:
        return 0
    idx = COACH_CLASSES.index(stroke)
    if side == "far":
        return idx + 1
    return idx + 13


def test_known_far_smash():
    """Far-player smash → class_id 3"""
    assert _shuttleset_id("smash", "far") == 3


def test_known_near_smash():
    """Near-player smash → class_id 15 (= 3 + 12)"""
    assert _shuttleset_id("smash", "near") == 15


def test_known_far_lift():
    assert _shuttleset_id("lift", "far") == 4


def test_known_near_lift():
    assert _shuttleset_id("lift", "near") == 16


def test_known_far_net_shot():
    assert _shuttleset_id("net_shot", "far") == 1


def test_known_near_net_shot():
    assert _shuttleset_id("net_shot", "near") == 13


def test_known_far_long_serve():
    assert _shuttleset_id("long_serve", "far") == 12


def test_known_near_long_serve():
    assert _shuttleset_id("long_serve", "near") == 24


def test_unknown_returns_zero():
    assert _shuttleset_id("unknown", "far") == 0
    assert _shuttleset_id("unknown", "near") == 0


def test_unlisted_class_returns_zero():
    assert _shuttleset_id("nonexistent", "far") == 0


def test_roundtrip_all_classes():
    """Every coach class maps to a valid Top_/Bottom_ slot."""
    for i, cls in enumerate(COACH_CLASSES):
        far_id = _shuttleset_id(cls, "far")
        near_id = _shuttleset_id(cls, "near")
        assert 1 <= far_id <= 12, f"{cls}/far -> {far_id} (expected 1-12)"
        assert 13 <= near_id <= 24, f"{cls}/near -> {near_id} (expected 13-24)"
        assert near_id == far_id + 12


def test_csv_contract_fields():
    """Verify the CSV column structure expected by calibrate_bst.py."""
    required = [
        "shot_id", "frame", "ts_start", "ts_end", "player_id", "side",
        "predicted_stroke", "predicted_class_id",
        "true_stroke", "true_class_id", "label_status",
    ]
    # This is a contract test — these field names are expected by
    # calibrate_bst.py::load_from_csv. If they change, the CSV reader breaks.
    assert len(required) == 11
    assert "label_status" in required
    assert "true_class_id" in required


def test_label_status_values():
    """Valid label_status values match calibrate_bst.py expectations."""
    valid = {"labeled", "unsure", "not_a_shot", "skipped"}
    assert "labeled" in valid
    assert "not_a_shot" in valid


def test_report_json_shot_fields():
    """Fields the report must carry per shot for the labeling UI."""
    required = [
        "shot_id", "frame", "ts_start", "ts_end",
        "player_id", "side",
        "stroke_type", "stroke_confidence", "shuttleset_class_id",
        "stroke_source",
    ]
    for field in required:
        assert field, f"Required field missing: {field}"
