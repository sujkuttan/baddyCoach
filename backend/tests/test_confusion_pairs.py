"""Tests for confusion-pair correction layer."""

import numpy as np
from app.models.bst import COACH_STROKE_CLASSES
from app.pipeline.shared.physics import Features

_CLS_NAMES = ["unknown"] + COACH_STROKE_CLASSES + COACH_STROKE_CLASSES


def _idx(name: str, player: int = 1) -> int:
    if name == "unknown":
        return 0
    base = COACH_STROKE_CLASSES.index(name)
    if player == 1:
        return 1 + base
    return 1 + len(COACH_STROKE_CLASSES) + base


def _probs_with_top2(top1: str, top2: str, conf1=0.5, conf2=0.2):
    """Create (1, 25) probs matrix where top1 and top2 are the top-2 classes."""
    probs = np.full((1, 25), 0.01, dtype=np.float64)
    probs[0, _idx(top1)] = conf1
    probs[0, _idx(top2)] = conf2
    probs[0, 0] = 1.0 - probs[0, 1:].sum()
    return probs


def _make_feats(**kwargs):
    defaults = dict(quality=0.8, usable=True, v_down=0.0, speed_mps=5.0,
                    speed_norm=None, arc_rise_fall=False, dx_total=0.2,
                    contact="overhead", zone="mid", depth="mid")
    defaults.update(kwargs)
    return Features(**defaults)


# ── Rule tests ──────────────────────────────────────────────────

def test_clear_drop_boost_clear_on_arc_deep():
    """Clear boosted when arc rise-fall and deep landing."""
    from app.pipeline.shared.confusion_pairs import rule_clear_drop
    feats = _make_feats(arc_rise_fall=True, depth="deep", speed_mps=8.0)
    assert rule_clear_drop(feats) == 0


def test_clear_drop_boost_drop_on_short_slow():
    """Drop boosted when short landing and slow speed."""
    from app.pipeline.shared.confusion_pairs import rule_clear_drop
    feats = _make_feats(depth="short", speed_mps=4.0)
    assert rule_clear_drop(feats) == 1


def test_clear_drop_no_boost_when_ambiguous():
    """No boost when neither condition is met."""
    from app.pipeline.shared.confusion_pairs import rule_clear_drop
    feats = _make_feats(depth="mid", speed_mps=7.0, arc_rise_fall=False)
    assert rule_clear_drop(feats) is None


def test_drop_smash_boost_smash_on_fast_steep():
    """Smash boosted when fast and steeply descending."""
    from app.pipeline.shared.confusion_pairs import rule_drop_smash
    feats = _make_feats(speed_mps=12.0, v_down=3.0, depth="mid")
    assert rule_drop_smash(feats) == 1


def test_drop_smash_boost_drop_on_slow_short():
    """Drop boosted when slow and short landing."""
    from app.pipeline.shared.confusion_pairs import rule_drop_smash
    feats = _make_feats(speed_mps=4.0, depth="short", v_down=1.0)
    assert rule_drop_smash(feats) == 0


def test_lift_clear_boost_lift_on_underarm_deep():
    """Lift boosted when underarm contact and deep landing."""
    from app.pipeline.shared.confusion_pairs import rule_lift_clear
    feats = _make_feats(contact="underarm", depth="deep")
    assert rule_lift_clear(feats) == 0


def test_lift_clear_boost_clear_on_overhead_arc():
    """Clear boosted when overhead contact and arc trajectory."""
    from app.pipeline.shared.confusion_pairs import rule_lift_clear
    feats = _make_feats(contact="overhead", arc_rise_fall=True, depth="deep")
    assert rule_lift_clear(feats) == 1


def test_drive_block_boost_drive_on_fast_flat():
    """Drive boosted when fast and flat trajectory."""
    from app.pipeline.shared.confusion_pairs import rule_drive_block
    feats = _make_feats(speed_mps=9.0, v_down=0.0)
    assert rule_drive_block(feats) == 0


def test_drive_block_boost_block_on_slow_descending():
    """Block boosted when slow and descending."""
    from app.pipeline.shared.confusion_pairs import rule_drive_block
    feats = _make_feats(speed_mps=3.0, v_down=1.5)
    assert rule_drive_block(feats) == 1


def test_net_shot_push_boost_net_shot_on_front_slow_low():
    """Net shot boosted when front zone, slow speed, low contact."""
    from app.pipeline.shared.confusion_pairs import rule_net_shot_push
    feats = _make_feats(zone="front", contact="low", speed_mps=2.0)
    assert rule_net_shot_push(feats) == 0


def test_net_shot_push_boost_push_on_mid_medium():
    """Push boosted when mid zone, medium speed, side/underarm contact."""
    from app.pipeline.shared.confusion_pairs import rule_net_shot_push
    feats = _make_feats(zone="mid", contact="side", speed_mps=4.0)
    assert rule_net_shot_push(feats) == 1


def test_short_serve_lift_boost_serve_on_front_low():
    """Short serve boosted when front zone and low contact."""
    from app.pipeline.shared.confusion_pairs import rule_short_serve_lift
    feats = _make_feats(zone="front", contact="low", speed_mps=2.0)
    assert rule_short_serve_lift(feats) == 0


def test_short_serve_lift_boost_lift_on_back_underarm():
    """Lift boosted when back zone and underarm contact."""
    from app.pipeline.shared.confusion_pairs import rule_short_serve_lift
    feats = _make_feats(zone="back", contact="underarm", speed_mps=6.0)
    assert rule_short_serve_lift(feats) == 1


# ── Integration tests ───────────────────────────────────────────

def test_resolve_increases_target_prob():
    """Resolve boosts the target class probability."""
    from app.pipeline.shared.confusion_pairs import resolve_confusion_pairs
    probs = _probs_with_top2("clear", "drop", conf1=0.4, conf2=0.25)
    shots = [{"frame": 0, "stroke_type": "clear", "stroke_confidence": 0.4}]
    shuttle_raw = _make_shuttle(0)
    court = _make_court()
    adjusted = resolve_confusion_pairs(
        probs, shots, shuttle_raw, None, court, 30, 1280, 720, boost=2.0)
    # Clear should remain top (boosted by arc+deep features from _make_shuttle)
    assert _idx("clear") not in (0,)
    assert adjusted.shape == (1, 25)


def test_resolve_skips_non_pair():
    """No correction applied when top-2 is not a known confusion pair."""
    from app.pipeline.shared.confusion_pairs import resolve_confusion_pairs
    probs = _probs_with_top2("smash", "drive", conf1=0.5, conf2=0.2)
    shots = [{"frame": 0, "stroke_type": "smash", "stroke_confidence": 0.5}]
    shuttle_raw = _make_shuttle(0)
    court = _make_court()
    adjusted = resolve_confusion_pairs(
        probs, shots, shuttle_raw, None, court, 30, 1280, 720, boost=2.0)
    # Distribution should be very close to original (no correction)
    assert abs(adjusted[0, _idx("smash")] - 0.5) < 0.05


def test_resolve_empty_shuttle_noop():
    """No correction when shuttle data is empty."""
    from app.pipeline.shared.confusion_pairs import resolve_confusion_pairs
    probs = _probs_with_top2("clear", "drop", conf1=0.4, conf2=0.3)
    shots = [{"frame": 0}]
    import pandas as pd
    adjusted = resolve_confusion_pairs(
        probs, shots, pd.DataFrame(), None, {}, 30, 1280, 720, boost=2.0)
    assert adjusted.shape == (1, 25)
    assert np.allclose(adjusted, probs)


# ── Helpers ─────────────────────────────────────────────────────

def _make_shuttle(start_frame: int, n_frames: int = 12) -> "pd.DataFrame":
    import pandas as pd
    frames = list(range(start_frame, start_frame + n_frames))
    return pd.DataFrame({
        "frame": frames,
        "x": [100 + i * 30 for i in range(n_frames)],
        "y": [300 + i * 10 for i in range(n_frames)],
        "confidence": [0.9] * n_frames,
    })


def _make_court():
    return {
        "valid": True,
        "homography": [[13.4 / 1280.0, 0.0, 0.0],
                       [0.0, 6.10 / 720.0, 0.0],
                       [0.0, 0.0, 1.0]],
        "court_length": 13.4,
        "court_width": 6.10,
        "corners_pixel": [(100, 500), (1820, 500), (200, 100), (1700, 100)],
    }
