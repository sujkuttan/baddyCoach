"""Tests for context fusion layer."""

import numpy as np
from app.models.bst import COACH_STROKE_CLASSES
from app.pipeline.shared.physics import Features


def _make_feats(v_down=1.0, speed_mps=8.0, speed_norm=None,
                zone="mid", contact="overhead", usable=True,
                quality=0.8):
    return Features(
        quality=quality, usable=usable,
        v_down=v_down, speed_mps=speed_mps, speed_norm=speed_norm,
        arc_rise_fall=False, dx_total=0.2,
        contact=contact, zone=zone, depth="mid",
    )


def test_sigmoid_smash_fast_descending():
    """Smash gets higher likelihood on fast descending vs slow ascending."""
    from app.pipeline.shared.context_fusion import _shuttle_likelihood
    fast = _shuttle_likelihood("smash", _make_feats(speed_mps=12.0, v_down=3.0))
    slow = _shuttle_likelihood("smash", _make_feats(speed_mps=2.0, v_down=-2.0))
    assert fast > slow


def test_sigmoid_smash_over_drive_on_steep_descending():
    """Smash outscores drive on steep descending shuttle; drive beats
    smash on flat shuttle."""
    from app.pipeline.shared.context_fusion import _shuttle_likelihood
    steep = _make_feats(speed_mps=12.0, v_down=3.0)
    flat = _make_feats(speed_mps=12.0, v_down=0.0)
    assert _shuttle_likelihood("smash", steep) > _shuttle_likelihood("drive", steep)
    assert _shuttle_likelihood("drive", flat) > _shuttle_likelihood("smash", flat)


def test_zone_net_shot_front():
    """Net shot gets high likelihood in front court."""
    from app.pipeline.shared.context_fusion import _zone_likelihood
    score = _zone_likelihood("net_shot", _make_feats(zone="front"))
    assert score > 0.8


def test_zone_clear_back():
    """Clear gets high likelihood in back court."""
    from app.pipeline.shared.context_fusion import _zone_likelihood
    score = _zone_likelihood("clear", _make_feats(zone="back"))
    assert score > 0.8


def test_zone_smash_mid_back():
    """Smash gets reasonable likelihood from mid or back court."""
    from app.pipeline.shared.context_fusion import _zone_likelihood
    mid_score = _zone_likelihood("smash", _make_feats(zone="mid"))
    back_score = _zone_likelihood("smash", _make_feats(zone="back"))
    assert mid_score > 0.3
    assert back_score > 0.5


def test_height_overhead_boosts_smash():
    """Overhead contact height boosts overhead strokes."""
    from app.pipeline.shared.context_fusion import _height_likelihood
    smash_score = _height_likelihood("smash", "overhead")
    lift_score = _height_likelihood("lift", "overhead")
    assert smash_score > 0.8
    assert lift_score < 0.2


def test_height_underarm_boosts_lift():
    """Underarm contact height boosts underarm strokes."""
    from app.pipeline.shared.context_fusion import _height_likelihood
    lift_score = _height_likelihood("lift", "underarm")
    smash_score = _height_likelihood("smash", "underarm")
    assert lift_score > 0.8
    assert smash_score < 0.2


def test_context_smash_followed_by_block():
    """Smash is often followed by block."""
    from app.pipeline.shared.context_fusion import _context_likelihood
    block_score = _context_likelihood("block", "smash")
    smash_score = _context_likelihood("smash", "smash")
    assert block_score > smash_score


def test_context_serve_followed_by_lift():
    """Serve is often followed by lift."""
    from app.pipeline.shared.context_fusion import _context_likelihood
    lift_score = _context_likelihood("lift", "short_serve")
    assert lift_score > 0.3


def test_context_no_prev():
    """No previous stroke → neutral score."""
    from app.pipeline.shared.context_fusion import _context_likelihood
    score = _context_likelihood("smash", None)
    assert score == 0.5


def test_logits_from_probs_preserves_order():
    """Recovered logits preserve argmax of original probs."""
    from app.pipeline.shared.context_fusion import _logits_from_probs
    probs = np.array([[0.1, 0.5, 0.2, 0.2],
                       [0.3, 0.1, 0.4, 0.2]])
    logits = _logits_from_probs(probs)
    assert logits.shape == probs.shape
    assert np.argmax(logits[0]) == 1
    assert np.argmax(logits[1]) == 2


def test_softmax_is_valid():
    """Softmax produces valid probabilities."""
    from app.pipeline.shared.context_fusion import _softmax
    logits = np.array([[1.0, 2.0, 0.5], [-1.0, 0.0, 3.0]])
    probs = _softmax(logits)
    assert probs.shape == logits.shape
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_fusion_nudges_smash_upward():
    """Fusion increases smash probability on fast descending shuttle."""
    from app.pipeline.shared.context_fusion import ContextFusion
    fusion = ContextFusion(w_shuttle=0.5, w_zone=0.0, w_height=0.0,
                           w_context=0.0, logit_clip=5.0)

    n_classes = 1 + 2 * len(COACH_STROKE_CLASSES)  # 25
    # Start with uniform probs
    probs = np.ones((1, n_classes), dtype=np.float64) / n_classes
    shots = [{"frame": 0, "stroke_type": "unknown",
              "stroke_confidence": 1.0 / n_classes}]

    # Shuttle_raw with fast descending trajectory (x: 100→600, y: 200→400)
    # Non-rectangular court corners for homography reliability
    shuttle_raw = _make_shuttle_df(0, 12, x_start=100, x_step=500/11,
                                    y_start=200, y_step=200/11)
    court = _make_court_trapezoid()
    pose_df = None

    adjusted = fusion.fuse(shots, probs.copy(), shuttle_raw, pose_df,
                           court, 30.0, 1280, 720)
    smash_idx = 1 + COACH_STROKE_CLASSES.index("smash")
    assert adjusted[0, smash_idx] > probs[0, smash_idx]


def test_fusion_preserves_identity_on_poor_features():
    """Fusion preserves BST output when physics features are unavailable."""
    from app.pipeline.shared.context_fusion import ContextFusion
    fusion = ContextFusion(w_shuttle=0.5, w_zone=0.5, w_height=0.5,
                           w_context=0.0, logit_clip=2.0)

    n_classes = 25
    # Non-uniform probs: clear is top
    probs = np.full((1, n_classes), 0.01)
    clear_idx = 1 + COACH_STROKE_CLASSES.index("clear")
    probs[0, clear_idx] = 0.5
    probs[0, 0] = 0.02  # unknown
    probs[0, 1:] = probs[0, 1:] / probs[0, 1:].sum() * 0.48

    shots = [{"frame": 0, "stroke_type": "clear",
              "stroke_confidence": 0.5}]

    # No shuttle data → features unavailable
    adjusted = fusion.fuse(shots, probs.copy(), None, None, {}, 30, 1280, 720)
    # Argmax should be unchanged
    assert np.argmax(adjusted[0]) == clear_idx


# ── Helpers ─────────────────────────────────────────────────────

def _make_shuttle_df(start_frame: int, n_frames: int,
                     x_start=200, x_step=10,
                     y_start=300, y_step=5):
    import pandas as pd
    frames = list(range(start_frame, start_frame + n_frames))
    return pd.DataFrame({
        "frame": frames,
        "x": [x_start + int(i * x_step) for i in range(n_frames)],
        "y": [y_start + int(i * y_step) for i in range(n_frames)],
        "confidence": [0.9] * n_frames,
    })


def _make_court():
    return {
        "valid": True,
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "court_length": 13.4,
        "court_width": 6.10,
        "corners_pixel": [(100, 500), (1820, 500), (100, 100), (1820, 100)],
    }


def _make_court_trapezoid():
    """Non-rectangular corners so court_geometry_reliable passes.
    Homography maps pixel → court meters (13.4×6.1)."""
    return {
        "valid": True,
        "homography": [[13.4 / 1280.0, 0.0, 0.0],
                       [0.0, 6.10 / 720.0, 0.0],
                       [0.0, 0.0, 1.0]],
        "court_length": 13.4,
        "court_width": 6.10,
        # top narrower than bottom → ratio ~0.87 < 0.92 ✓
        "corners_pixel": [(100, 500), (1820, 500), (200, 100), (1700, 100)],
    }
