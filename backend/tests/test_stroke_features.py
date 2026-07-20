import numpy as np

from app.pipeline.shared.stroke_features import extract_clip_features


def test_extract_clip_features_includes_racket():
    seq_len = 10
    clip = {
        "shuttle": np.tile(np.array([0.5, 0.5]), (seq_len, 1)).astype(float),
        "pos": np.zeros((seq_len, 2, 2)),
        "JnB": np.zeros((seq_len, 2, 72)),
        "video_len": seq_len,
        "racket_head": np.zeros((seq_len, 2, 2)),
        "racket_present": np.ones((seq_len, 2), dtype=bool),
    }
    # put a near-player racket head close to shuttle at frame 0
    clip["racket_head"][0, 0] = [0.5, 0.5]
    feats = extract_clip_features(clip)
    assert "racket_contact_distance" in feats
    assert "racket_present_frac" in feats
    assert "racket_peak_speed" in feats
    assert feats["racket_contact_distance"] < 0.1
    assert feats["racket_present_frac"] == 1.0


def test_racket_contact_gate_blocks_without_near_contact():
    """When a racket is present (frac>0) but never approaches the shuttle, the
    contact-gated strokes (smash/net_shot/block) must not be predicted."""
    from app.pipeline.shared.stroke_features import (
        extract_clip_features, classify_by_family, classify_family,
    )

    seq_len = 10
    clip = {
        "shuttle": np.tile(np.array([0.5, 0.5]), (seq_len, 1)).astype(float),
        "pos": np.zeros((seq_len, 2, 2)),
        "JnB": np.zeros((seq_len, 2, 72)),
        "video_len": seq_len,
        # racket present but far from the shuttle (distance 0.9 >= threshold)
        "racket_head": np.full((seq_len, 2, 2), 0.5 + 0.9),
        "racket_present": np.ones((seq_len, 2), dtype=bool),
    }
    feats = extract_clip_features(clip)
    assert feats["racket_present_frac"] == 1.0
    assert feats["racket_contact_distance"] >= 0.8
    # Force a smash-shaped feature set, then confirm the gate downgrades it.
    feats["max_speed"] = 0.05
    feats["outgoing_dy"] = 0.02
    feats["landing_x"] = 0.1
    feats["trajectory_curvature"] = 0.0
    feats["outgoing_speed"] = 0.05
    feats["distance_to_net"] = 0.5
    feats["contact_x"] = 0.5
    feats["player_x"] = 0.5
    feats["usable"] = True
    fam = classify_family(feats)
    stroke = classify_by_family(fam, feats)
    assert stroke != "smash"
    assert stroke != "net_shot"
    assert stroke != "block"

