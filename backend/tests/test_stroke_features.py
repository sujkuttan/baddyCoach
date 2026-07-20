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
