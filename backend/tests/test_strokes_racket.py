import numpy as np
import pandas as pd

from app.pipeline.strokes import _build_clip


def test_build_clip_includes_racket_arrays():
    frames = list(range(100))
    racket_det = [
        {"frame": 10, "player_side": "near", "bbox": (1, 2, 3, 4), "conf": 0.9, "head_point": (2.0, 1.0)},
        {"frame": 10, "player_side": "far", "bbox": (5, 6, 7, 8), "conf": 0.9, "head_point": (6.0, 5.0)},
    ]
    clip = _build_clip(
        frames=frames,
        shuttle_df=None,
        pose_df=None,
        vid_w=1920,
        vid_h=1080,
        court_length=13.4,
        court_width=6.10,
        seq_len=100,
        player_sides={"near": "player_1", "far": "player_2"},
        racket_detections=racket_det,
        original_len=100,
    )
    assert "racket_head" in clip
    assert clip["racket_head"].shape == (100, 2, 2)
    # near player is p_idx=1, far player is p_idx=0
    assert np.allclose(clip["racket_head"][10, 1], [2.0, 1.0])
    assert np.allclose(clip["racket_head"][10, 0], [6.0, 5.0])
    assert "racket_present" in clip
    assert clip["racket_present"].shape == (100, 2)
    assert clip["racket_present"][10, 1] == True
    assert clip["racket_present"][10, 0] == True
    assert clip["racket_present"][5, 0] == False
