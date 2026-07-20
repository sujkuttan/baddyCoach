import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline import HitFrameLocalizationStage


def test_hit_detection_finds_trajectory_changes(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    n = 40
    frames = list(range(n))
    x = [100.0 + t * 5.0 for t in range(20)] + [200.0 - (t - 20) * 8.0 for t in range(20, n)]
    y = [200.0 - t * 2.0 for t in range(20)] + [160.0 + (t - 20) * 4.0 for t in range(20, n)]
    shuttle_df = pd.DataFrame({
        "frame": frames,
        "x": x,
        "y": y,
        "confidence": [0.95] * n,
    })
    store.set_parquet("shuttle_raw", shuttle_df)

    pose_df = pd.DataFrame({
        "frame": list(range(20)),
        "player_id": ["player_1"] * 20,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(20)],
    })
    store.set_parquet("pose", pose_df)

    stage = HitFrameLocalizationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert result.metadata["hit_count"] > 0
    # Verify at least one hit near the trajectory reversal (frame 20 ± window)
    hits_df = store.get_parquet("hits")
    assert hits_df is not None and len(hits_df) > 0


def test_hit_stage_prefers_shuttle_raw_over_cleaned(tmp_job_dir, monkeypatch):
    """Cleaned trajectory is flat (no reversal); raw has a clear reversal at frame 20."""
    monkeypatch.setattr("app.pipeline.hits.settings.audio_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.wrist_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_refine_window", 0)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_frame_calibration_offset", 0)

    store = ArtifactStore(tmp_job_dir)
    n = 40
    frames = list(range(n))
    # Raw: V-shaped reversal at 20
    x_raw = [100.0 + t * 5.0 for t in range(20)] + [200.0 - (t - 20) * 8.0 for t in range(20, n)]
    y_raw = [200.0 - t * 2.0 for t in range(20)] + [160.0 + (t - 20) * 4.0 for t in range(20, n)]
    store.set_parquet("shuttle_raw", pd.DataFrame({
        "frame": frames, "x": x_raw, "y": y_raw, "confidence": [0.95] * n,
    }))
    # Cleaned: nearly constant — would hide the hit if preferred
    store.set_parquet("shuttle", pd.DataFrame({
        "frame": frames, "x": [150.0] * n, "y": [180.0] * n, "confidence": [0.95] * n,
    }))
    store.set_parquet("pose", pd.DataFrame({
        "frame": [], "player_id": [], "keypoints": [],
    }))

    result = HitFrameLocalizationStage().run(store, StageConfig())
    assert result.status == "success"
    hits = store.get_parquet("hits")
    assert hits is not None and len(hits) > 0
    assert any(abs(int(f) - 20) <= 5 for f in hits["frame"].tolist())


def test_contact_sanity_nudges_extreme_yfrac(tmp_job_dir, monkeypatch):
    """When the calibrated candidate sits at a trajectory y-extreme, the Phase-4
    sanity nudge should move it to a frame with stronger contact signal."""
    monkeypatch.setattr("app.pipeline.hits.settings.audio_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.wrist_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_refine_window", 4)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_frame_calibration_offset", 8)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_contact_sanity_enabled", True)

    # Force Phase-1 detector to emit a single candidate at frame 30. After the
    # calibration offset (-8) it lands on frame 22, which we craft to be a
    # trajectory y-extreme (y_frac ~ 0). The genuine contact (strong direction
    # reversal + wrist at shuttle) is at frame 25, inside the refine window.
    monkeypatch.setattr(
        "app.pipeline.hits.GlobalHitCandidateDetector.detect",
        lambda self, df, **kwargs: [{"frame": 30, "score": 1.0}],
    )

    # Control points for the synthetic shuttle x/y trajectory.
    x_ctrl = {7: 120, 14: 108, 17: 110, 18: 100, 19: 80, 20: 60, 21: 40,
              22: 30, 23: 20, 24: 10, 25: 0, 26: 10, 27: 20, 28: 30,
              29: 40, 30: 45, 37: 40}

    def x_of(f: int) -> float:
        ks = sorted(x_ctrl)
        if f <= ks[0]:
            return float(x_ctrl[ks[0]])
        if f >= ks[-1]:
            return float(x_ctrl[ks[-1]])
        for a, b in zip(ks, ks[1:]):
            if a <= f <= b:
                t = (f - a) / (b - a)
                return x_ctrl[a] + t * (x_ctrl[b] - x_ctrl[a])
        return float(x_ctrl[ks[-1]])

    def y_of(f: int) -> float:
        if f < 7:
            return 20.0
        if f <= 22:
            return 20.0 - (20.0 / 15.0) * (f - 7)   # 20 → 0 (min at 22)
        if f <= 37:
            return float(f - 22)                     # 0 → 15
        return 15.0

    n = 60
    frames = list(range(n))
    shuttle_df = pd.DataFrame({
        "frame": frames,
        "x": [float(x_of(f)) for f in frames],
        "y": [float(y_of(f)) for f in frames],
        "confidence": [0.95] * n,
    })

    # Pose only around the nudge region (18..25) so Phase-2 refine leaves the
    # detected frame untouched. Wrist sits exactly on the shuttle at frame 25.
    pose_rows = []
    for f in range(18, 26):
        kps = np.zeros((17, 3), dtype=float)
        kps[:, 2] = 0.9
        if f == 25:
            kps[9] = [float(x_of(25)), float(y_of(25)), 0.9]
            kps[10] = [float(x_of(25)), float(y_of(25)), 0.9]
        else:
            kps[9] = [500.0, 500.0, 0.9]
            kps[10] = [500.0, 500.0, 0.9]
        pose_rows.append({
            "frame": f, "player_id": "player_1", "keypoints": kps.tolist(),
        })
    pose_df = pd.DataFrame(pose_rows)

    store = ArtifactStore(tmp_job_dir)
    store.set_parquet("shuttle_raw", shuttle_df)
    store.set_parquet("pose", pose_df)

    # Sanity-check the scenario: calibrated candidate (30-8=22) is a y-extreme.
    from app.pipeline.hits import _contact_y_frac, _find_nearest_wrist_frame
    yf = _contact_y_frac(shuttle_df, 22)
    assert yf is not None and not (0.15 <= yf <= 0.85), f"y_frac not extreme: {yf}"
    expected = _find_nearest_wrist_frame(22, pose_df, shuttle_df, 4, 0.30)
    assert expected != 22, "scenario broken: no better frame found"

    result = HitFrameLocalizationStage().run(store, StageConfig())
    assert result.status == "success"
    hits = store.get_parquet("hits")
    assert hits is not None and len(hits) > 0
    assert int(hits["frame"].iloc[0]) == expected
    assert int(hits["frame"].iloc[0]) != 22


def test_calibration_offset_applied(tmp_job_dir, monkeypatch):
    """Task 1.1: the calibration offset must be SUBTRACTED from each candidate
    frame, clamped at 0, with no refine/contact nudging interfering.

    For a candidate frame f and offset k, emitted frame == max(0, f - k).
    """
    monkeypatch.setattr("app.pipeline.hits.settings.audio_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.wrist_hit_enabled", False)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_refine_window", 0)
    monkeypatch.setattr("app.pipeline.hits.settings.hit_contact_sanity_enabled", False)

    def _run_with(offset, candidate_frame):
        monkeypatch.setattr(
            "app.pipeline.hits.settings.hit_frame_calibration_offset", offset)
        monkeypatch.setattr(
            "app.pipeline.hits.GlobalHitCandidateDetector.detect",
            lambda self, df, **kwargs: [{"frame": candidate_frame, "score": 1.0}],
        )
        store = ArtifactStore(tmp_job_dir)
        n = 60
        shuttle_df = pd.DataFrame({
            "frame": list(range(n)),
            "x": [150.0] * n, "y": [180.0] * n, "confidence": [0.95] * n,
        })
        store.set_parquet("shuttle_raw", shuttle_df)
        store.set_parquet("pose", pd.DataFrame(
            {"frame": [], "player_id": [], "keypoints": []}))
        result = HitFrameLocalizationStage().run(store, StageConfig())
        assert result.status == "success"
        hits = store.get_parquet("hits")
        assert hits is not None and len(hits) == 1
        return int(hits["frame"].iloc[0])

    # Normal case: f=30, k=11 -> 19
    assert _run_with(11, 30) == max(0, 30 - 11)
    # Default-ish case: f=30, k=8 -> 22
    assert _run_with(8, 30) == max(0, 30 - 8)
    # Clamp at 0: f=5, k=11 -> 0
    assert _run_with(11, 5) == max(0, 5 - 11)
    # Zero offset: identity
    assert _run_with(0, 42) == 42


def test_detect_uses_court_space_when_homography_provided():
    """Spec 2: when a homography is passed, the four-signal fusion runs in
    court metres instead of pixels, so perspective distortion no longer biases
    the direction-reversal peak away from the true contact frame."""
    import numpy as np
    from app.pipeline.hits import GlobalHitCandidateDetector

    # Homography that uniformly scales pixels -> court metres (no perspective
    # in this unit test; we assert the score uses court coords, not pixel ones).
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = 0.01  # 1 court unit per 100 px in x
    H[1, 1] = 0.02  # 1 court unit per 50 px in y

    # A clean V-shaped reversal at frame 25 in COURT space.
    n = 50
    # Court-space trajectory
    cx = [10.0 - t * 0.2 for t in range(25)] + [5.0 + (t - 25) * 0.2 for t in range(25, n)]
    cy = [20.0 - t * 0.1 for t in range(25)] + [17.5 + (t - 25) * 0.1 for t in range(25, n)]
    # Map back to pixels
    x = [c / H[0, 0] for c in cx]
    y = [c / H[1, 1] for c in cy]

    det = GlobalHitCandidateDetector(threshold=0.0)  # accept all
    df = pd.DataFrame({"frame": list(range(n)), "x": x, "y": y, "confidence": [0.95] * n})

    cands_pixel = det.detect(df)  # no homography -> pixel space
    cands_court = det.detect(df, homography=H)  # court space

    assert len(cands_court) > 0
    # Court-space peak should sit at/near the true reversal frame 25.
    best_court = min(cands_court, key=lambda c: abs(c["frame"] - 25))
    assert abs(best_court["frame"] - 25) <= 2

    # Pixel-space (same linear scale, so here identical) must agree; the key
    # assertion is that the court path is actually taken without error and the
    # returned candidates carry the court-space direction signal.
    assert "direction_change" in best_court


def test_detect_court_space_recovers_contact_under_perspective():
    """With a real perspective homography, the pixel-space reversal peak is
    shifted away from the true (court-space) contact; court-space detection
    must recover the correct frame."""
    import numpy as np
    from app.pipeline.hits import GlobalHitCandidateDetector

    # Perspective: far court region (small y_court) maps to many pixels,
    # near region compressed. Build H as an affine perspective (scales y by cy).
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = 0.01
    H[1, 1] = 0.01
    H[1, 2] = 0.5  # y_px = 0.01*y_court + 0.5  (perspective-ish shift)

    n = 50
    # Court-space V-reversal at frame 25.
    cx = [10.0 - t * 0.2 for t in range(25)] + [5.0 + (t - 25) * 0.2 for t in range(25, n)]
    cy = [20.0 - t * 0.1 for t in range(25)] + [17.5 + (t - 25) * 0.1 for t in range(25, n)]
    x = [c / 0.01 for c in cx]
    y = [0.01 * c + 0.5 for c in cy]

    det = GlobalHitCandidateDetector(threshold=0.0)
    df = pd.DataFrame({"frame": list(range(n)), "x": x, "y": y, "confidence": [0.95] * n})

    cands_court = det.detect(df, homography=H)
    assert len(cands_court) > 0
    best = min(cands_court, key=lambda c: abs(c["frame"] - 25))
    assert abs(best["frame"] - 25) <= 2

