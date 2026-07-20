import numpy as np

from app.pipeline.shared.models import get_racket


def test_racket_disabled_yields_none(monkeypatch):
    from app.config.settings import settings

    monkeypatch.setattr(settings, "racket_enabled", False)
    assert get_racket() is None


def test_racket_motion_score_none_fallback():
    from app.pipeline.shared.ownership_scorer import racket_motion_score

    assert racket_motion_score(None, hit_idx=1) == (0.5, 0.5)


def test_proximity_wrist_only_when_no_racket():
    from app.pipeline.shared.ownership_scorer import normalized_proximity_score

    near_kps = np.random.rand(17, 3)
    far_kps = np.random.rand(17, 3)
    ns, fs = normalized_proximity_score(
        shuttle_px=np.array([10.0, 10.0]),
        shuttle_court=None,
        near_kps=near_kps,
        far_kps=far_kps,
        near_bbox_h=100.0,
        far_bbox_h=100.0,
        H_arr=None,
    )
    assert 0.0 <= ns <= 1.0
    assert 0.0 <= fs <= 1.0
    assert np.isfinite(ns) and np.isfinite(fs)


def test_racket_enabled_default_true():
    from app.config.settings import settings

    assert settings.racket_enabled is True
