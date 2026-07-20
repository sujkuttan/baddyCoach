import numpy as np


def test_racket_tracker_head_point_extraction():
    from app.models.racket import RacketTracker
    tr = RacketTracker.__new__(RacketTracker)
    # bbox (x1,y1,x2,y2) = (100,200,140,300); head = top-center + margin
    head = tr._head_point((100, 200, 140, 300), margin=0.1)
    # top-center x = (100+140)/2 = 120 ; y nudged up by margin*height = 0.1*100 =10 => 200-10=190
    assert abs(head[0] - 120.0) < 1e-6
    assert abs(head[1] - 190.0) < 1e-6


def test_get_racket_returns_none_when_disabled(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "racket_enabled", False)
    from app.pipeline.shared.models import get_racket
    assert get_racket() is None
