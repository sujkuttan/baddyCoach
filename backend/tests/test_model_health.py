import pytest

from app.pipeline.shared.models import (
    _checked_load, record_model_health, get_model_health, _model_health,
)


class DummyModel:
    def __init__(self):
        self._sd = {"enc1.weight": None, "enc5.weight": None, "out.weight": None}
    def state_dict(self):
        return dict(self._sd)
    def load_state_dict(self, state_dict, strict=False):
        missing = [k for k in self._sd if k not in state_dict]
        unexpected = [k for k in state_dict if k not in self._sd]
        return type("Incomp", (), {"missing_keys": missing, "unexpected_keys": unexpected})()


def test_checked_load_valid():
    model = DummyModel()
    status = _checked_load(model, {"enc1.weight": None, "enc5.weight": None,
                                   "out.weight": None},
                           core_prefixes=("enc",))
    assert status["loaded"] is True
    assert status["n_missing"] == 0


def test_checked_load_missing_core():
    model = DummyModel()
    status = _checked_load(model, {"other.weight": None},
                           core_prefixes=("enc",))
    assert status["loaded"] is False
    assert len(status["core_missing"]) > 0


def test_model_health_roundtrip():
    _model_health.clear()
    record_model_health("rtmpose", {"loaded": True, "error": None})
    record_model_health("tracknet", {"loaded": False, "error": "prefix mismatch"})
    health = get_model_health()
    assert "rtmpose" in health
    assert "tracknet" in health
    assert health["rtmpose"]["loaded"] is True
    assert health["tracknet"]["error"] == "prefix mismatch"


def test_model_health_defaults():
    _model_health.clear()
    health = get_model_health()
    assert health == {}


def test_record_model_health_overwrite():
    _model_health.clear()
    record_model_health("bst", {"loaded": True})
    record_model_health("bst", {"loaded": False, "error": "wrong shape"})
    health = get_model_health()
    assert health["bst"]["loaded"] is False
