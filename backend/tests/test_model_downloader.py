import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.pipeline.shared.models import ensure_model, MODEL_REGISTRY
from app.config.model_downloader import verify_all_models, download_all_models, BACKEND_MODELS


class TestEnsureModel:
    def test_ensure_model_uses_local(self, tmp_path):
        local = tmp_path / "test.pt"
        local.touch()
        reg = {"test": (local, None, None)}
        result = ensure_model("test", registry=reg)
        assert result == local

    def test_ensure_model_unknown(self):
        result = ensure_model("nonexistent_model")
        assert result is None

    def test_ensure_model_missing_no_download(self, tmp_path):
        local = tmp_path / "missing.pt"
        reg = {"test": (local, None, None)}
        result = ensure_model("test", registry=reg)
        assert result is None


class TestVerifyAllModels:
    def test_verify_all_models_returns_dict(self):
        status = verify_all_models()
        assert isinstance(status, dict)
        for name in BACKEND_MODELS:
            assert name in status, f"Missing {name} in verify output"

    def test_verify_all_models_contains_tracknet(self):
        status = verify_all_models()
        assert "tracknet" in status


class TestDownloadAllModels:
    def test_download_all_models_runs(self):
        with patch("app.config.model_downloader.ensure_model") as mock_ensure:
            mock_ensure.return_value = Path("/fake/path.pt")
            results = download_all_models(force=True)
            assert len(results) == len(BACKEND_MODELS)
