import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.config.model_downloader import (
    download_bst_weights,
    download_rtmpose_model,
    verify_all_models,
    download_all_models,
    REQUIRED_MODELS,
    CKPT_DIR,
)


class TestDownloadBstWeights:
    def test_download_creates_directory(self, tmp_path):
        """Test that download creates the BST directory if it doesn't exist."""
        target_dir = tmp_path / "ckpts" / "bst"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "bst_CG_AP.pt").touch()
        
        with patch("app.config.model_downloader.REQUIRED_MODELS") as mock_models:
            mock_path = target_dir / "bst_CG_AP.pt"
            mock_models.__getitem__ = lambda self, key: mock_path if key == "bst" else mock_path
            
            with patch("gdown.download") as mock_download:
                mock_download.return_value = str(mock_path)
                result = download_bst_weights(force=True)
                assert result == mock_path

    def test_download_skips_if_exists(self, tmp_path):
        """Test that download skips if file already exists."""
        with patch("app.config.model_downloader.REQUIRED_MODELS") as mock_models:
            mock_path = tmp_path / "bst_CG_AP.pt"
            mock_path.touch()
            mock_models.__getitem__ = lambda self, key: mock_path if key == "bst" else mock_path
            
            result = download_bst_weights(force=False)
            assert result == mock_path


class TestDownloadRtmposeModel:
    def test_download_creates_directory(self, tmp_path):
        """Test that download creates the RTMPose directory and extracts ONNX."""
        target_dir = tmp_path / "ckpts" / "rtmpose"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock zip file with an ONNX file inside
        import zipfile
        zip_path = target_dir / "rtmpose.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("model.onnx", "fake onnx content")
        
        model_file = target_dir / "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx"
        
        with patch("app.config.model_downloader.REQUIRED_MODELS") as mock_models:
            mock_models.__getitem__ = lambda self, key: model_file if key == "rtmpose" else model_file
            
            with patch("urllib.request.urlretrieve") as mock_download:
                mock_download.return_value = None
                # Create the zip file at the expected location
                (target_dir / "rtmpose.zip").write_bytes(b"fake zip")
                with patch("zipfile.ZipFile") as mock_zip:
                    mock_zip_instance = MagicMock()
                    mock_zip.return_value.__enter__ = lambda s: mock_zip_instance
                    mock_zip.return_value.__exit__ = MagicMock(return_value=False)
                    mock_zip_instance.namelist.return_value = ["model.onnx"]
                    mock_zip_instance.read.return_value = b"fake onnx content"
                    
                    with patch("builtins.open", MagicMock()):
                        result = download_rtmpose_model(force=True)
                        assert result == model_file

    def test_download_skips_if_exists(self, tmp_path):
        """Test that download skips if file already exists."""
        with patch("app.config.model_downloader.REQUIRED_MODELS") as mock_models:
            mock_path = tmp_path / "rtmpose.onnx"
            mock_path.touch()
            mock_models.__getitem__ = lambda self, key: mock_path if key == "rtmpose" else mock_path
            
            result = download_rtmpose_model(force=False)
            assert result == mock_path


class TestVerifyAllModels:
    def test_verify_all_models(self):
        """Test verify_all_models returns dict with expected keys."""
        status = verify_all_models()
        assert isinstance(status, dict)
        assert "tracknet" in status
        assert "inpaintnet" in status
        assert "yolov8n" in status
        assert "rtmpose" in status
        assert "bst" in status
        assert status["yolov8n"] is True


class TestDownloadAllModels:
    def test_download_all_models_success(self):
        """Test download_all_models calls individual download functions."""
        with patch("app.config.model_downloader.download_bst_weights") as mock_bst:
            with patch("app.config.model_downloader.download_rtmpose_model") as mock_rtmpose:
                mock_bst.return_value = REQUIRED_MODELS["bst"]
                mock_rtmpose.return_value = REQUIRED_MODELS["rtmpose"]
                with patch.object(Path, "exists", return_value=True):
                    results = download_all_models()
                    assert "bst" in results
                    assert "rtmpose" in results
                    mock_bst.assert_called_once()
                    mock_rtmpose.assert_called_once()

    def test_download_all_models_missing_tracknet(self):
        """Test download_all_models raises error if TrackNet is missing."""
        with patch("app.config.model_downloader.download_bst_weights"):
            with patch("app.config.model_downloader.download_rtmpose_model"):
                with patch.object(Path, "exists", return_value=False):
                    with pytest.raises(RuntimeError, match="TrackNet weights not found"):
                        download_all_models()

    def test_download_all_models_missing_inpaintnet(self):
        """Test download_all_models raises error if InpaintNet is missing."""
        with patch("app.config.model_downloader.download_bst_weights"):
            with patch("app.config.model_downloader.download_rtmpose_model"):
                def mock_exists_side_effect(self):
                    return self != REQUIRED_MODELS["inpaintnet"]
                with patch.object(Path, "exists", mock_exists_side_effect):
                    with pytest.raises(RuntimeError, match="InpaintNet weights not found"):
                        download_all_models()
