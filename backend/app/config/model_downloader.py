import os
import urllib.request
import zipfile
from pathlib import Path


# Base directory for checkpoints (relative to project root)
CKPT_DIR = Path(__file__).resolve().parents[3] / "ckpts"

# BST model: CG-AP (25 classes, seq_len=100) trained on ShuttleSet with bone joints
# Google Drive file ID from ShuttleSet BST weights folder
BST_FILE_ID = "1oM2cGM4gQRDXpcS3J5lIMDY2sBJlUvJ4"

# RTMPose ONNX model from MMPose (medium, 256x192, trained on Body8 dataset)
RTMPOSE_ONNX_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"

# HRNet-W32 ONNX model for MMPose (COCO 17 keypoints, 256x192)
HRNET_FILE_ID = "1LFUEbHB-D3WCyjzf9aSJ_V_kVB8igsnr"

# Required model files
REQUIRED_MODELS = {
    "tracknet": CKPT_DIR / "TrackNet_best.pt",
    "inpaintnet": CKPT_DIR / "InpaintNet_best.pt",
    "yolov8n": Path("yolov8n.pt"),
    "rtmpose": CKPT_DIR / "rtmpose" / "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx",
    "hrnet": CKPT_DIR / "mmpose" / "hrnet_w32_coco_256x192.onnx",
    "bst": CKPT_DIR / "bst" / "bst_CG_AP.pt",
}


def download_bst_weights(force: bool = False) -> Path:
    """Download BST-CG-AP weights from Google Drive.
    
    Args:
        force: If True, re-download even if file exists.
        
    Returns:
        Path to downloaded model file.
        
    Raises:
        ImportError: If gdown is not installed.
        RuntimeError: If download fails.
    """
    import gdown
    
    output_path = REQUIRED_MODELS["bst"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not force:
        print(f"BST weights already exist at {output_path}")
        return output_path
    
    print("Downloading BST-CG-AP weights from Google Drive...")
    
    try:
        url = f"https://drive.google.com/uc?id={BST_FILE_ID}"
        gdown.download(id=BST_FILE_ID, output=str(output_path), quiet=False)
        
        if not output_path.exists():
            raise RuntimeError(f"BST model file not found after download at {output_path}")
            
        print(f"BST weights downloaded to {output_path}")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to download BST weights: {e}")


def download_rtmpose_model(force: bool = False) -> Path:
    """Download RTMPose ONNX model from MMPose.
    
    The model is distributed as a zip file containing the ONNX model.
    
    Args:
        force: If True, re-download even if file exists.
        
    Returns:
        Path to downloaded model file.
        
    Raises:
        RuntimeError: If download fails.
    """
    output_path = REQUIRED_MODELS["rtmpose"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not force:
        print(f"RTMPose model already exists at {output_path}")
        return output_path
    
    print("Downloading RTMPose ONNX model...")
    
    zip_path = output_path.parent / "rtmpose.zip"
    
    try:
        urllib.request.urlretrieve(RTMPOSE_ONNX_URL, str(zip_path))
        
        print("Extracting ONNX model from zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            onnx_files = [f for f in zip_ref.namelist() if f.endswith('.onnx')]
            if not onnx_files:
                raise RuntimeError("No ONNX file found in the downloaded zip")
            
            # Extract the ONNX file
            for onnx_file in onnx_files:
                data = zip_ref.read(onnx_file)
                with open(output_path, 'wb') as f:
                    f.write(data)
                break  # Use the first ONNX file
        
        # Clean up zip
        zip_path.unlink(missing_ok=True)
        
        print(f"RTMPose model downloaded to {output_path}")
        return output_path
    except Exception as e:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download RTMPose model: {e}")


def download_hrnet_model(force: bool = False) -> Path:
    """Download pre-exported HRNet-W32 ONNX model from Google Drive.
    
    This model was exported from MMPose's MMPoseInferencer('human') and provides
    COCO 17-keypoint format at 256x192 resolution.
    
    Args:
        force: If True, re-download even if file exists.
        
    Returns:
        Path to downloaded model file.
        
    Raises:
        ImportError: If gdown is not installed.
        RuntimeError: If download fails.
    """
    import gdown
    
    output_path = REQUIRED_MODELS["hrnet"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not force:
        print(f"HRNet model already exists at {output_path}")
        return output_path
    
    print("Downloading HRNet-W32 ONNX model from Google Drive...")
    
    try:
        gdown.download(id=HRNET_FILE_ID, output=str(output_path), quiet=False)
        
        if not output_path.exists():
            raise RuntimeError(f"HRNet model file not found after download at {output_path}")
            
        print(f"HRNet model downloaded to {output_path}")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to download HRNet model: {e}")


def verify_all_models() -> dict[str, bool]:
    """Verify that all required model files exist.
    
    Returns:
        Dictionary mapping model name to existence status.
    """
    status = {}
    for name, path in REQUIRED_MODELS.items():
        if name == "yolov8n":
            status[name] = True  # Ultralytics auto-downloads
        else:
            status[name] = path.exists()
    return status


def download_all_models(force: bool = False) -> dict[str, Path]:
    """Download all required models.
    
    Args:
        force: If True, re-download even if files exist.
        
    Returns:
        Dictionary mapping model name to downloaded path.
        
    Raises:
        RuntimeError: If any download fails.
    """
    results = {}
    
    results["bst"] = download_bst_weights(force=force)
    results["rtmpose"] = download_rtmpose_model(force=force)
    
    # HRNet is optional (for hybrid/mmpose mode)
    try:
        results["hrnet"] = download_hrnet_model(force=force)
    except Exception as e:
        print(f"HRNet download skipped: {e}")
    
    if not REQUIRED_MODELS["tracknet"].exists():
        raise RuntimeError(f"TrackNet weights not found: {REQUIRED_MODELS['tracknet']}")
    if not REQUIRED_MODELS["inpaintnet"].exists():
        raise RuntimeError(f"InpaintNet weights not found: {REQUIRED_MODELS['inpaintnet']}")
    
    return results


if __name__ == "__main__":
    print("Verifying existing models...")
    status = verify_all_models()
    for name, exists in status.items():
        print(f"  {name}: {'OK' if exists else 'MISSING'}")
    
    print("\nDownloading missing models...")
    try:
        results = download_all_models()
        print("\nAll models downloaded successfully!")
        
        print("\nVerifying all models...")
        status = verify_all_models()
        for name, exists in status.items():
            print(f"  {name}: {'OK' if exists else 'MISSING'}")
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)
