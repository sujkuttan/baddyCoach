from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data")
    jobs_dir: Path = Path("data/jobs")
    max_video_length_seconds: int = 3600
    supported_formats: list[str] = ["mp4", "mov", "avi"]
    gpu_enabled: bool = False
    processing_fps: int = 30
    court_detection_fps: int = 1
    sample_rate: int = 0  # 0=auto (10fps), 1=every frame, 2=every 2nd, etc.

    # Model paths
    tracknet_model_path: Path = Path("ckpts/TrackNet_best.pt")
    inpaintnet_model_path: Path = Path("ckpts/InpaintNet_best.pt")
    court_kpRCNN_model_path: Path = Path("ckpts/court_kpRCNN.pth")
    yolov8_model_path: Path | None = None
    rtmpose_model_path: Path | None = Path("ckpts/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx")
    hrnet_model_path: Path | None = Path("ckpts/mmpose/hrnet_w32_coco_256x192.onnx")
    bst_model_path: Path | None = Path("BST/weight/bst_CG_JnB_bone_merged.pt")
    pose_model: str = "rtmpose"  # rtmpose, mmpose, hybrid

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def device(self) -> str:
        if self.gpu_enabled:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"


settings = Settings()
