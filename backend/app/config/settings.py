from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
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
    bst_model_path: Path | None = Path("ckpts/bst/bst_CG_AP.pt")
    pose_model: str = "rtmpose"  # rtmpose, mmpose, hybrid

    # Environment variables
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    fps: float = 30.0
    court_length: float = 13.4
    court_width: float = 5.18

    # Frame defaults (used when real video resolution is unavailable)
    default_frame_width: int = 1280
    default_frame_height: int = 720

    # Hit detection weights & thresholds
    hit_trajectory_weight: float = 0.4
    hit_speed_weight: float = 0.3
    hit_proximity_weight: float = 0.2
    hit_swing_weight: float = 0.1
    hit_confidence_threshold: float = 0.3
    hit_dedup_gap_seconds: float = 0.1

    # Stroke classification thresholds
    stroke_smoothing_window: int = 2  # ±neighbors
    stroke_smoothing_majority_count: int = 3
    stroke_dedup_gap_seconds: float = 0.2

    # Attributed player lookback
    attribution_lookback_frames: int = 5

    # Rally segmentation thresholds
    rally_gap_threshold: int = 60
    rally_min_shots: int = 3
    rally_ending_gap_primary: int = 45
    rally_ending_gap_high_conf: int = 25
    rally_ending_gap_net: int = 15
    rally_ending_high_conf_min: float = 0.6

    # Court corner fallback (proportional to frame dimensions)
    court_corner_margin_x: float = 0.08
    court_corner_top_y: float = 0.28
    court_corner_bottom_y: float = 0.72

    # Footwork analytics
    footwork_jump_filter_pixels: int = 500
    footwork_recovery_threshold_meters: float = 0.3
    footwork_recovery_lookahead_frames: int = 30

    # Trust / Data quality
    quality_shuttle_conf_thr: float = 0.5
    quality_min_shots_tactical: int = 15
    quality_max_fallback_patterns: float = 0.30
    model_max_missing_frac: float = 0.05

    # Shot context / pressure
    pressure_time_s: float = 0.9
    pressure_dist_m: float = 2.5
    pattern_lookahead_k: int = 2

    # Patterns
    pattern_min_samples: int = 5
    pattern_excess_loss: float = 0.15
    pattern_loss_floor: float = 0.45

    # Technique reference
    technique_reference_tier: str = "intermediate"
    technique_min_history_sessions: int = 3
    technique_pressure_delta_deg: float = 8.0

    # Progress
    progress_default_window: int = 5

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
