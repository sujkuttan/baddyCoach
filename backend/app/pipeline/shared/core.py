"""
Core shared functionality for both colab and backend pipelines.
"""

# Import court module first (has constants used by other modules)
from .court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, COURT_MODEL,
    COURT_ASPECT_RATIO, _detect_court_color_line, _correct_court_points,
    _validate_court_geometry, compute_homography, image_to_court,
    HomographySmoother, make_undistorter, detect_court_hough_lines,
    foot_midpoint_from_pose, foot_point_from_bbox,
)

# Import utility functions
from .utils import (
    get_video_info, frame_generator, detect_court_from_frame,
    compute_court_homography,
    _rule_based_shuttle_predict,
    _infer_end_reason, _is_rally_ending_shot,
    stage_rally_stats,
)

# Import model loading
from .models import setup_models, _download_model_from_gdown, _extract_zip

# Import logging
from .logging import PipelineLogger, logger

# Stroke classes
STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]

# GPU batch configuration
def _get_gpu_batch_config(device: str) -> dict:
    """Detect GPU VRAM and return optimal batch sizes per pipeline stage."""
    tiers = [
        (12, {"yolo_chunk": 200, "yolo_batch": 16, "tracknet_chunk": 16, "rtmpose_chunk": 128, "bst_batch": 128}),
        (6,  {"yolo_chunk": 200, "yolo_batch": 16, "tracknet_chunk": 16, "rtmpose_chunk": 128, "bst_batch": 64}),
        (2,  {"yolo_chunk": 100, "yolo_batch": 8,  "tracknet_chunk": 16, "rtmpose_chunk": 64,  "bst_batch": 32}),
        (0,  {"yolo_chunk": 50,  "yolo_batch": 4,  "tracknet_chunk": 8,  "rtmpose_chunk": 32,  "bst_batch": 16}),
    ]
    cpu_cfg = {"yolo_chunk": 100, "yolo_batch": 8, "tracknet_chunk": 8, "rtmpose_chunk": 32, "bst_batch": 16}
    if "cuda" not in device.lower():
        return dict(cpu_cfg)
    try:
        import torch
        if not torch.cuda.is_available():
            return dict(cpu_cfg)
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        vram_gb = total_mem / (1024 ** 3)
        for min_gb, cfg in tiers:
            if vram_gb >= min_gb:
                return dict(cfg)
    except Exception:
        pass
    return dict(cpu_cfg)
