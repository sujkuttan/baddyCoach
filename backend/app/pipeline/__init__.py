"""
Backend pipeline package.
"""

from .shared.court import (
    COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, COURT_MODEL,
    COURT_ASPECT_RATIO, _detect_court_color_line, _correct_court_points,
    _validate_court_geometry, compute_homography, image_to_court,
    HomographySmoother, make_undistorter,
    foot_midpoint_from_pose, foot_point_from_bbox,
)
from .shared.utils import (
    get_video_info, frame_generator, detect_court_from_frame,
    compute_court_homography,
    _rule_based_shuttle_predict,
    _infer_end_reason, _is_rally_ending_shot,
    _find_dead_shuttle_window, _winner_from_shuttle_landing,
    stage_rally_stats,
)
from .shared.models import setup_models, _download_model_from_gdown, _extract_zip
from .shared.logging import PipelineLogger, logger
from .shared.core import STROKE_CLASSES, _get_gpu_batch_config

# Import stage classes for export (wrapped in try/except for environments
# where settings/model dependencies are not available)
try:
    from .court import CourtDetectionStage, CourtKeypointDetector
    from .players import PlayerTrackingStage, stitch_tracks
    from .shuttle import ShuttleTrackingStage
    from .pose import PoseEstimationStage
    from .hits import HitFrameLocalizationStage
    from .strokes import StrokeClassificationStage
    from .attribution import PlayerAttributionStage
    from .rallies import RallySegmentationStage
    from .analytics import (
        CourtPositionAnalyticsStage,
        FitnessAnalyticsStage,
        FootworkAnalyticsStage,
        TacticalAnalyticsStage,
        TechnicalAnalyticsStage,
    )
except (ImportError, Exception):
    CourtDetectionStage = None
    CourtKeypointDetector = None
    PlayerTrackingStage = None
    ShuttleTrackingStage = None
    PoseEstimationStage = None
    HitFrameLocalizationStage = None
    StrokeClassificationStage = None
    PlayerAttributionStage = None
    RallySegmentationStage = None
    CourtPositionAnalyticsStage = None
    FitnessAnalyticsStage = None
    FootworkAnalyticsStage = None
    TacticalAnalyticsStage = None
    TechnicalAnalyticsStage = None

# Export key components
__all__ = [
    # Core components
    'COURT_LENGTH',
    'COURT_WIDTH',
    'NET_HEIGHT',
    'COURT_MODEL',
    'STROKE_CLASSES',
    '_get_gpu_batch_config',
    
    # Utility functions
    'get_video_info',
    'frame_generator',
    'detect_court_from_frame',
    'compute_homography',
    'compute_court_homography',
    'image_to_court',
    'make_undistorter',
    'foot_midpoint_from_pose',
    'foot_point_from_bbox',
    
    # Stroke classification helpers
    '_rule_based_shuttle_predict',
    # Rally segmentation helpers
    '_infer_end_reason',
    '_is_rally_ending_shot',
    '_find_dead_shuttle_window',
    '_winner_from_shuttle_landing',
    'stage_rally_stats',
    
    # Model loading
    'setup_models',
    '_download_model_from_gdown',
    '_extract_zip',
    
    # Logging
    'logger',
    'PipelineLogger',
    
    # Court functions
    '_detect_court_color_line',
    '_correct_court_points',
    '_validate_court_geometry',
    'HomographySmoother',
    
    # Player tracking
    'stitch_tracks',

    # Stage classes
    'CourtDetectionStage',
    'CourtKeypointDetector',
    'PlayerTrackingStage',
    'ShuttleTrackingStage',
    'PoseEstimationStage',
    'HitFrameLocalizationStage',
    'StrokeClassificationStage',
    'PlayerAttributionStage',
    'RallySegmentationStage',
    'CourtPositionAnalyticsStage',
    'FitnessAnalyticsStage',
    'FootworkAnalyticsStage',
    'TacticalAnalyticsStage',
    'TechnicalAnalyticsStage',
]
