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
    _project_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    tracknet_model_path: Path = _project_root / "ckpts/TrackNet_best.pt"
    inpaintnet_model_path: Path = _project_root / "ckpts/InpaintNet_best.pt"
    court_kpRCNN_model_path: Path = _project_root / "ckpts/court_kpRCNN.pth"
    yolov8_model_path: Path | None = None
    rtmpose_model_path: Path | None = _project_root / "ckpts/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx"
    hrnet_model_path: Path | None = _project_root / "ckpts/mmpose/hrnet_w32_coco_256x192.onnx"
    bst_model_path: Path | None = _project_root / "ckpts/bst/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt"
    pose_model: str = "rtmpose"  # rtmpose, mmpose, hybrid

    # Environment variables
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    fps: float = 30.0
    court_length: float = 13.4
    court_width: float = 6.10

    # Shuttle detection confidence gate
    shuttle_min_conf: float = 0.30  # sub-threshold detections treated as missing

    # Shuttle trajectory cleaning (applied before any downstream consumer)
    shuttle_clean_enabled: bool = True
    shuttle_clean_min_conf: float = 0.30  # confidence gate for cleaning (matches shuttle_min_conf)
    shuttle_max_jump_px: float = 200.0  # there-and-back spike threshold
    shuttle_max_interp_gap: int = 7  # max frames to linearly interpolate across gaps
    shuttle_smooth_window: int = 3  # moving median window (0=off, 3=de-jitter)

    # Frame defaults (used when real video resolution is unavailable)
    default_frame_width: int = 1280
    default_frame_height: int = 720

    # Player tracking
    max_players: int = 2
    track_stitch_enabled: bool = True
    tracker_config_path: Path = _project_root / "backend/app/config/bytetrack_badminton.yaml"

    # Hit detection — shuttle-centric GlobalHitCandidateDetector (Section 7)
    hit_window_frames: int = 3           # ±window for velocity vectors
    hit_direction_weight: float = 0.45   # direction-change signal weight
    hit_speed_weight: float = 0.30       # speed-delta signal weight
    hit_curvature_weight: float = 0.20   # curvature signal weight
    hit_visibility_weight: float = 0.05  # visibility-transition signal weight
    hit_candidate_threshold: float = 0.62  # minimum event score to accept a candidate
    hit_min_gap_frames: int = 6          # non-maximum suppression window
    hit_refine_window: int = 16          # ±frames for pose-based contact refinement. Median frame error to labels is 8 frames, so window must exceed that.

    # Wrist-speed hit detector — pose-only fallback (from Haimantika/badminton-coach)
    wrist_hit_enabled: bool = True
    wrist_hit_min_speed: float = 0.15    # min normalised speed (px/frame at 30fps)
    wrist_hit_min_interval_s: float = 0.3  # min seconds between hits
    wrist_hit_min_conf: float = 0.30     # min keypoint confidence to use wrist
    wrist_hit_score_weight: float = 0.40 # score weight when merging with shuttle candidates

    # Ownership scoring weights (Section 10)
    # Tuned via grid search against 100 manual labels (see tune_ownership_weights.py)
    ownership_trajectory_weight: float = 0.20
    ownership_court_side_weight: float = 0.22
    ownership_proximity_weight: float = 0.18
    ownership_motion_weight: float = 0.18
    ownership_pose_feasibility_weight: float = 0.12
    ownership_turn_prior_weight: float = 0.06
    ownership_bst_weight: float = 0.07
    ownership_bst_alpha_threshold: float = 0.15
    ownership_bst_conf_min: float = 0.3
    ownership_window_frames: int = 3         # ±window for trajectory vector
    ownership_net_margin: float = 0.75       # metres — ambiguous zone around net
    ownership_prox_sigma_norm: float = 0.15  # normalised proximity scaling
    ownership_prox_sigma_meters: float = 0.75
    ownership_prox_min_pose_conf: float = 0.25
    ownership_min_pose_conf: float = 0.35    # minimum keypoint confidence
    ownership_unknown_score: float = 0.50    # default score when data missing
    ownership_strong_reach: float = 0.75     # arm-reach ratio: natural reach
    ownership_medium_reach: float = 1.25     # plausible reach upper bound
    ownership_weak_reach: float = 1.75       # max stretched reach
    ownership_alternate_score: float = 0.95  # turn prior: alternation
    ownership_same_player_score: float = 0.05
    ownership_first_hit_score: float = 0.50

    # Trajectory sub-score (YAML trajectory section)
    ownership_traj_min_shuttle_conf: float = 0.30
    ownership_traj_interp_penalty: float = 0.80

    # Court-side sub-score (YAML court_side section)
    ownership_court_net_y: float = 6.7
    ownership_court_wrong_side_score: float = 0.20

    # Motion sub-score weights (YAML motion section)
    ownership_motion_wrist_weight: float = 0.50
    ownership_motion_elbow_weight: float = 0.30
    ownership_motion_shoulder_weight: float = 0.20

    # Viterbi transition probabilities (YAML viterbi section)
    viterbi_p_alternate: float = 0.95
    viterbi_p_same: float = 0.05
    viterbi_epsilon: float = 1e-6

    # Side-specific calibration stats (YAML calibration section)
    calib_near_mean: float = 0.62
    calib_near_std: float = 0.14
    calib_far_mean: float = 0.48
    calib_far_std: float = 0.18

    # Post-attribution confidence / uncertainty (YAML confidence section)
    confidence_min_owner_confidence: float = 0.60
    confidence_uncertain_margin: float = 0.12

    # Stroke classification thresholds
    stroke_smoothing_window: int = 2  # ±neighbors
    stroke_smoothing_majority_count: int = 3
    stroke_dedup_gap_seconds: float = 0.2
    rule_based_shuttle_norm: str = "court"  # normalize shuttle by court dims for rule-based fallback
    bst_temperature: float = 1.0  # DEPRECATED: use bst_temperature_far/near instead. Global default.
    bst_temperature_far: float = 1.0   # softmax temperature for far-player strokes; >1 = softer
    bst_temperature_near: float = 1.0  # softmax temperature for near-player strokes; >1 = softer
    bst_shuttle_norm: str = "resolution"  # "resolution" (x/vid_w, y/vid_h) or "court" (x/court_length, y/court_width)
    bst_joint_norm: str = "bbox"  # "bbox" (diagonal + center_align, as in ShuttleSet) or "court" (homography court-space)
    bst_bbox_margin: float = 0.15  # expand keypoint bbox by this fraction per side; compensates for keypoint bboxes being ~30% tighter than detection bboxes
    joint_velocity_amplification: float = 0.7  # >0 amplifies bone vectors by joint motion (adds temporal discriminability)
    bst_adapt_batchnorm: bool = False  # use batch stats for BN layers (helps court-space norm adapt)
    bst_min_clip_frames: int = 15  # minimum real frames per clip; prevents zero-padded dominance
    bst_prior_correction_enabled: bool = True  # enabled with bias from 327 clips (2025-07-01 run, bbox-norm fix, keypoint-bbox norm)
    bst_prior_correction_strength: float = 0.75  # α; 0 = off (reproduces pre-Spec-5 output)
    bst_logit_bias_path: Path | None = _project_root / "ckpts/bst/bst_logit_bias.json"
    bst_prior_min_clips: int = 30  # min clips for self-calibration fallback
    bst_clip_boundary: str = "hit_start"  # "hit_start" (frame 0 = hit) or "midpoint" (midpoint-to-midpoint + resample)
    bst_validation_level: str = "error"  # "off" | "warn" | "error" — BST input tensor validation; set to "error" during debugging for loud failures

    # Attributed player lookback
    attribution_lookback_frames: int = 5
    attribution_bst_min_conf: float = 0.3

    # Rally segmentation thresholds
    rally_gap_threshold: int = 90
    rally_min_shots: int = 3
    rally_ending_gap_primary: int = 90
    rally_ending_gap_high_conf: int = 25
    rally_ending_gap_net: int = 45
    rally_ending_high_conf_min: float = 0.6
    rally_dead_frames: int = 25  # min consecutive frames with shuttle speed ≈ 0 to declare rally dead
    rally_dead_speed_px: float = 4.0  # per-frame shuttle displacement below this = "dead"
    rally_winner_search_frames: int = 150  # frames past rally_end to scan for landing/dead window
    rally_winner_min_landing_conf: float = 0.30  # min shuttle conf for a point to count as true landing
    rally_winner_degenerate_warn: bool = True  # warn + fall back if all rallies resolve to one player

    # Court corner fallback (proportional to frame dimensions)
    court_corner_margin_x: float = 0.08
    court_corner_top_y: float = 0.28
    court_corner_bottom_y: float = 0.72

    # Footwork analytics
    footwork_jump_filter_pixels: int = 500
    footwork_recovery_threshold_meters: float = 0.3
    footwork_recovery_lookahead_frames: int = 30
    footwork_split_step_enabled: bool = True
    footwork_split_step_drop_frac: float = 0.02  # hip-y drop fraction for split-step detection

    # Trust / Data quality
    quality_shuttle_conf_thr: float = 0.5
    quality_min_shots_tactical: int = 15
    quality_max_fallback_patterns: float = 0.30
    quality_min_stroke_conf: float = 0.30  # below this → penalized + distrusted
    model_max_missing_frac: float = 0.05

    # Report / labeling
    report_include_logits: bool = True   # embed per-shot 25-logits for label-driven calibration
    label_pre_roll_s: float = 0.7        # seconds before the hit for ts_start

    # Physics consistency gate (Spec 6)
    physics_gate_enabled: bool = True
    physics_window_frames: int = 12       # post-contact analysis window (~0.4s @30fps)
    physics_min_valid: int = 4            # min real shuttle points in window to use physics
    physics_quality_min: float = 0.35     # below this, defer entirely to BST
    physics_min_conf_override: float = 0.30  # skip physics override when BST conf below this
    physics_speed_fast_mps: float = 8.0   # court-space speed thresholds (homography valid)
    physics_speed_slow_mps: float = 3.0
    physics_speed_fast_norm: float = 0.45   # calibrated from phone footage — px_speed_per_s ~0.2-1.1
    physics_speed_slow_norm: float = 0.18   # ~half of median observed ~0.35
    physics_zone_front: float = 0.33      # court_x fraction: front court
    physics_zone_back: float = 0.66
    physics_cross_court_dx: float = 0.30  # normalized lateral travel for cross_court cue
    physics_agree_boost: float = 0.5      # confidence boost weight when BST & physics agree
    physics_max_override_frac: float = 0.0  # disabled — physics agrees with BST < 3% of the time, actively harmful
    physics_contact_search_window: int = 3    # ±frames to locate true contact frame
    physics_contact_overhead_frac: float = 0.15  # wrist above shoulder by this x torso → overhead
    physics_contact_side_frac: float = 0.30     # wrist within this x torso of shoulder → side

    # Hierarchical classifier (family-level structural prior)
    hierarchical_enabled: bool = True
    hierarchical_penalty: float = 1.5

    # Confusion-pair correction (within-family pairwise disambiguation)
    confusion_pair_enabled: bool = True
    confusion_pair_boost: float = 0.3

    # MMAction2 adapter settings (optional ensemble with BST)
    mmaction2_enabled: bool = False  # set True to enable MMAction2 ensemble
    mmaction2_mode: str = "posec3d"  # "posec3d" (skeleton), "slowfast" (RGB), "pytorchvideo" (light RGB)
    mmaction2_ensemble_weight: float = 0.3  # weight for MMAction2 in (1-w)*BST + w*MMAction ensemble
    mmaction2_seq_len: int = 48  # PoseC3D default clip length
    mmaction2_num_classes: int = 25  # ShuttleSet class count
    bst_n_classes: int = 25  # ShuttleSet class count (used by adapter)

    # Context fusion layer (soft logit nudge before physics gate)
    fusion_enabled: bool = True
    fusion_shuttle_weight: float = 0.15
    fusion_zone_weight: float = 0.10
    fusion_height_weight: float = 0.10
    fusion_context_weight: float = 0.05
    fusion_logit_clip: float = 2.0

    # Court geometry reliability
    geometry_max_trapezoid_ratio: float = 0.92  # top_width/bottom_width threshold; >0.92 → rectangle → unreliable

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
