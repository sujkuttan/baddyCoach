# BaddyCoach - Agent Guidance

## Architecture Overview

**16-stage FastAPI pipeline** processing badminton videos into coaching reports:

```
court_detection → player_tracking → shuttle_tracking → pose_estimation → 
hit_frame_localization → stroke_classification → rally_segmentation → 
player_attribution → rally_finalization → shot_context → 
court_position_analytics → footwork_analytics → fitness_analytics → 
tactical_analytics → technical_analytics → data_quality
```

**Key components:**
- Backend: FastAPI with WebSocket progress tracking
- Frontend: React + TypeScript + Vite dashboard
- ML models: YOLOv8, RTMPose, TrackNetV3, BST (Badminton Stroke Transformer)
- Coaching engine: YAML rule-based system (33+ rules in `backend/app/shuttle_coach/feedback/rules.yaml`)

## Critical Architecture Notes

### ✅ Stage Ordering: rally → attribution (Viterbi needs rally data)
- **Current order:** `rally_segmentation` (index 6) → `player_attribution` (index 7)
- **Rationale:** Viterbi HMM decoder requires rally boundaries to assign owners per-rally; attribution after rally is the correct ordering
- **Historical note:** An earlier heuristic attribution did not need rally data and ran before rally_segmentation; this was correct for that approach but incompatible with the current Viterbi-based method

### ✅ BST Model Integration (Fixed)
- **Issue:** BST likely not running due to sequence-length mismatch
- **Details:** Clip uses hardcoded `SEQ_LEN=30` (`strokes.py:14`), but model expects `seq_len=100`
- **Impact:** 20-40% "unknown" predictions, falls back to rule-based classification (`bst.py:152-160`)
- **Fix:** Read `classifier.seq_len` in `strokes.py` instead of constant; dynamic n_classes detection; temperature scaling (P6)

### ✅ Model Path Inconsistency (Fixed)
- **Settings path:** `BST/weight/bst_CG_JnB_bone_merged.pt` (`settings.py:22`)
- **Downloader path:** `ckpts/bst/bst_CG_AP.pt` (`model_downloader.py:27`)
- **Impact:** Configured path doesn't exist after download, silent fallback
- **Fix:** CKPT_DIR anchored via `parents[5]`; settings paths made absolute; `get_bst()` uses `ensure_model()` return path; stale 35-class ckpt copies removed

### ✅ TrackNetV3 Architecture (Fixed — 2025-06-29)
- **Was:** Backend used VGG-style backbone (9→1) incompatible with checkpoint (27→8 custom UNet)
- **Fix:** Replaced VGG `TrackNetV3Backbone` with colab's `TrackNetV3Model` (custom UNet, 27→8 matching checkpoint)
- **Changes:** 9-frame windows → 27 channels input; 8-channel output with first channel used for peak extraction; sigmoid activation on heatmap; load_backbone accepts 27-channel weights; `_build_9frame_window` replaces `_build_3frame_window`
- **Impact:** Backend now loads `TrackNet_best.pt` and `InpaintNet_best.pt` successfully, producing valid shuttle detections

### ⚠️ RTMPose Bug (Fixed)
- **Was:** x/y rescale divisors swapped, hardcoded `"input"` instead of `self.input_name`
- **Fix:** Corrected dimension order (height=256, width=192) and `self.input_name` from model metadata

### ✅ Recovery Time Units (Fixed)
- **Was:** `threshold = 0.3` in pixels, but recovery distances in meters
- **Fix:** Both query points (`com_points`) and base position converted to court-space via homography before threshold comparison; jump filter uses 2.0m in court-space

### ✅ Hit Detection Normalization (Fixed)
- **Was:** All four evidence signals used `score / score.max()`, causing dynamic-range collapse in long videos
- **Fix:** 95th-percentile normalization — `m = np.percentile(score, 95); score / (m + 1e-6)` — robust to extreme frames

### ✅ BST Second-Best Threshold (Fixed)
- **Was:** `SECOND_BEST_THRESHOLD = 0.05`, allowing any >5% alternative to override `unknown`, producing erratic predictions
- **Fix:** Raised to `0.3` so only meaningfully-confident alternatives (>30%) replace `unknown`

### ✅ Temporal Smoothing Scope (Fixed)
- **Was:** Window majority vote overwrote all low-confidence predictions, including determinate ones
- **Fix:** Only `unknown` strokes are smoothed; determinate predictions (even low-confidence) are preserved untouched

### ✅ Technique Score Overhaul (Fixed)
- **Was:** Single-frame `_evaluate_shot` fallback with 2 features (elbow extension, shoulder angle); no coaching rules consumed technique data
- **Fix:** Removed `_evaluate_shot` entirely; `_analyze_swing_mechanics` now uses 5 temporal features: elbow extension, peak shoulder angle, hip-shoulder separation, knee flexion (stroke-type-specific bounds), follow-through displacement. Technique scores wired into `analyze_from_pipeline` with 8 YAML coaching rules.

### ✅ Ownership-Based Post-Attribution Consistency Check (Added — 2025-06-30)
- **What:** Compares BST's internal AimPlayer attention (raw cosine sims from `BST_CG_AP.forward()`) against the final external owner assigned by Viterbi. Flags conflicts when the model's own attention focus disagrees with the pipeline-assigned owner.
- **Raw sims exposed:** `bst_model.py` now stores `_last_p1_sim` and `_last_p2_sim` (cos(p0/p1_shuttle_CLS, shuttle_CLS)) alongside `_last_alpha`. These propagate through `bst.py` as `aim_attention_p0`/`aim_attention_p1` in the 6-tuple results `(stroke_type, confidence, raw_class_id, alpha, aim_attention_p0, aim_attention_p1)`.
- **Per-shot fields in shots_df:**
  - `aim_attention_p0`, `aim_attention_p1` — raw cosine similarities (stored in `strokes.py`)
  - `attention_alpha_owner` — derived from alpha: `"far"` if alpha > 0.5, `"near"` if alpha < 0.5, `None` if alpha == 0.5
  - `attention_owner_match` — `True` if `attention_alpha_owner == side`, `False` if they disagree, `None` if alpha is 0.5 or side missing
- **Use case:** `attention_owner_match=False` flags bad clips, wrong p0/p1 ordering, or hit-frame errors for debugging.
- **4 new tests** + 1 updated test. 350 pass total.

### ✅ BST AimPlayer Alpha for Attribution (Fixed — 2025-06-29)
- **Was:** Player attribution Tier 1 only used `shuttleset_class_id` prefix (Top_/Bottom_) to determine the hitter, gated at `attribution_bst_min_conf=0.5`. Mean BST confidence ~0.33, so only 36/264 shots (13.6%) got model-based attribution; 76% fell to heuristic tiers.
- **Fix:** Surfaces AimPlayer alpha from `BST_CG_AP.forward()` (internal cosine-similarity weighting between each player's shuttle CLS token) via `self._last_alpha`. `predict_from_clips` now returns 4-tuples: `(stroke_type, confidence, raw_class_id, alpha)` where alpha ∈ [0,1] (>0.5 = far player).
- **Changes:** `bst_model.py` stores alpha; `bst.py` propagates through all paths; `strokes.py` stores `aimplayer_alpha` per shot; `settings.py` lowered `attribution_bst_min_conf` 0.5→0.3; `attribution.py` uses alpha as primary signal (|α-0.5| > 0.15) and class_id as fallback.
- **Impact:** Alpha-based attribution catches shots with class_id=0 (unknown) when the model internally knows which player hit it. Combined with the lowered threshold, estimated BST coverage improves from 13.6% → ~50%+ of all shots.

### ✅ BST Bottom_ Prefix Leak (Fixed)
- **Was:** `map_to_coach_class` returned `Bottom_smash` for the near player but `smash` for the far player — every exact-string consumer (rally end-reason, technique bounds, tactical distribution, frontend charts) broke for near-player strokes
- **Fix:** `map_to_coach_class` now returns bare stroke type for both players; side is preserved in `shuttleset_class_id` and available via `get_shuttleset_class_info`

### ✅ Upload Zero-Byte Bug (Fixed)
- **Was:** `routes.py` called `await file.read()` at line 268 (validation) and again at line 294 (write) — second read on consumed `UploadFile` returned `b""`, writing empty videos
- **Fix:** Reuse the `content` buffer from the validation read; dropped the redundant second read

### ✅ GPU OOM on T4 (Fixed)
- **Fix (commit `080eb9b`):** `gpu_batch.py` tiers reduced — ≥12GB: `yolo_chunk=200, yolo_batch=16, tracknet_chunk=16, rtmpose_chunk=128`. `colab/pipeline.py` `BATCH_SIZE`: 500→300. `torch.cuda.empty_cache()` between batches.

### ✅ Multi-Signal Ownership + Viterbi HMM (Added — 2025-06-29)
- **Was:** Attribution used a heuristic cascade: Tier 1 (BST AimPlayer alpha / class_id prefix), Tier 2 (racket proximity), Tier 3 (greedy rally alternation), Tier 4 (fallback). Tier 3/4 had no physics or pose-awareness — shuttle direction + alternation alone.
- **Now:** Six sub-scores (trajectory_ownership, court_side_feasibility, normalized_proximity, racket_motion, pose_contact_feasibility, initial_turn_prior) weighted and combined per-shot. Emissions fed into per-rally Viterbi HMM (`p_alternate=0.95`, `p_same=0.05`).
- **New file:** `backend/app/pipeline/shared/ownership_scorer.py` — 6 sub-score functions + `OwnershipScorer` class + `ViterbiConfig` + Viterbi decoder.
- **Restructured attribution.py:** Old Tiers 2-4 removed; `PlayerAttributionStage.run()` now calls `OwnershipScorer.score()` per shot, runs Viterbi per rally, and sets `owner_uncertain` flag.
- **Settings:** 16 new fields in `Settings` (`trajectory_*`, `court_side_*`, `motion_*`, `viterbi_*`, `calibration_*`, `confidence_*`) matching YAML config.
- **Sub-score details:**
  - `trajectory_ownership_score` — court-space cosine similarity of `v_in→to_player` and `v_out→away_from_player`
  - `court_side_feasibility_score` — per-side logic: near player must be on near side of net, far player on far side
  - `normalized_proximity_score` — court-coordinate distance `exp(-dist_m / sigma_meters)` with bbox-pixel fallback
  - `racket_motion_score` — wrist/elbow/shoulder angular velocities (0.50/0.30/0.20 weights), central difference at hit frame
  - `pose_contact_feasibility_score` — wrist-to-shuttle distance / arm length tiers (<0.75→1.0, <1.25→0.7, <1.75→0.4, ≥1.75→0.1)
  - `initial_turn_prior_score` — unchanged (already matched spec)
- **Side-specific calibration:** `near_z = (near_raw - near_mean) / near_std`, sigmoid, renormalize — applied as last step in `OwnershipScorer.score()`.
- **Uncertainty flag:** `owner_uncertain = True` if owner score < 0.60 or near/far gap < 0.12.
- **Dead code removed:** `_wrist_from_kps`, `_elbow_from_kps`, `_shoulder_from_kps`, `_arm_length`, `_normalize_by_p95`.
- **Commit:** `8b8f701` (12 files, +1737 lines, new `ownership_scorer.py`). All 313 tests pass.

### Hardware-Aware Testing
- Auto-skip based on RAM/GPU/model availability (`backend/tests/conftest.py`)
- **Markers:** `gpu`, `cpu_only`, `model`, `memory_intensive`, `slow`, `integration`
- **Minimum requirements:** 4GB RAM, CUDA GPU for GPU tests, model checkpoints

### Test Structure
- 350 tests (core), 7 skipped (hardware-dependent), 0 failed
- Most tests use synthetic inputs and mocked models
- Integration tests require real model checkpoints

### Key Test Files
- `backend/tests/test_api.py` - API endpoint tests
- `backend/tests/test_coach.py` - Coach engine tests
- `backend/tests/test_real_pipeline.py` - End-to-end pipeline tests

## Development Commands

### Backend Development
```bash
# Run tests with hardware awareness
cd backend
python -m pytest

# Run specific test category
python -m pytest -m "gpu"      # GPU tests
python -m pytest -m "model"    # Model tests (need checkpoints)
python -m pytest -m "slow"     # Long-running tests

# Run integration tests
python -m pytest -m "integration"
```

### Frontend Development
```bash
cd frontend
npm run dev    # Start dev server
npm run lint   # ESLint
npm run build  # Build production
```

### Model Management
```bash
# Download required checkpoints
cd backend
python app/config/model_downloader.py

# Required checkpoints:
- ckpts/TrackNet_best.pt
- ckpts/rtmpose/rtmpose-m_8xb64-270e_coco-256x192.onnx
- BST/weight/bst_CG_JnB_bone_merged.pt
```

## Configuration & Environment

### ⚠️ Critical Configuration Issues
- **GPU:** `gpu_enabled=False` by default, CPU-only `onnxruntime` in requirements
- **Magic numbers:** Hardcoded FPS=30.0, court dims, thresholds scattered across ~6 files

### Environment Setup
```bash
# Create .env file with:
GEMINI_API_KEY=your_api_key_here

# Note: Backend has no authentication - anyone can upload/list/fetch jobs
```

## Common Pitfalls

### ⚠️ Model Loading Failures
- TrackNet crashes silently to zeros if loading fails (`tracknet.py:223-227`)
- BST falls back to rule-based classification when sequence mismatch
- RTMPose transpose bug used to corrupt all pose-derived analytics (fixed)

### ⚠️ Data Quality Issues
- Synthetic detections (`_generate_synthetic_detections`) presented as real (removed — pipeline now fails early with error)
- Rule-based strokes masquerading as model output
- No data quality flags in coaching reports

### ✅ Architecture Conflicts (Resolved)
- Two coaching engines merged into `shuttle_coach/`; `coach/engine.py` and `coach/rules.yaml` deleted
- Colab pipeline reduced from ~1,966 to ~1,411 lines — removed `_prepare_stroke_classification`, `CoachEngine`, shuttle-coach functions; uses `analyze_from_pipeline`
- Model abstraction layer created: `shared/models.py` with lazy singleton registry (`get_yolov8`, `get_tracknet`, `get_rtmpose`, `get_bst`)

### ⚠️ Security & Reliability
- No authentication on any endpoint
- Videos stored indefinitely with no cleanup
- `torch.load(..., weights_only=False)` - pickle deserialization risk

## Debugging & Troubleshooting

### Model Integration Issues
```bash
# Check if checkpoints exist
ls -la ckpts/TrackNet_best.pt
ls -la ckpts/rtmpose/rtmpose-m_8xb64-270e_coco-256x192.onnx
ls -la BST/weight/bst_CG_JnB_bone_merged.pt
```

### Pipeline Issues
- **Stage failures:** Check WebSocket progress at `/api/jobs/{job_id}/progress`
- **Data quality:** no synthetic detections (removed — pipeline fails early with error)
- **Performance:** CPU-only path means minutes-to-hours per video

### Test Failures
```bash
# Run with verbose output
python -m pytest backend/tests/test_api.py -v

# Skip hardware-dependent tests
python -m pytest -m "not gpu and not model"
```

## Key Files & Entry Points

### Backend Entry Points
- `backend/app/main.py` - FastAPI app
- `backend/app/api/routes.py` - All API endpoints
- `backend/app/shuttle_coach/engine.py` - Coaching logic
- `backend/app/pipeline/base.py` - Pipeline base class

### Model Files
- `backend/app/models/yolov8.py` - Person detection/tracking
- `backend/app/models/rtmpose.py` - Pose estimation
- `backend/app/models/tracknet.py` - Shuttle tracking
- `backend/app/models/bst.py` - Stroke classification (BST)
- `backend/app/models/mmaction_adapter.py` - MMAction2 adapter (PoseC3D/SlowFast ensemble)

### Configuration
- `backend/app/config/settings.py` - Model paths, thresholds (16 ownership/Viterbi fields, 6 MMAction2 fields)
- `backend/app/config/gpu_batch.py` - GPU batch sizing
- `backend/app/shuttle_coach/feedback/rules.yaml` - 33+ coaching rules
- `backend/app/pipeline/shared/ownership_scorer.py` - Multi-signal ownership scoring + Viterbi HMM

## Migration Notes

### ⚠️ Colab Pipeline Parity
- Colab has ~2,000-line reimplementation of backend
- BST adapts to detected sequence length (`colab/pipeline.py:1348-1350`)
- Coach rules duplicated (backend reads YAML, colab hardcodes)

### ✅ Shuttle Coach Endpoint
- **Was:** Broken — required `player_detections.parquet` but backend stores `players.json`
- **Fix:** Removed endpoint

### ✅ Debug Logging Instrumentation (Fixed — 2025-06-25)
- **Was:** Only `logger.info()` stage summaries and `print()` in model files; no structured capture of model I/O for post-mortem
- **Fix:** Added `debug_level` field to `StageConfig` (0-3); full softmax distribution captured in `debug_bst_outputs.parquet`; per-frame hit scores in `debug_hit_scores.parquet`; clip construction metadata in shots.parquet columns (`clip_n_frames`, `clip_n_missing_bbox`, `clip_n_missing_pose`); attribution tier tracking in `attribution_tier` column. Migrated all `print()` calls in `bst.py` to `logger.info/warning/error`.

### ✅ Rule-Based Classifier Normalization (Fixed — 2025-06-25)
- **Was:** Clip shuttle normalized by court dims (13.4, 6.1) but thresholds tuned for pixel-space (1920×1080) → `end_y` always negative → lift/drop/net_shot can never trigger → all 69 fallbacks predict "drive"
- **Fix:** `_rule_based_predict` now denormalizes shuttle by court dims then renormalizes by video dims; uses only post-hit half of trajectory to avoid V-shaped between-2-hits averaging. `_build_clip` now passes `vid_w`, `vid_h`, `court_length`, `court_width` in clip dict.

### ✅ Temporal Bbox Interpolation (Fixed — 2025-06-25)
- **Was:** Per-frame YOLO tracking → 166 unique track IDs → `det_bbox_lookup` fails for 30-40% of frames → joints normalized with fallback keypoint bbox → garbled BST features
- **Fix:** Added `_interpolate_bboxes()` in `_build_clip` that linearly interpolates bbox for missing frames per player. Tracks missing bbox/pose counts in `_debug_clip` stats.

### ✅ Temporal Smoothing Scope (Fixed — 2025-06-25, Revised)
- **Was:** `if stype != "unknown": continue` — only unknown strokes smoothed; low-confidence "drive" (conf=0.089) never corrected
- **Fix (initial):** Smooth any stroke with confidence < 0.2, not just "unknown"
- **Fix (revised):** Reverted to unknown-only smoothing. The expanded scope caused rule-based "net_shot" bias (78 shots, conf~0.22) to overwrite 13 determinate BST predictions (lift, smash, short_serve, etc.) to net_shot via majority vote. Determinate predictions, even low-confidence, are preserved to avoid rule-based neighborhood dominance.

### ✅ Rally Winner Threshold (Fixed — 2025-06-25)
- **Was:** `_infer_end_reason` required conf ≥ 0.5 for "winner"; max BST conf 0.633 → 13/14 rallies ended in "unforced_error"
- **Fix:** Lowered winner threshold to 0.3; added speed-based winner detection (smash > 8 m/s = winner); passed shuttle speed to `_infer_end_reason`

### ✅ Re-run Validation (2025-06-25, new 5-min video with fixes)
- **Bbox interpolation (Fix 2) is the single biggest win:** missing bbox 199→0 per clip; player balance 27%/73%→50%/50%; player_1 BST coverage 22%→69%
- **BST class diversity:** 10 classes active (was 8), including **drop** for first time
- **BST shots:** 122/200 (61%), up from 108/200 (54%)
- **Rule-based:** 78/200 (39%), down from 92/200 (46%), still 78/78 → "net_shot"
- **Mean confidence unchanged** (~0.22), needs temperature scaling re-investigation
- **15/25 classes still never activated** — model can't predict 0-2, 6-12, 18, 20-22, 24

### ✅ PipelineLogger Formatting (Fixed — 2025-06-25)
- **Was:** `logger.info("Attribution tiers: %s", tier_counts)` — PipelineLogger.info() takes only `message` as positional, causing TypeError
- **Fix:** Changed to `logger.info("Attribution tiers", tiers=str(tier_counts))`

### ✅ Debug BST Output Persistence (Fixed — 2025-06-25)
- **Was:** `bst_debug_collector` list collected per-shot debug info but was never saved to parquet — data existed in memory only
- **Fix:** After `predict_from_clips`, save `artifacts.set_parquet("debug_bst_outputs", df)` when debug_level >= 1

### ✅ Full Logits Capture for Temperature Calibration (Fixed — 2025-06-26)
- **Was:** Debug collector captured only `logit_class_0`, `logit_max`, and `top5` — insufficient for temperature recalibration. Cached T=1.4224 was computed from 12-class test data with broken InpaintNet features, so it's invalid for the fixed pipeline.
- **Fix:** Added `logits_all` field (JSON string of full 25-class logits vector) to each debug entry in `bst.py:328`. This enables post-hoc calibration via:
  ```python
  df = pd.read_parquet("debug_bst_outputs.parquet")
  logits = np.array([json.loads(s) for s in df["logits_all"]])
  labels = df["pred_class_id"].values
  T = BSTClassifier.compute_optimal_temperature(logits, labels)
  BSTClassifier._save_temperature(T)
  ```
- **`_load_temperature`** updated with inline docstring recipe and startup warning that cached temperature may be stale after InpaintNet fix.

### ⚠️ Double InpaintNet + Missing Homography Conversion (Fixed — 2025-06-26)
- **Issue:** Shuttle coordinates had range x ∈ [-7.32, 14.14] far beyond court (13.4×6.1m). `_build_clip` divided these by court_length/court_width (treating them as meters), producing garbage inputs to BST.
- **Root cause:** Two separate bugs compounded:
  1. **Double InpaintNet:** TrackNetV3 internally runs `_rectify_trajectory` (linear interpolation + moving average smoothing). The colab pipeline then ran a **second** `InpaintNet` instance on the already-rectified pixel coords, completely overwriting them with garbage values from a checkpoint trained on a different coordinate space.
  2. **Missing homography:** Neither pipeline applied `image_to_court(homography, (x, y))` to TrackNet's pixel output. The shuttle coordinates (pixels) were divided directly by court_length (13.4m), e.g., 1920px / 13.4m ≈ 143 — until the double InpaintNet warped them to intermediate garbage values.
- **Impact:** Feature quality collapsed — JnB and shuttle stats nearly identical across all classes (zero_frac=0.0535, jnb_min=-0.569, jnb_max=0.682 for class_23, other_BST, and unknown). Model saw negligible discriminative signal.
- **Fix (colab pipeline `colab/pipeline.py:972-988`):** Removed the second InpaintNet pass entirely. Added `image_to_court(H, (x, y))` to convert pixel → court-space meters before storing shuttle data.
- **Fix (backend `backend/app/pipeline/strokes.py:121-128`):** Added `image_to_court(homography, (sx, sy))` in `_build_clip` alongside the existing foot position homography conversion.

### ✅ Colab Re-run with Double InpaintNet + Homography Fix (2025-06-26)
- **Expected:** Shuttle range should shrink to ±6.7m × ±3.05m (court dimensions). Feature diversity should increase as JnB/shuttle inputs are no longer garbage. BST should escape the 49% short_serve bias.

## 2025-06-28: Pipeline Quality Fixes (Batch 2)

### ✅ Scene-Cut Rally Segmentation (Fixed — 2025-06-28)
- **Was:** Rally segmentation relied on dead-shuttle windows (25+ consecutive frames with near-zero speed). For pause-record videos (recording paused between points), no usable dead zones exist → false hits fragment rallies.
- **Fix:** Added scene-cut detection in `rallies.py` — detects recording discontinuities via shuttle position jumps (>50× median displacement). Also fixed `_find_dead_shuttle_window` in `utils.py` to respect its `min_gap_frames` parameter.

### ✅ Player Attribution Balance Flip (Fixed — 2025-06-28)
- **Was:** Per-frame YOLO tracking → shuttle_direction (`dy>0 → player_1`) systematically favored one player (73/27 split) when camera angle biased far-player dominance.
- **Fix:** Per-rally balance check in `attribution.py`: if >60% of shuttle_direction-assigned shots go to one player, flip all assignments in that rally. Side mapping flips alongside player_id.

### ✅ Rule-Based Predictor: max_speed Thresholds (Fixed — 2025-06-28)
- **Was:** `_rule_based_predict` used `mean_speed` → all 78 fallbacks predicted "net_shot" because `mean_speed < 0.03` is overbroad. Fallback defaulted to "net_shot" instead of "unknown".
- **Fix:** Rewrote with `max_speed` thresholds: checks fast strokes first (smash >0.08, drive >0.06), then direction/endpoint for slower strokes (clear, drop, lift). Falls back to "unknown" instead of defaulting to a single class.

### ✅ Physics Gate: Low-Confidence BST Skip (Fixed — 2025-06-28)
- **Was:** `apply_physics_ensemble` overrode BST at any confidence, causing 95.6% override rate. Physics injected block/smash over BST's predictions.
- **Fix:** Added `physics_min_conf_override: float = 0.25` (settings.py) — skip physics override when BST confidence is below this threshold. Tag as `bst_no_physics`.

### ✅ Physics Block Pivot Guard (Fixed — 2025-06-28)
- **Was:** `best_consistent_class` pivoted to "block" when BST's top-1 class was physically impossible. Block's physical conditions (`descend + slow + short`) are trivially satisfied by any decaying shuttle trajectory → 35/72 physics overrides forced to block.
- **Fix (Option A+C):** Skip block unless its softmax probability ≥ 50% of top-1 probability; require candidate probability > 2× unknown probability.

### ✅ Temperature Cache Cleanup (2025-06-28)
- **Was:** `ckpts/bst/bst_temperature.json` cached T=1.3415 from broken InpaintNet era. Loading it silently lowered confidence (mean conf 0.23 vs 0.33 at T=1.0).
- **Fix:** Deleted stale cache. Default T=1.0 restored.
- **Investigation confirmed:** 122 rule-based fallbacks (37%) are a genuine model limitation, not a data quality issue. Feature stats are identical between rule-based and model-processed clips (missing_bbox=0, missing_pose=0, shuttle_valid=96, jnb_std=0.23). The model outputs uniform logits for these clips regardless of temperature or prior correction.
- **Prior correction kept:** Cached `bst_logit_bias.json` prevents 28 model-processed clips from predicting unknown. Self-calibrated bias would be worse (65 vs 55 unknown).

### ✅ Spec-Aligned Rule-Based Classifier (Fixed — 2025-06-29)
- **Was:** `_rule_based_predict` used 3 features (max_speed, mean_dy, end_y) → 78/78 fallbacks predicted "net_shot"
- **Fix:** Created `backend/app/pipeline/shared/stroke_features.py` with spec-aligned 35+ feature extraction (court-space shuttle stats, JnB-derived joint angles, player zones, contact height, landing zones), two-level hierarchical classifier (family→specific), confidence estimation with evidence consistency check (capped at 0.85), structured evidence dict (contact_height, player_zone, outgoing_trajectory, landing_zone), and top-3 alternatives.
- **Impact:** 8 different rule-based stroke types (smash, defensive_lift, soft_lift_or_push, drive, net_shot, etc.) with explainable evidence. Old flat if-else replaced with `extract_clip_features` → `classify_family` → `classify_by_family` pipeline.
- **Files:** `backend/app/pipeline/shared/stroke_features.py` (NEW), `backend/app/models/bst.py` (_rule_based_predict rewritten)

### ✅ Balance Flip Iteration Bug (Fixed — 2025-06-29)
- **Was:** `attribution.py:183` used `for i in heuristic_idx` where `heuristic_idx` was a boolean mask Series when `debug_level < 1`. Iterating a boolean mask yields `True`/`False` values, not integer indices → `shots_df.at[True, "player_id"]` accesses a non-existent label → nothing gets flipped. The balance flip **never actually ran** in colab (debug_level=0).
- **Impact:** Player balance skewed 58.8%/41.2% instead of expected ~50/50. 22/32 rallies had >60% one player.
- **Fix:** `shots_df[r_mask].index` instead of bare `r_mask`. Always get `.index` for correct iteration.

### ✅ NaN Side Fill (Fixed — 2025-06-29)
- **Was:** `attribution.py:242` guarded by `if "side" not in shots_df.columns` — but Tier 1 (BST alpha/class_id) creates the column first, so the fillna never runs for Tiers 2-4. 108/250 shots (43%) had `side=NaN`.
- **Fix:** Unconditional fillna: `shots_df["side"] = shots_df["side"].fillna(shots_df["player_id"].map(_side_lookup).fillna("near"))`

### ✅ Internal Label Leaks (Fixed — 2025-06-29)
- **Was:** `classify_by_family` in `stroke_features.py` returned family-level names (`mid_height_unknown`, `overhead_unknown`, `underhand_unknown`, `net_unknown`) as stroke types when no specific match was found in that family. 4 `mid_height_unknown` shots leaked to final output.
- **Fix:** Post-routing remap: `mid_height_unknown` → `drive`, `overhead_unknown` → `clear`, `underhand_unknown` → `lift`, `net_unknown` → `net_shot`.

### ✅ Rule-Based Confidence Cap (Fixed — 2025-06-29)
- **Was:** `estimate_confidence` used max 0.95 cap with feature-margin boost only. 10 rule-based shots had 0.99 confidence despite contradictory evidence (e.g., "net_shot" with `landing_zone=deep (rear court)` and `player_zone=back court`).
- **Fix:** Added `_evidence_consistent()` check per stroke type (e.g., net_shot expects `contact=below waist` + `zone=front/mid court`). Evidence mismatch applies -0.20 penalty. Max cap lowered to 0.85. [User caveat: net kill shots can have deep landing — the consistency check uses zone, not landing zone, for net_shot]

### ✅ Shot Log Display in LabelingView UI (Added — 2025-06-29)
- **Was:** LabelingView showed only final predicted stroke type + confidence + source.
- **Fix:** Added per-shot detail showing BST output (class_id, pre-override stroke/conf), rule-based evidence (formatted key-value rows), physics override trail (bst_stroke → final). All data already in report.json via `shots_df.to_dict(orient="records")`.
- **File:** `frontend/src/views/LabelingView.tsx`

## 2026-07-02: Court + Physics Reliability Updates

### ✅ Hough-Line Court Detector + Manual Corners (Added — commit `ae1fd4a`)
- **What:** Added a line-based court fallback that detects court boundary lines, intersects them into a true trapezoid, and plugs into the existing court detector fallback chain without replacing the detector class.
- **Why:** Phone footage and non-broadcast views often produce rectangular/bad kpRCNN outputs. Hough-derived trapezoids and manual clicked corners preserve homography quality for zone/contact/physics cues.
- **Colab parity:** `colab/pipeline.py` now consumes manual court corners and shares the backend court-detection path instead of only auto-detecting.
- **Tests:** Added/updated court, shared-module, and Colab pipeline tests for Hough/manual-corner behavior.

### ✅ Invalid Court Geometry Degrades Gracefully (Fixed — commit `2d7a06e`)
- **Was:** Rectangular/degenerate court detections could be accepted as `valid=True`, silently crippling homography-based physics and analytics.
- **Fix:** Court geometry reliability is part of validation/homography validity, so degenerate courts trigger fallbacks instead of being accepted.
- **Backend behavior:** Player tracking, attribution, and court-position analytics now warn/degrade gracefully on invalid court geometry instead of aborting the pipeline when only non-homography cues are needed.
- **Orientation fix:** Trapezoid reliability accepts either narrower-at-top or narrower-at-bottom perspective by comparing `min(widths) / max(widths)`.

### ✅ Physics Uses Cleaned Shuttle Kinematics with Raw-Point Quality (Fixed — commit `7f564d9`)
- **Was:** Physics read sparse `shuttle_raw` for kinematics to avoid fake trajectories, then failed the density gate on phone footage.
- **Fix:** `extract_physics_features()` computes speed/direction/arc/depth from cleaned/interpolated `shuttle`, while `quality`, `real_points`, and usability are derived from raw detections. `quality = real_points / K` still down-weights sparse evidence.
- **Gate semantics:** `physics_min_valid=4` controls minimum real detections; `physics_quality_min=0.35` is no longer a hard skip once the minimum real-point gate passes.
- **Veto redesign:** Physics consistency now weights contact/zone/depth as strong cues and speed/descent/arc as weak cues. Weak monocular shuttle cues cannot veto alone.
- **Reporting:** `physics_summary` is written as an artifact, included in stroke-stage metadata, backend `report.json`, and Colab reports. Counts include `bst`, `bst_no_physics`, `physics_fallback`, `agree`, `physics_override`, `bst_gate_distrusted`, `usable`, `skipped`, `distrusted`, and `overrides`.
- **Verification:** Focused tests passed: `76 passed` across physics, context fusion, confusion pairs, and report-generator tests; `compileall` and `git diff --check` passed. Full backend suite was attempted but did not complete in-session, so do not claim a full-suite pass for this change.

## 2026-07-06: MMAction2 Ensemble Integration + Colab Setup Fixes

### ✅ MMAction2 Adapter (Added — commits `007555c`+)
- **What:** Added `backend/app/models/mmaction_adapter.py` — `MMActionClassifier` with `predict_from_clips()` matching BST interface
- **Modes:** `posec3d` (skeleton-based, reuses existing JnB pose keypoints), `slowfast` (RGB-based, requires video clips saved to disk), `pytorchvideo` (lighter alternative)
- **Ensemble strategy:** Probability-matrix level ensemble at Phase 2b in `strokes.py:481-515`: `probs = (1-w)*BST + w*MMAction` where `w=0.3` default
- **Lazy singleton getter:** `get_mmaction2()` in `backend/app/pipeline/shared/models.py:345` — returns None gracefully when not installed
- **Settings:** 6 new fields in `settings.py` (`mmaction2_enabled`, `mode`, `ensemble_weight`, `seq_len`, `num_classes`, `bst_n_classes`)
- **Version constraint:** Requires `mmcv>=2.0.0rc4, <2.2.0` — critical for install
- **Colab parity:** Form fields (`MMACTION2_ENABLED`, `MMACTION2_MODE`, `MMACTION2_WEIGHT`) in Cell 3; CLI args `--mmaction2`, `--mmaction2-mode`, `--mmaction2-weight` in `colab/pipeline.py`
- **All 443 tests pass** (7 skipped due to hardware)

### ✅ Colab Notebook Setup Overhaul (Fixed — multiple commits)
- **Problem:** `mim` crashes on Python 3.12+ (`pkgutil.ImpImporter` removed), `pip install mmcv` builds from source (10+ min), `numpy<2` conflicts with Colab's numpy 2.x
- **Fixes applied:**
  1. Remove `mim` — use `pip` directly (`mim` relies on old `setuptools/pkg_resources` incompatible with Python 3.12)
  2. Skip `pip install torch` — Colab has GPU torch pre-installed; redundant 223 MB download risks replacing with CPU version
  3. Remove `numpy<2` constraint — causes 10+ dependency conflicts on Colab
  4. Split install into 4 stages with progress prints (`[1/4]` through `[4/4]`)
  5. Replace `mmcv` (no pre-built cp312 wheels, builds from source) with `mmcv-lite` (universal `py2.py3-none-any` wheel, ~5s install)
  6. Pin `mmcv-lite>=2.0.0rc4,<2.2.0` — mmaction2 rejects v2.2.0 with version mismatch error (`MMCV==2.2.0 is used but incompatible`)
- **Result:** Cell 1 setup completes in ~2-3 min (down from 10+ min)

### ✅ Physics Override Sanity Guard Tuned (Fixed — commit `49c7aed`)
- **Was:** `physics_max_override_frac=0.70` — sanity guard never triggered (57% override rate in dry run)
- **Changed to:** `0.40` — triggers at 40% override rate; reverts all physics overrides to BST when exceeded
- **Rationale:** 57% override rate means physics dominates BST; guard forces physics to prove its value before overriding

### ✅ Colab Dry Run Results (2026-07-05, phone footage on T4)
- **201 shots**, **18 rallies**, 14 unique stroke types
- **Stroke source breakdown:**
  - `physics_override`: 115 (57.2%) — physics dominates
  - `bst_no_physics`: 58 (28.9%) — BST too low-confidence (<0.30), skipped
  - `physics_fallback`: 15 (7.5%) — both uncertain, rule-based fallback
  - `bst`: 13 (6.5%) — pure BST prediction
  - `agree`: 0 — physics and BST never agreed on a stroke
- **Attribution:** 100% Viterbi (6 sub-score OwnershipScorer), side split 102/99 (balanced)
- **Mean confidence:** 0.494; BST raw mean: 0.393
- **Shuttle detection rate:** 43% ⚠️ — affects physics quality
- **Pose coverage:** 100% ✅; **Court coverage:** 99.5% ✅
- **MMA2 status:** skipped — `mmcv-lite==2.2.0` failed mmaction2 version check (pinned to `<2.2.0` in fix)
- **Dominant strokes:** rush (30.4%), cross_court (12.7%), drive (11.8%), smash (8.8%)
- **Rare strokes:** net_shot (1.0%), drop (1.0%) — flagged as weaknesses
- **Rallies:** 13/18 end in forced_error, 2 in winner, 2 in unforced_error, 1 in net
- **Quality score:** 0.9 (high) with caveats: BST fallback 24%, low shuttle detection 43%

### Confirmed Model Limitations
- **31.2% rule-based fallback is intrinsic** — feature quality identical between RB and model clips (missing_bbox=0, shuttle_valid=93-95, jnb_std=0.22-0.23). Model outputs uniform logits for these clips regardless of temperature.
- **11/25 BST classes never activated** — model cannot predict these regardless of pipeline quality.
- **defensive_lift, soft_lift_or_push are 100% rule-based** — BST never outputs these classes.
- **Prior correction** (`bst_logit_bias.json`) is essential — prevents 28 model clips from predicting unknown.
- **aimplayer_alpha** mean=0.498 (barely above 0.5) — AimPlayer head barely distinguishes near/far players for attribution.

### Rule-Based Classifier Overview
```
stroke_features.py:
  extract_clip_features() → ~35 features (shuttle, joint angles, zones)
    ↓
  classify_family() → overhead/underhand/net/mid_height/serve
    ↓
  classify_by_family() → 15 specific types (smash, drop, clear, lift, 
                          defensive_lift, net_shot, drive, block, push, etc.)
    ↓
  estimate_confidence() → 0.10-0.85 with evidence consistency check
  _build_evidence() → structured dict (contact_height, player_zone, 
                      outgoing_trajectory, landing_zone)
  top3_alternatives() → (stroke, confidence) alternatives
```

## Recommended Actions (Priority)

### Critical (correctness)
1. ~~Fix BST seq_len wiring and weight path~~ (Done)
2. ~~Reorder stages for correct rally winners~~ (Done)
3. ~~Fix RTMPose x/y rescale transpose~~ (Done)
4. ~~Fix recovery-time pixel/meter mismatch~~ (Done)
5. ~~Scene-cut rally segmentation~~ (Done)
6. ~~Player attribution balance flip~~ (Done)
7. ~~Rule-based predictor spec-aligned rewrite~~ (Done)
8. ~~Physics gate: low-confidence BST skip~~ (Done)
9. ~~Physics block pivot guard + aggressive block guard~~ (Done)
10. ~~Balance flip iteration bug (boolean vs .index)~~ (Done — e9640e9)
11. ~~NaN side fill for Tiers 2-4~~ (Done — e9640e9)
12. ~~Internal label leak (mid_height_unknown)~~ (Done — e9640e9)
13. ~~Rule-based confidence cap + evidence consistency~~ (Done — e9640e9)

### High (reliability)
14. ~~Fix TrackNet integration (arch sync + InpaintNet)~~ (Done — 2025-06-29)
15. ~~Use BST Top/Bottom output for attribution~~ (Done — 2025-06-29)
16. ~~Compute analytics in meters via homography~~ (Done)
17. Replace per-frame YOLO with proper tracking
18. ~~Externalize config with pydantic-settings~~ (Done)
19. Add auth + upload validation
20. ~~Respect `court.valid` flag~~ (Done — graceful invalid-court degradation, `2d7a06e`)
21. ~~Reject degenerate court geometry before homography use~~ (Done — `2d7a06e`)
22. ~~Add Hough/manual-corner court fallback for phone footage~~ (Done — `ae1fd4a`)
23. ~~Use cleaned shuttle for physics kinematics while preserving raw-point quality~~ (Done — `7f564d9`)

### Medium (quality)
24. Re-run colab pipeline with pinned `mmcv-lite<2.2.0` and MMAction2 enabled to test ensemble
25. Re-run phone-video pipeline to measure physics override sanity guard (now 0.40) impact
26. Add shot_log formal table to report.json schema (data already in shots array)
27. Temperature recalibration: use `debug_bst_outputs.parquet` logits with fixed pipeline
28. ~~Add multi-signal ownership + Viterbi HMM~~ (Done — 8b8f701, 2025-06-29)
29. Re-run pipeline with new OwnershipScorer to measure attribution quality improvement
30. Fine-tune PoseC3D on ShuttleSet for meaningful ensemble signal (currently random weights)

### Phone-Video Pipeline
- ~~Temporal gap detection for scene cuts~~ (Done — `rallies.py`: NaN-streak check alongside spatial displacement)
- ~~Scene-cut propagation to shots + rally metadata~~ (Done — `scene_cut_before` column on rallies and shots)
- ~~UI attribution tier badges~~ (Done — green/orange/red/magenta dots in StrokeListPanel)
- ~~UI scene-cut warning~~ (Done — "SC" column with ⚠️ in rally breakdown)
- ~~UI shot log with BST/rule-based/physics trail~~ (Done — LabelingView.tsx)

### Nice-to-have
30. Unify backend/colab pipelines
31. ~~Replace single-frame technique score~~ (Done)
32. Cross-session progress tracking
33. Structured logging + data-quality score
34. Promote grounded LLM narration
35. License compliance audit
