# BaddyCoach - Agent Guidance

## Architecture Overview

**13-stage FastAPI pipeline** processing badminton videos into coaching reports:

```
court_detection → player_tracking → shuttle_tracking → pose_estimation → 
hit_frame_localization → stroke_classification → player_attribution → 
rally_segmentation → court_position_analytics → footwork_analytics → 
fitness_analytics → tactical_analytics → technical_analytics → coach_recommendations
```

**Key components:**
- Backend: FastAPI with WebSocket progress tracking
- Frontend: React + TypeScript + Vite dashboard
- ML models: YOLOv8, RTMPose, TrackNetV3, BST (Badminton Stroke Transformer)
- Coaching engine: YAML rule-based system (33+ rules in `backend/app/shuttle_coach/feedback/rules.yaml`)

## Critical Architecture Notes

### ✅ Stage Ordering Bug (Fixed)
- **Was:** `rally_segmentation` ran before `player_attribution`, misattributing winners to player_1
- **Fix:** Reordered so `player_attribution` (index 6) runs before `rally_segmentation` (index 7), with rally alternation algorithm for winner determination

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

### ⚠️ TrackNetV3 Architecture
- **Issue:** Custom UNet with `in_channels=27, num_classes=8` (not published TrackNetV3)
- **Impact:** Official `TrackNet_best.pt` won't load, crashes shuttle stage
- **Issue:** Throws away 7 of 8 output channels, no InpaintNet implemented
- **Impact:** Zero shuttle detections poison hit/stroke/analytics downstream

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

### ✅ BST Bottom_ Prefix Leak (Fixed)
- **Was:** `map_to_coach_class` returned `Bottom_smash` for the near player but `smash` for the far player — every exact-string consumer (rally end-reason, technique bounds, tactical distribution, frontend charts) broke for near-player strokes
- **Fix:** `map_to_coach_class` now returns bare stroke type for both players; side is preserved in `shuttleset_class_id` and available via `get_shuttleset_class_info`

### ✅ Upload Zero-Byte Bug (Fixed)
- **Was:** `routes.py` called `await file.read()` at line 268 (validation) and again at line 294 (write) — second read on consumed `UploadFile` returned `b""`, writing empty videos
- **Fix:** Reuse the `content` buffer from the validation read; dropped the redundant second read

### ⚠️ GPU OOM on T4 (YOLO Conv2d Fragmentation)
- **Symptom:** CUDA OOM at batch 7/18 despite 3.78 GiB reserved but unallocated
- **Root cause:** `gpu_batch.py` ≥12GB tier had `yolo_chunk=1000, yolo_batch=64`. Each YOLO batch allocated ~750 MiB Conv2d tensors that fragment the allocator; by batch 7, no contiguous 750 MiB block available.
- **Tier values were never committed from initial fix** — AGENTS.md described the reduction but `gpu_batch.py` still had aggressive values
- **Fix (commit `080eb9b`):** (1) `gpu_batch.py` tiers actually reduced — ≥12GB: `yolo_chunk=200, yolo_batch=16, tracknet_chunk=16, rtmpose_chunk=128` (was 1000/64/128/256). (2) `colab/pipeline.py` `BATCH_SIZE`: 500→300. (3) `torch.cuda.empty_cache()` added between batches.

## Testing & Development

### Hardware-Aware Testing
- Auto-skip based on RAM/GPU/model availability (`backend/tests/conftest.py`)
- **Markers:** `gpu`, `cpu_only`, `model`, `memory_intensive`, `slow`, `integration`
- **Minimum requirements:** 4GB RAM, CUDA GPU for GPU tests, model checkpoints

### Test Structure
- 313 tests (core), 7 skipped (hardware-dependent), 0 failed
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
- `backend/app/models/bst.py` - Stroke classification

### Configuration
- `backend/app/config/settings.py` - Model paths, thresholds
- `backend/app/config/gpu_batch.py` - GPU batch sizing
- `backend/app/shuttle_coach/feedback/rules.yaml` - 33+ coaching rules

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

### ✅ Latest Colab Run Results (test_match.mp4, 300s, 2025-06-28)
- **320 shots**, **11 rallies** (~29 shots/rally), 45%/55% player split
- **12 unique stroke types** per player (was 7) — drop, push, rush, cross_court, short_serve now visible
- **13/25 BST classes active** (was 7/25) — Top_smash(56), Bottom_block(79), Bottom_lift(26), Top_clear(9), Top_push(8), Top_rush(9), Top_drop(3), etc.
- **0 rule-based shots** in final output (was 78+)
- **73.1% BST coverage** (234/320 = bst_no_physics), **22.5% physics_override**, **3.4% physics_fallback**
- **Mean BST model conf: 0.33** (up from ~0.22)
- Rule-based fallback rate unchanged at 36.7% (model limitation)

## 2025-06-25: Stroke Classification Root Cause Analysis

### ⚠️ Rule-Based Classifier: Court-Space vs Pixel-Space Normalization (Fixed)
- **Issue:** All 69 rule-based shots (33.8%) predict "drive" regardless of actual trajectory
- **Root cause:** Clip construction (`strokes.py:82-83`) normalizes shuttle by court dimensions (`x/13.4`, `y/6.1`), but rule-based thresholds (`bst.py:363-376`) were designed for pixel-space normalization (`x/1920`, `y/1080` → range [0,1])
- **Impact:** `end_y` ALWAYS negative after court-normalization → "lift" (needs `end_y > 0.5`) and "drop" (needs `end_y > 0.7`) can NEVER trigger; `mean_speed > 0.03` always → "net_shot" can NEVER trigger; most trajectories fall through to "drive" or "unknown".
- **Secondary issue:** Between-2-hits clips span ~3.3s (100 frames at 30fps), covering BOTH incoming shuttle (toward player) and outgoing shuttle (away from player). The trajectory direction reverses at hit point → V-shaped average → "drive"-like signal.
- **Validation:** Reproduced with actual shuttle.parquet data (69 shots → 18 drive/21 clear/16 smash/14 unknown with 11-frame window, but 100-frame clip collapses to always-drive).
- **Fix:** (1) Add `vid_w, vid_h` to clip dict; denormalize shuttle by court dims, renormalize by video dims before rule-based predict. (2) Extract only POST-HIT frames from clip for rule-based analysis.

### ⚠️ BST Predicts Only 7 of 25 Classes (Unfixed)
- **Issue:** Model outputs only class IDs 3, 4, 5 (Top) and 16, 17, 18, 23 (Bottom) → smash, lift, clear, short_serve. NEVER net_shot, block, drop, push, rush, cross_court.
- **Confirmed across two matches:** Run 1 (same 7 classes) and Run 2 (same 7 classes). Both matches known to contain net shots and drops.
- **Mean confidence:** 0.26, max 0.633 — very low entropy distribution across all clips.
- **Root cause (hypothesized):** (a) Per-frame YOLO tracking with no temporal ID linking → `det_bbox_lookup` fails when player track ID changes mid-clip (166 unique track_ids for 18,000 detections across 2 players). Missing detections → zeros in joints/bbox → garbled features. (b) Class ordering in `SHUTTLESET_CLASSES` may not match training checkpoint order — checkpoint filename `CG_JnB_bone_merged` uses ShuttleSet ordering, but code defines extra classes (block, short_serve, long_serve) at different positions.
- **Fix:** (a) Add temporal detection smoothing: interpolate bbox across frames when track ID switches. (b) Verify class ordering by running inference on a labeled ShuttleSet sample.

### ⚠️ Temporal Smoothing Skips Non-Unknown Strokes (Revised 2025-06-25)
- **Issue:** Line 273: `if stype != "unknown": continue` — smoothing only corrects unknown strokes. Low-confidence "drive" (1 BST drive at conf=0.089) and other determinate predictions remain untouched even when surrounded by opposite stroke types.
- **Fix (initial):** Smooth any stroke with confidence < 0.2, not just "unknown"
- **Fix (revised):** Reverted to unknown-only smoothing. The expanded scope caused rule-based "net_shot" bias (78 shots, conf~0.22) to overwrite 13 determinate BST predictions (lift, smash, short_serve, etc.) to net_shot via majority vote. Determinate predictions, even low-confidence, are preserved to avoid rule-based neighborhood dominance.

### ✅ Rally Winner Logic Fragile (Fixed — 2025-06-25)
- **Was:** `_infer_end_reason` required confidence ≥ 0.5 for "winner" → no shot in Run 2 qualifies (max conf 0.633 for BST, 0.3 for rule-based). 13/14 rallies end in "unforced_error". Winner = "player who didn't hit last shot" — accidentally correct for errors but wrong for genuine winners/net shots.
- **Fix:** Lower winner confidence threshold to 0.3, or add trajectory-speed-based winner detection (smash/kill near net = likely winner).

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

## Current Status (2025-06-28)

### Pipeline Performance (test_match.mp4, 300s)
- **313 shots**, **22 rallies** (was 11 after scene-cut fix), 47/53 player split
- **12 unique stroke types**, 13/25 BST classes active
- **~20% physics override** (after aggressive block guard), 57.5% bst_no_physics, 3% fallback
- **0% rule-based** in final output (125 rule-based fallbacks → all "unknown")
- **Mean conf: 0.33** (model clips, T=1.0 vs 0.23 at T=1.3415)

### Key Findings from Latest Colab Run
- **T=1.0 increased override rate:** Mean conf 0.33 (vs 0.23 at T=1.3415) pushed 36% more clips above `physics_min_conf_override=0.25` → 117 overrides (37.4%), up from 72 (22.5%). The temperature change, not the code, caused the surge.
- **Block no-op overrides:** 55/117 overrides were BST→block → physics→block (no-op). These just changed the source tag without altering the stroke type.
- **Aggressive block guard (commit `70927c1`):** If BST predicted "block" and physics would keep it as "block", the override is skipped entirely. Physics can still override TO block from a non-block BST prediction (legitimate correction), but no-op block-overrides are eliminated.
- **Scene-cut rally segmentation** produced 22 rallies (up from 11) with reasonable structure (max gaps 26-84 frames, mean 14 shots/rally). No over-splitting detected.
- **37% rule-based fallback is confirmed intrinsic** to the BST model, not a pipeline bug. Feature quality identical between rule-based and model-processed clips.

### Confirmed Model Limitations
- **37% rule-based fallback is intrinsic** — feature quality identical between RB and model clips (missing_bbox=0, shuttle_valid=93-95, jnb_std=0.22-0.23). Model outputs uniform logits for these clips regardless of temperature.
- **14/25 classes active** (model predicts 14 of 25 ShuttleSet classes). Classes 1, 4, 10-12, 15, 19-22, 24 never activated.
- **Prior correction** (`bst_logit_bias.json`) is essential — prevents 28 model clips from predicting unknown. Self-calibrated bias would be worse.

## Recommended Actions (Priority)

### Critical (correctness)
1. ~~Fix BST seq_len wiring and weight path~~ (Done)
2. ~~Reorder stages for correct rally winners~~ (Done)
3. ~~Fix RTMPose x/y rescale transpose~~ (Done)
4. ~~Fix recovery-time pixel/meter mismatch~~ (Done)
5. ~~Scene-cut rally segmentation~~ (Done)
6. ~~Player attribution balance flip~~ (Done)
7. ~~Rule-based predictor max_speed rewrite~~ (Done)
8. ~~Physics gate: low-confidence BST skip~~ (Done)
9. ~~Physics block pivot guard (Option A+C)~~ (Done)
10. ~~Aggressive block guard (no-op override prevention)~~ (Done)

### High (reliability)
10. Fix TrackNet integration (official arch + InpaintNet)
11. Use BST Top/Bottom output for attribution
12. ~~Compute analytics in meters via homography~~ (Done)
13. Replace per-frame YOLO with proper tracking
14. ~~Externalize config with pydantic-settings~~ (Done)
15. Add auth + upload validation
16. Respect `court.valid` flag

### Nice-to-have
17. Unify backend/colab pipelines
18. ~~Replace single-frame technique score~~ (Done)
19. Cross-session progress tracking
20. Structured logging + data-quality score
21. Promote grounded LLM narration
22. License compliance audit
