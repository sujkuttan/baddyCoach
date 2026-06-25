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
- 220 tests (42 spec + 178 core), 9 skipped (hardware-dependent), 0 failed
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

### ✅ Temporal Smoothing Scope (Fixed — 2025-06-25)
- **Was:** `if stype != "unknown": continue` — only unknown strokes smoothed; low-confidence "drive" (conf=0.089) never corrected
- **Fix:** Now smooths any stroke with confidence < 0.2, not just "unknown"

### ✅ Rally Winner Threshold (Fixed — 2025-06-25)
- **Was:** `_infer_end_reason` required conf ≥ 0.5 for "winner"; max BST conf 0.633 → 13/14 rallies ended in "unforced_error"
- **Fix:** Lowered winner threshold to 0.3; added speed-based winner detection (smash > 8 m/s = winner); `_compute_rally_winner_after_attribution` passes shuttle speed to `_infer_end_reason`

## 2025-06-25: Stroke Classification Root Cause Analysis

### ⚠️ Rule-Based Classifier: Court-Space vs Pixel-Space Normalization (Unfixed)
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

### ⚠️ Temporal Smoothing Skips Non-Unknown Strokes (Unfixed)
- **Issue:** Line 273: `if stype != "unknown": continue` — smoothing only corrects unknown strokes. Low-confidence "drive" (1 BST drive at conf=0.089) and other determinate predictions remain untouched even when surrounded by opposite stroke types.
- **Fix:** Extend smoothing to correct low-confidence predictions below an entropy/confidence threshold (e.g., `confidence < 0.2`), not just "unknown".

### ⚠️ Rally Winner Logic Fragile (Unfixed)
- **Issue:** `_infer_end_reason` requires confidence ≥ 0.5 for "winner" → no shot in Run 2 qualifies (max conf 0.633 for BST, 0.3 for rule-based). 13/14 rallies end in "unforced_error". Winner = "player who didn't hit last shot" — accidentally correct for errors but wrong for genuine winners/net shots.
- **Sequence:** `_infer_end_reason(stroke_type, confidence)` at `rallies.py:28` → `end_reason = "unforced_error"` for most low-confidence shots → `winner = other_player` at line 35.
- **Fix:** Lower winner confidence threshold to 0.3, or add trajectory-speed-based winner detection (smash/kill near net = likely winner).

### Key Verification Data (Run 2, `results/mmpose_results/debug/`)
- **Shots:** 204 total (135 BST, 69 rule-based). BST: 97 clear, 22 smash, 14 lift, 1 drive, 1 short_serve. Rule-based: 69 drive (all).
- **BST classes used:** 3 (22), 4 (6), 5 (53), 16 (8), 17 (44), 18 (1), 23 (1). Classes 0-2, 6-15, 19-22, 24 NEVER activated.
- **Shuttle range:** x ∈ [-2.35, 32.72], y ∈ [-16.21, 3.99] — far outside court dimensions (13.4×6.1m), suggesting InpaintNet rectification produces extreme values.
- **Player tracking:** 166 unique YOLO track_ids for 18,000 detections across 2 players — no temporal linking, causing feature gaps in clips.

## Recommended Actions (Priority)

### Critical (correctness)
1. ~~Fix BST seq_len wiring and weight path~~ (Done — P5/P6)
2. ~~Reorder stages for correct rally winners~~ (Done)
3. ~~Fix RTMPose x/y rescale transpose~~ (Done)
4. ~~Fix recovery-time pixel/meter mismatch~~ (Done — homography-based court-space comparison)
5. Respect `court.valid` flag
6. ~~Flag synthetic/fallback data in reports~~ (Done — fallback removed entirely)

### High (reliability)
7. Fix TrackNet integration (official arch + InpaintNet)
8. Use BST Top/Bottom output for attribution
9. ~~Compute analytics in meters via homography~~ (Done — footwork distances, recovery times, jump filter all use homography)
10. Replace per-frame YOLO with proper tracking
11. ~~Externalize config with pydantic-settings~~ (Done)
12. Add auth + upload validation

### Nice-to-have
13. Unify backend/colab pipelines
14. ~~Replace single-frame technique score~~ (Done — 5 temporal features + 8 YAML coaching rules)
15. Cross-session progress tracking
16. Structured logging + data-quality score
17. Promote grounded LLM narration
18. License compliance audit
