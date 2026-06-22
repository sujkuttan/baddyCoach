# BaddyCoach - Agent Guidance

## Architecture Overview

**13-stage FastAPI pipeline** processing badminton videos into coaching reports:

```
court_detection → player_tracking → shuttle_tracking → pose_estimation → 
hit_frame_localization → stroke_classification → rally_segmentation → 
player_attribution → court_position_analytics → footwork_analytics → 
fitness_analytics → tactical_analytics → technical_analytics → coach_recommendations
```

**Key components:**
- Backend: FastAPI with WebSocket progress tracking
- Frontend: React + TypeScript + Vite dashboard
- ML models: YOLOv8, RTMPose, TrackNetV3, BST (Badminton Stroke Transformer)
- Coaching engine: YAML rule-based system (25+ rules in `backend/app/coach/rules.yaml`)

## Critical Architecture Notes

### ⚠️ Stage Ordering Bug
- **Issue:** `rally_segmentation` (index 7) runs before `player_attribution` (index 8)
- **Impact:** Rally winners systematically misattributed to player_1
- **Fix:** Reorder stages or recompute winners after attribution (`backend/app/api/routes.py:79-80`)

### ⚠️ BST Model Integration
- **Issue:** BST likely not running due to sequence-length mismatch
- **Details:** Clip uses hardcoded `SEQ_LEN=30` (`strokes.py:14`), but model expects `seq_len=100`
- **Impact:** 20-40% "unknown" predictions, falls back to rule-based classification (`bst.py:152-160`)
- **Fix:** Read `classifier.seq_len` in `strokes.py` instead of constant

### ⚠️ Model Path Inconsistency
- **Settings path:** `BST/weight/bst_CG_JnB_bone_merged.pt` (`settings.py:22`)
- **Downloader path:** `ckpts/bst/bst_CG_AP.pt` (`model_downloader.py:27`)
- **Impact:** Configured path doesn't exist after download, silent fallback

### ⚠️ TrackNetV3 Architecture
- **Issue:** Custom UNet with `in_channels=27, num_classes=8` (not published TrackNetV3)
- **Impact:** Official `TrackNet_best.pt` won't load, crashes shuttle stage
- **Issue:** Throws away 7 of 8 output channels, no InpaintNet implemented
- **Impact:** Zero shuttle detections poison hit/stroke/analytics downstream

### ⚠️ RTMPose Bug
- **Issue:** x/y rescale divisors swapped (`rtmpose.py:62-63`)
- **Impact:** All joints mislocated, corrupts BST features and analytics
- **Issue:** Hardcoded input name `"input"` vs `self.input_name`

### ⚠️ Recovery Time Units
- **Issue:** `threshold = 0.3` in pixels, but recovery distances in meters
- **Impact:** `avg_recovery` ≈ 0 for everyone, recovery loop scans all players

## Testing & Development

### Hardware-Aware Testing
- Auto-skip based on RAM/GPU/model availability (`backend/tests/conftest.py`)
- **Markers:** `gpu`, `cpu_only`, `model`, `memory_intensive`, `slow`, `integration`
- **Minimum requirements:** 4GB RAM, CUDA GPU for GPU tests, model checkpoints

### Test Structure
- 39 test files covering all major components
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
- **Settings:** `BaseModel` not `BaseSettings` (`settings.py:5`) - no env/.env override
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
- RTMPose transpose bug corrupts all pose-derived analytics

### ⚠️ Data Quality Issues
- Synthetic detections (`_generate_synthetic_detections`) presented as real
- Rule-based strokes masquerading as model output
- No data quality flags in coaching reports

### ⚠️ Architecture Conflicts
- Two coaching engines (`coach/engine.py` vs `shuttle_coach/`) with overlapping purpose
- Colab pipeline (3,400 lines) duplicates backend logic - high drift risk
- No model abstraction layer - each stage hardcodes model imports

### ⚠️ Security & Reliability
- No authentication on any endpoint
- No file size/length/MIME validation on upload
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
- **Data quality:** Look for synthetic detections in report metadata
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
- `backend/app/coach/engine.py` - Coaching logic
- `backend/app/pipeline/base.py` - Pipeline base class

### Model Files
- `backend/app/models/yolov8.py` - Person detection/tracking
- `backend/app/models/rtmpose.py` - Pose estimation
- `backend/app/models/tracknet.py` - Shuttle tracking
- `backend/app/models/bst.py` - Stroke classification

### Configuration
- `backend/app/config/settings.py` - Model paths, thresholds
- `backend/app/config/gpu_batch.py` - GPU batch sizing
- `backend/app/coach/rules.yaml` - 25+ coaching rules

## Migration Notes

### ⚠️ Colab Pipeline Parity
- Colab has 3,400-line reimplementation of backend
- BST adapts to detected sequence length (`colab/pipeline.py:1348-1350`)
- Coach rules duplicated (backend reads YAML, colab hardcodes)

### ⚠️ Shuttle Coach Endpoint
- **Broken:** Requires `player_detections.parquet` but backend stores `players.json`
- **Error:** `Missing required tables: ['player_detections']`
- **Solution:** Fix data format or remove endpoint

## Recommended Actions (Priority)

### Critical (correctness)
1. Fix BST seq_len wiring and weight path
2. Reorder stages for correct rally winners
3. Fix RTMPose x/y rescale transpose
4. Fix recovery-time pixel/meter mismatch
5. Respect `court.valid` flag
6. Flag synthetic/fallback data in reports

### High (reliability)
7. Fix TrackNet integration (official arch + InpaintNet)
8. Use BST Top/Bottom output for attribution
9. Compute analytics in meters via homography
10. Replace per-frame YOLO with proper tracking
11. Externalize config with pydantic-settings
12. Add auth + upload validation

### Nice-to-have
13. Unify backend/colab pipelines
14. Replace single-frame technique score
15. Cross-session progress tracking
16. Structured logging + data-quality score
17. Promote grounded LLM narration
18. License compliance audit
