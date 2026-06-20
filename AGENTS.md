# AGENTS.md — BMCA Development Guide

## Project Overview

**BMCA — Badminton Post-Match Coaching Assistant** converts match video into coach-grade insights via a 14-stage ML pipeline.

**Stack:** Python 3.14 / FastAPI / PyTorch / React + TypeScript / Vite / Tailwind CSS

## Commands

### Backend
```bash
# Run server (always set PYTHONPATH)
PYTHONPATH=/home/sujith/baddyCoach/backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/ -v

# Run single test
.venv/bin/pytest backend/tests/test_strokes.py -v
```

### Frontend
```bash
cd /home/sujith/baddyCoach/frontend
npx tsc --noEmit          # Type check
npx vite build            # Build
npx vite                  # Dev server (port 5173, proxies /api to :8000)
```

### Colab Pipeline
```bash
python colab/pipeline.py video.mp4 --output report.json --device cuda
```

## Architecture

### Pipeline Stages (sequential)
1. Court Detection → 2. Player Tracking (YOLOv8) → 3. Shuttle Tracking (TrackNetV3) → 4. Pose Estimation (RTMPose) → 5. Hit Frame Localization → 6. Stroke Classification (BST) → 7. Rally Segmentation → 8. Player Attribution → 9-13. Analytics (Court/Footwork/Fitness/Tactical/Technical) → 14. Coach Recommendations

**Note:** Rally segmentation (7) runs BEFORE player attribution (8) because attribution uses rally alternation.

### Data Flow
- All tabular data stored as **Parquet** (shuttle, pose, shots, rallies)
- JSON for structured data (court, players, analytics, report)
- `ArtifactStore` manages all read/write per job directory
- Reports saved to `data/jobs/{id}/report.json`

### Key Files
| File | Purpose |
|------|---------|
| `backend/app/api/routes.py` | API endpoints + pipeline runner |
| `backend/app/api/websocket.py` | WebSocket broadcast manager |
| `backend/app/pipeline/*.py` | Pipeline stages (court, players, shuttle, pose, hits, strokes, attribution, rallies) |
| `backend/app/pipeline/analytics/*.py` | Analytics sub-stages (court_position, footwork, fitness, tactical, technical) |
| `backend/app/models/*.py` | ML model wrappers |
| `backend/app/models/bst_model.py` | BST-CG architecture (TCN + Transformer + Cross Attention + Clean Gate) |
| `backend/app/models/bst_preprocessing.py` | BST normalization, bone creation, sequence extraction |
| `backend/app/models/bst.py` | BSTClassifier with model loading and rule-based fallback |
| `backend/app/coach/engine.py` | Rule-based coach engine |
| `backend/app/config/settings.py` | Config + model paths |
| `frontend/src/views/ReportView.tsx` | Main report dashboard |
| `frontend/src/views/UploadView.tsx` | Video upload UI |
| `frontend/src/views/ProcessingView.tsx` | Pipeline progress UI |
| `frontend/src/components/VideoPlayer.tsx` | Video.js player with rally timeline + stroke timeline |
| `frontend/src/components/StrokeTimeline.tsx` | Color-coded stroke markers on timeline bar |
| `frontend/src/components/StrokeListPanel.tsx` | Filterable stroke list grouped by rally |
| `frontend/src/components/StageProgress.tsx` | Pipeline stage progress display |
| `frontend/src/hooks/useWebSocket.ts` | WebSocket hook for job progress events |
| `colab/pipeline.py` | Self-contained GPU pipeline (2800+ lines, all models inlined) |
| `colab/BMCA_Colab.ipynb` | Colab notebook for running pipeline |

## Conventions

### Python
- `StageResult.from_error(msg)` — NOT `StageResult.error()` (naming conflict)
- `gpu_enabled: bool = False` default — use `settings.device` property for auto-detect
- Parquet keypoint data: always reshape with `np.array(kps.tolist())` — pyarrow flattens nested lists
- Pipeline stages accept optional kwargs beyond `(artifacts, config)` for testing

### Player Attribution
- **Rally alternation is the primary method** — Players MUST alternate hits within a rally. Determine first hitter via shuttle direction, then alternate P1/P2. Fallback: shuttle trajectory direction.
- **Side assignment uses relative bbox center_y** — The player with the larger center_y (lower in frame) is "near", smaller is "far". NEVER use `bbox[1] > court_mid_y` (broken for broadcast angles where both bboxes are above the midline).
- **No track ID filtering** — Don't use `top2` track ID filtering. Group all detections by side directly. YOLOv8 creates 200+ unique track IDs on broadcast footage; top 2 cover only ~13% of frames.
- Pipeline order: rally segmentation BEFORE attribution (attribution depends on rally structure).

### BST Integration
- BST_CG inputs: `JnB` (batch, seq_len, 2 players, 72), `shuttle` (batch, seq_len, 2), `pos` (batch, seq_len, 2, 2), `video_len` (batch,)
- Backend clipping: Extract from previous opponent's hit to next opponent's hit
- Colab clipping: Center-aligned windows (`hit_frame ± seq_len // 2`) — simpler but less accurate
- Colab preprocessing includes: shuttle zero-interpolation (linear + bfill/ffill), center_align joint normalization, closest-in-time RTMPose fallback (±10 frames)
- Class mapping: 25 ShuttleSet → 12 coaching classes (net_shot, block, smash, lift, clear, drive, drop, push, rush, cross_court, short_serve, long_serve)
- Joints normalized by bbox diagonal with center_align, shuttle by video resolution [0,1]
- Bones: 19 COCO pairs from 17 keypoints
- **Known issue:** ~20-40% "unknown" predictions remain due to colab clipping/normalization differences from BST training data

### Frontend
- No router — state machine in App.tsx (`upload | processing | report`)
- Views: `UploadView`, `ProcessingView`, `ReportView`
- Components: `CourtHeatmap`, `FatigueTrendChart`, `FitnessStats`, `CoachPanel`, `ShotChart`, `VideoPlayer`, `StrokeTimeline`, `StrokeListPanel`, `StageProgress`
- `VideoPlayer` uses `forwardRef` + `useImperativeHandle` exposing `seekTo(time)` — `StrokeListPanel` click-to-seek wired via ref
- Dark theme: `court-dark` (#0f1419), `shuttle-lime` (#c8ff00), `feather-green` (#00e676)
- Fonts: Bebas Neue (display), DM Sans (body), JetBrains Mono (mono)
- All charts use Recharts. Video uses Video.js. WebSocket for progress events.
- `ReportView` accepts either `jobId` (fetches from API) or `reportData` (direct JSON for Colab-imported reports)

### Git
- Commit messages: `feat:`, `fix:`, `docs:` prefixes
- Never commit `ckpts/`, `data/`, `.venv/`, `node_modules/`, `dist/`, `BST/`, `RTMPose/`, `results/`, `videos/`, `*.pt`

### Colab GPU Constraints
- **YOLOv8**: Must chunk frames to 200 per call + `stream=True`. Passing all 2000 frames causes 14.65GiB allocation on T4 (14.56GiB) → CUDA OOM.
- **RTMPose ONNX**: Must chunk crops to 64 per ONNX call. 3997 crops in one call → "Failed to allocate 1019381760 bytes".
- **TrackNet**: Pre-process frames once into `(N, 288, 512, 3)` array, build sliding windows via numpy slicing. Eliminates 9x redundant cv2.resize.
- **Pipeline stage order**: Rally segmentation runs BEFORE player attribution (attribution depends on rally alternation).
- **Colab download**: Filter out `Zone.Identifier` files from zip. `files.download()` can fail silently — add try/except with individual file fallback.

## Models

| Model | Path | Purpose |
|-------|------|---------|
| TrackNetV3 | `ckpts/TrackNet_best.pt` | Shuttle tracking |
| YOLOv8s | auto-download | Player detection |
| RTMPose | `ckpts/rtmpose/*.onnx` | Pose estimation |
| BST-CG (best) | `BST/weight/bst_CG_JnB_bone_merged.pt` | Stroke classification (ShuttleSet merged 25 classes) |

**Known limitations:**
- YOLOv8n/s fails on wide-angle broadcast footage (players too small). Synthetic fallback used. Both players ARE detected on 99% of frames with broadcast footage — the issue was side assignment, not detection.
- BST weights are raw state_dict (no bundled model class). Falls back to rule-based classification using shuttle trajectory when model unavailable or produces class 0.
- TrackNetV3 input: 27 channels (9 frames × 3 RGB). Sigmoid postprocessing required.
- RTMPose uses `onnxruntime-gpu` for CUDA acceleration; falls back to CPU gracefully.

## Testing

70 tests in `backend/tests/`. Run full suite before commits:
```bash
PYTHONPATH=/home/sujith/baddyCoach/backend .venv/bin/pytest backend/tests/ -v
```

## Dependencies

### Backend (pip)
`torch`, `ultralytics`, `onnxruntime-gpu`, `opencv-python-headless`, `scipy`, `numpy`, `pandas`, `pyarrow`, `fastapi`, `uvicorn`, `websockets`, `pyyaml`, `gdown` (optional, for model downloads)

### Frontend (npm)
`react`, `react-dom`, `recharts`, `video.js`, `vite`, `tailwindcss`, `typescript`
