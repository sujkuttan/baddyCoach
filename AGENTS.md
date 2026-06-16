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
1. Court Detection → 2. Player Tracking (YOLOv8) → 3. Shuttle Tracking (TrackNetV3) → 4. Pose Estimation (RTMPose) → 5. Hit Frame Localization → 6. Stroke Classification (BST) → 7. Player Attribution → 8. Rally Segmentation → 9-13. Analytics (Court/Footwork/Fitness/Tactical/Technical) → 14. Coach Recommendations

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
| `backend/app/pipeline/*.py` | Pipeline stages |
| `backend/app/pipeline/analytics/*.py` | Analytics sub-stages |
| `backend/app/models/*.py` | ML model wrappers |
| `backend/app/models/bst_model.py` | BST-CG architecture (TCN + Transformer + Cross Attention + Clean Gate) |
| `backend/app/models/bst_preprocessing.py` | BST normalization, bone creation, sequence extraction |
| `backend/app/models/bst.py` | BSTClassifier with model loading and rule-based fallback |
| `backend/app/coach/engine.py` | Rule-based coach engine |
| `backend/app/config/settings.py` | Config + model paths |
| `frontend/src/views/ReportView.tsx` | Main report dashboard |
| `colab/pipeline.py` | Self-contained GPU pipeline |

## Conventions

### Python
- `StageResult.from_error(msg)` — NOT `StageResult.error()` (naming conflict)
- `gpu_enabled: bool = False` default — use `settings.device` property for auto-detect
- Parquet keypoint data: always reshape with `np.array(kps.tolist())` — pyarrow flattens nested lists
- Pipeline stages accept optional kwargs beyond `(artifacts, config)` for testing

### BST Integration
- BST_CG inputs: `JnB` (batch, seq_len, 2 players, 72), `shuttle` (batch, seq_len, 2), `video_len` (batch,)
- Clipping: Extract from previous opponent's hit to next opponent's hit (not fixed-width windows)
- Class mapping: 25 ShuttleSet → 12 coaching classes (net_shot, block, smash, lift, clear, drive, drop, push, rush, cross_court, short_serve, long_serve)
- Joints normalized by bbox diagonal, shuttle by video resolution [0,1]
- Bones: 19 COCO pairs from 17 keypoints

### Frontend
- No router — state machine in App.tsx (`upload | processing | report`)
- Components: `CourtHeatmap`, `FatigueTrendChart`, `FitnessStats`, `CoachPanel`, `ShotChart`, `VideoPlayer`
- Dark theme: `court-dark` (#0f1419), `shuttle-lime` (#c8ff00), `feather-green` (#00e676)
- Fonts: Bebas Neue (display), DM Sans (body), JetBrains Mono (mono)
- All charts use Recharts. Video uses Video.js.
- `ReportView` accepts either `jobId` (fetches from API) or `reportData` (direct JSON)

### Git
- Commit messages: `feat:`, `fix:`, `docs:` prefixes
- Never commit `ckpts/`, `data/`, `.venv/`, `node_modules/`, `dist/`

## Models

| Model | Path | Purpose |
|-------|------|---------|
| TrackNetV3 | `ckpts/TrackNet_best.pt` | Shuttle tracking |
| YOLOv8s | auto-download | Player detection |
| RTMPose | `ckpts/rtmpose/*.onnx` | Pose estimation |
| BST-CG-AP | `ckpts/bst/bst_CG_AP.pt` | Stroke classification (old path) |
| BST-CG (best) | `BST/weight/bst_CG_JnB_bone_merged.pt` | Stroke classification (ShuttleSet merged 25 classes) |

**Known limitations:**
- YOLOv8n/s fails on wide-angle broadcast footage (players too small). Synthetic fallback used.
- BST weights are raw state_dict (no bundled model class). Falls back to random classification.
- TrackNetV3 input: 27 channels (9 frames × 3 RGB). Sigmoid postprocessing required.

## Testing

68 tests in `backend/tests/`. Run full suite before commits:
```bash
.venv/bin/pytest backend/tests/ -v
```

## Dependencies

### Backend (pip)
`torch`, `ultralytics`, `onnxruntime`, `opencv-python-headless`, `scipy`, `numpy`, `pandas`, `pyarrow`, `fastapi`, `uvicorn`, `websockets`, `pyyaml`, `gdown` (optional, for model downloads)

### Frontend (npm)
`react`, `react-dom`, `recharts`, `video.js`, `vite`, `tailwindcss`, `typescript`
