# Badminton Post-Match Coaching Assistant (BMCA) — Technical Design Spec

**Version:** 1.0
**Date:** 2026-06-14
**Status:** Design Approved — Ready for Implementation Planning

---

## 1. Overview

BMCA is a local-first badminton analytics platform that converts raw match video into coach-grade insights. A user uploads a singles match video, the system processes it through a sequential ML pipeline, and produces a comprehensive coaching report with technical, tactical, fitness, and footwork analytics.

**Target users:** Competitive players, academy coaches, parents of junior players, performance analysts.

**Success metrics:**
- Stroke classification: >90% top-10 classes, >85% overall
- Player tracking: >98% correct attribution
- Rally detection: >95%
- GPU processing: <15 min for 1 hr video (RTX 4060+)
- CPU processing: <60 min for 1 hr video
- Memory: <16GB

---

## 2. System Architecture

### 2.1 High-Level Structure

```
┌─────────────────────────────────────────────┐
│  Frontend (React + TypeScript)               │
│  - Upload form                              │
│  - Processing progress (WebSocket)          │
│  - Video player + timeline                  │
│  - Report dashboard                         │
└──────────────────┬──────────────────────────┘
                   │ HTTP + WebSocket
┌──────────────────▼──────────────────────────┐
│  Backend (FastAPI + Python)                  │
│  - Upload handler                           │
│  - Pipeline orchestrator                    │
│  - Analytics engine                         │
│  - Coach recommendation engine              │
│  - Report generator                         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Processing Pipeline (sequential stages)     │
│  Stage 1:  Court Detection                  │
│  Stage 2:  Player Detection + Tracking      │
│  Stage 3:  Shuttle Tracking (TrackNetV3)    │
│  Stage 4:  Pose Estimation (RTMPose)        │
│  Stage 5:  Hit Frame Localization           │
│  Stage 6:  Stroke Classification (BST)      │
│  Stage 7:  Player Attribution               │
│  Stage 8:  Rally Segmentation               │
│  Stage 9:  Court Position Analytics         │
│  Stage 10: Footwork Analytics               │
│  Stage 11: Fitness Analytics                │
│  Stage 12: Tactical Analytics               │
│  Stage 13: Technical Analytics              │
│  Stage 14: Coach Recommendations            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Storage (local filesystem)                  │
│  video.mp4 | court.json | players.json      │
│  shuttle.parquet | pose.parquet             │
│  shots.parquet | rallies.parquet            │
│  report.json                                 │
└─────────────────────────────────────────────┘
```

### 2.2 Deployment Model

- **Local server + web UI** for MVP
- FastAPI runs on localhost, React frontend served via Vite dev server or built static files
- No cloud dependencies — all processing local
- Docker optional for reproducibility

---

## 3. Pipeline Stage Interface

### 3.1 Common Stage Contract

Every processing stage implements a common interface:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

@dataclass
class StageResult:
    status: str          # "success" | "error" | "skipped"
    artifacts: dict      # output file paths keyed by artifact name
    metadata: dict       # stage-specific statistics
    error: str | None    # error message if failed

class PipelineStage(Protocol):
    name: str
    input_keys: list[str]    # artifacts this stage reads
    output_keys: list[str]   # artifacts this stage writes

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult: ...
```

### 3.2 ArtifactStore

A typed dictionary backed by the filesystem. Each stage declares what it reads and writes, making the DAG explicit and enabling partial re-runs (e.g., re-run stroke classification without re-detecting the court).

### 3.3 Data Format Conventions

| Artifact | Format | Rationale |
|----------|--------|-----------|
| `court.json` | JSON | Small config, human-readable |
| `players.json` | JSON | Small config |
| `shuttle.parquet` | Parquet | High-volume time-series, efficient |
| `pose.parquet` | Parquet | High-volume time-series |
| `shots.parquet` | Parquet | Tabular, queryable with DuckDB |
| `rallies.parquet` | Parquet | Tabular |
| `report.json` | JSON | Final output, human-readable |

---

## 4. Processing Pipeline — Stage Details

### Stage 1: Court Detection (FR-1)

- **Input:** Video frames (sampled at 1 FPS for initial detection)
- **Method:** Hough line transform + court template matching against standard badminton court geometry
- **Output:** `court.json` — homography matrix, court key points, net line position
- **Fallback:** Manual calibration UI — user clicks 4 court corners in the video player to define the court quadrilateral; homography is computed from these points
- **Key points detected:** Singles sidelines, baselines, service lines, net line

### Stage 2: Player Detection + Tracking (FR-2)

- **Input:** Video frames, court homography
- **Method:**
  1. YOLOv8 person detection per frame
  2. ByteTrack for multi-object tracking across frames
- **Identity persistence (3 levels):**
  - Level 1: Court-side constraint (near/far half)
  - Level 2: Appearance embeddings (color histogram)
  - Level 3: Lightweight ReID CNN (if needed)
- **Output:** `players.json` — per-frame bounding boxes + persistent player IDs + near/far assignment

### Stage 3: Shuttle Tracking (FR-4)

- **Input:** Video frames
- **Model:** TrackNetV3 (sliding window of 5 consecutive frames)
- **Output:** `shuttle.parquet` — per-frame shuttle (x, y, confidence)
- **Performance:** Must match TrackNetV3 benchmark accuracy

### Stage 4: Pose Estimation (FR-3)

- **Input:** Video frames, player bounding boxes from Stage 2
- **Model:** RTMPose-lite (primary), ViTPose (fallback), YOLOv8-Pose (CPU fallback)
- **Smoothing:** One-Euro filter applied to all keypoints to reduce jitter
- **Keypoints:** 17-joint COCO format (wrist, elbow, shoulder, hip, knee, ankle mandatory)
- **Output:** `pose.parquet` — per-frame per-player keypoint coordinates + confidence

### Stage 5: Hit Frame Localization (FR-5)

**Highest priority requirement — downstream errors propagate from here.**

- **Input:** Shuttle trajectory, player poses, court data
- **Multi-signal fusion:**
  1. **Shuttle trajectory change** — direction reversal or significant angle change (primary signal)
  2. **Shuttle speed peak** — local maximum in shuttle velocity
  3. **Shuttle-racket proximity** — minimum distance between shuttle and player wrist
  4. **Arm swing velocity peak** — peak angular velocity of racket arm
- **Confidence scoring:** Weighted combination of signal agreement
- **Output:** List of hit frames with confidence scores (written to `shots.parquet` as preliminary entries)

### Stage 6: Stroke Classification (FR-6)

- **Input:** Hit frames, shuttle trajectory window, player pose window, court position, rally context (previous shot)
- **Model:** BST-CG-AP (hierarchical classification)
  - Stage 1: Overhead vs Underarm
  - Stage 2: Specific stroke subtype
- **Required MVP classes:** Serve, Short Serve, Flick Serve, Clear, Lift, Smash, Drop, Net Shot, Drive, Push, Block, Kill
- **Extended:** All ShuttleSet classes
- **Output:** `shots.parquet` updated with stroke type + confidence

### Stage 7: Player Attribution (FR-7)

- **Input:** Hit frames, player tracking, stroke classifications
- **Method:**
  1. Proximity of player to shuttle at hit frame
  2. Swing arm detection (which player's arm is in motion)
  3. Court-side constraint as tiebreaker
- **Output:** `shots.parquet` updated with `player_id` per shot

### Stage 8: Rally Segmentation (FR-8)

- **Input:** Stroke timeline, shuttle tracking
- **Method:** Detect serve initiation → rally-end patterns (shuttle lands, error, let call)
- **Output:** `rallies.parquet` — rally ID, start frame, end frame, duration, shot count

### Stage 9: Court Position Analytics (FR-9)

- **Input:** Player positions, court homography, shot data
- **Method:**
  - 9-zone grid: front/mid/rear × left/center/right
  - Per-shot zone transitions (start zone → end zone)
  - Court coverage heatmap per player per rally
- **Output:** Zone transition matrices, coverage heatmaps

### Stage 10: Footwork Analytics (FR-10)

- **Input:** Player pose (hip keypoints for COM), court data, shot data
- **Metrics:**
  - Distance covered: integral of COM trajectory per rally
  - Recovery time: time from shot execution to return to base position (base = center of the player's half-court, computed from court homography)
  - Split step detection: low-velocity pause before explosive movement
  - Adjustment steps: small corrective steps counted via velocity profile
  - Court coverage: spatial histogram of player positions
  - Balance score: COM stability metric (variance of hip midpoint)
- **Output:** Footwork metrics per rally and per match

### Stage 11: Fitness Analytics (FR-11)

- **Input:** Footwork metrics, rally data, shot data
- **Metrics:**
  - Rally intensity: average movement velocity per rally
  - Movement velocity: peak and average per rally
  - Distance covered: cumulative per game (game = set, e.g., first game to 21 points) and match
  - Fatigue trend: rolling average of recovery time, movement speed, court coverage
- **Fatigue indicators:**
  - Reduced recovery speed in later rallies
  - Reduced court coverage area
  - Increased reaction latency (time to first movement after opponent shot)
- **Output:** Fitness metrics with game/rally-level granularity

### Stage 12: Tactical Analytics (FR-12)

- **Input:** Shot data, court positions, rally sequences
- **Analyses:**
  - Shot distribution: histogram of stroke types per player
  - Direction analysis: straight vs cross-court (shuttle landing zone relative to hit position)
  - Rally construction: n-gram patterns of stroke sequences (e.g., clear → drop → net)
  - Opponent pressure response: attacking vs defensive tendency when rally extends
- **Output:** Tactical summary per player

### Stage 13: Technical Analytics (FR-13)

- **Input:** Pose data, shuttle data, court data, shot data
- **Evaluations:**
  - Overheads: contact height relative to max reach, preparation timing
  - Smashes: contact point consistency, body alignment angle at contact
  - Net play: lunge depth (knee angle), recovery speed after net shot
  - Footwork: base positioning relative to court center, recovery efficiency
- **Output:** Technical assessment scores per stroke category

### Stage 14: Coach Recommendation Engine (FR-14)

- **Input:** All analytics outputs
- **Method (MVP):** Rule-based system
  - Each rule has: name, condition (threshold on metrics), recommendation text, required evidence
  - Rules cover common coaching scenarios (smash efficiency, recovery speed, shot selection, fatigue management)
- **Output structure:**
  ```json
  {
    "strengths": ["..."],
    "weaknesses": ["..."],
    "top_3_improvements": ["..."],
    "recommended_drills": ["..."],
    "evidence": [
      {
        "finding": "Late recovery after rear-court clears",
        "metrics": ["43 occurrences", "avg recovery 1.4 sec"],
        "peer_benchmark": "0.9 sec"
      }
    ]
  }
  ```
- **Explainability requirement:** Every recommendation references supporting metrics. No unsupported claims.

---

## 5. Feature Engineering

Computed from raw pipeline outputs before analytics:

### Shuttle Features
- Speed (px/frame → m/s via homography)
- Acceleration
- Trajectory curvature
- Apex height
- Descent angle
- Landing zone (court coordinates via homography)

### Pose Features
- Joint angles: shoulder, elbow, wrist, knee
- Limb velocities
- Angular velocities
- Body rotation angle (torso orientation from shoulder-hip vector)

### Court Features
- Player court position (x, y in court coordinates)
- Opponent court position
- Distance to net
- Distance to base position
- Distance between players

---

## 6. Frontend (React + TypeScript)

### 6.1 Views

**Upload View:**
- Drag-and-drop video upload
- Format validation (MP4, MOV, AVI)
- Player selection: click player in first frame OR toggle near/far
- Processing configuration (GPU/CPU mode)

**Processing View:**
- Real-time progress via WebSocket
- Current stage indicator, ETA, frame count
- Stage-by-stage completion status

**Report View (main dashboard):**
- Video player with synchronized timeline (Video.js)
- Rally-by-rally breakdown table
- Shot distribution charts (pie/bar)
- Court heatmap visualization (canvas overlay)
- Footwork/fitness trend line charts
- Coach recommendations panel with expandable evidence
- Export to PDF option

### 6.2 Tech Stack

- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Video.js (video playback)
- Recharts (charts/visualizations)
- WebSocket (real-time progress)

---

## 7. Backend (FastAPI + Python)

### 7.1 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload video, return job ID |
| GET | `/api/jobs/{id}` | Get job status + progress |
| WS | `/api/jobs/{id}/progress` | WebSocket for real-time progress |
| GET | `/api/jobs/{id}/report` | Get completed report |
| GET | `/api/jobs/{id}/video` | Stream processed video |
| GET | `/api/jobs/{id}/artifacts/{name}` | Download specific artifact |

### 7.2 Job Management

- Each upload creates a job with unique ID
- Job state machine: `uploading → processing → complete | failed`
- Processing state tracks current stage + stage progress
- Results stored in `data/jobs/{id}/` directory

### 7.3 Processing Orchestration

```python
class PipelineOrchestrator:
    stages: list[PipelineStage]
    
    async def run(self, job_id: str, video_path: Path) -> None:
        artifacts = ArtifactStore(job_id)
        for stage in self.stages:
            await self.broadcast_progress(stage.name, "running")
            result = stage.run(artifacts, self.config)
            if result.status == "error":
                await self.broadcast_progress(stage.name, "failed", result.error)
                return
            await self.broadcast_progress(stage.name, "complete", result.metadata)
```

---

## 8. Project Structure

```
baddyCoach/
├── docs/
│   ├── superpowers/specs/    # This spec
│   └── BaddyCoachReq.txt     # Original PRD
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── api/              # Route handlers
│   │   ├── pipeline/         # Processing stages
│   │   │   ├── court.py
│   │   │   ├── players.py
│   │   │   ├── shuttle.py
│   │   │   ├── pose.py
│   │   │   ├── hits.py
│   │   │   ├── strokes.py
│   │   │   ├── attribution.py
│   │   │   ├── rallies.py
│   │   │   └── analytics/    # Analytics sub-stages
│   │   ├── coach/            # Recommendation engine
│   │   ├── models/           # Model wrappers
│   │   └── storage/          # ArtifactStore, file I/O
│   ├── config/
│   │   └── rules.yaml        # Coach recommendation rules
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── views/
│   │   ├── hooks/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── jobs/                 # Per-job storage
└── docker-compose.yml        # Optional
```

---

## 9. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| CV/ML | TrackNetV3, BST-CG-AP, RTMPose | State-of-art, pre-trained, research-validated |
| ML Framework | PyTorch, ONNX Runtime | Standard, good GPU support |
| Backend | Python, FastAPI | Async support, auto-docs, Python ML ecosystem |
| Frontend | React, TypeScript, Vite | Modern, fast dev experience |
| Video | Video.js | Reliable, extensible |
| Charts | Recharts | React-native, declarative |
| Data | Parquet, DuckDB | Efficient tabular storage, SQL queries |
| Styling | Tailwind CSS | Utility-first, fast prototyping |
| Deployment | Docker (optional) | Reproducibility |

---

## 10. Phased Implementation Plan

### Phase 1: Core CV Pipeline
1. Court calibration + homography
2. Player tracking + player selection
3. TrackNetV3 shuttle tracking integration
4. Hit-frame localization engine
5. BST stroke classification integration
6. Rally segmentation

### Phase 2: Analytics + Coaching
7. Court position analytics
8. Footwork analytics
9. Fitness analytics
10. Tactical analytics
11. Technical analytics
12. Coach recommendation engine

### Phase 3: Frontend + Integration
13. FastAPI backend with job management
14. React frontend — upload + processing views
15. Report dashboard with video player
16. Export (PDF)

---

## 11. Non-Goals (MVP)

- Live coaching / real-time analysis
- Doubles / mixed doubles
- Wearable integration
- Audio analysis
- Heart-rate integration
- Automatic drill video generation
- Cloud deployment

---

## 12. Open Questions for Future Phases

- Doubles player tracking (4-player attribution)
- LLM-based coach recommendations (vs rule-based)
- Batch processing (multiple matches)
- Player profile aggregation across matches
- Mobile app frontend
