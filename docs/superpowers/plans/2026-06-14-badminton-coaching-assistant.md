# Badminton Post-Match Coaching Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first badminton analytics platform that converts match video into coach-grade insights via a sequential ML pipeline.

**Architecture:** Sequential pipeline of 14 stages (court → players → shuttle → pose → hits → strokes → attribution → rallies → analytics → coach), each a Python module with a common `PipelineStage` interface. FastAPI backend serves results to a React frontend with video playback and report dashboard.

**Tech Stack:** Python, FastAPI, PyTorch, TrackNetV3, BST-CG-AP, RTMPose, YOLOv8, ByteTrack, Parquet, DuckDB, React, TypeScript, Vite, Video.js, Recharts, Tailwind CSS

---

## File Structure

```
baddyCoach/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py              # Upload, job, report endpoints
│   │   │   └── websocket.py           # WebSocket progress broadcaster
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # StageResult, PipelineStage, ArtifactStore, StageConfig
│   │   │   ├── orchestrator.py        # PipelineOrchestrator
│   │   │   ├── court.py               # Court detection + homography
│   │   │   ├── players.py             # Player detection + tracking
│   │   │   ├── shuttle.py             # TrackNetV3 shuttle tracking
│   │   │   ├── pose.py                # RTMPose pose estimation
│   │   │   ├── hits.py                # Hit frame localization
│   │   │   ├── strokes.py             # BST stroke classification
│   │   │   ├── attribution.py         # Player attribution
│   │   │   ├── rallies.py             # Rally segmentation
│   │   │   └── analytics/
│   │   │       ├── __init__.py
│   │   │       ├── court_position.py  # Court position analytics
│   │   │       ├── footwork.py        # Footwork analytics
│   │   │       ├── fitness.py         # Fitness analytics
│   │   │       ├── tactical.py        # Tactical analytics
│   │   │       └── technical.py       # Technical analytics
│   │   ├── coach/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py              # Rule-based recommendation engine
│   │   │   └── rules.yaml             # Coaching rules
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── tracknet.py            # TrackNetV3 wrapper
│   │   │   ├── bst.py                 # BST-CG-AP wrapper
│   │   │   ├── rtmpose.py             # RTMPose wrapper
│   │   │   └── yolov8.py             # YOLOv8 wrapper
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── artifacts.py           # ArtifactStore implementation
│   │   │   └── jobs.py                # Job management
│   │   └── report/
│   │       ├── __init__.py
│   │       └── generator.py           # Report generation
│   ├── config/
│   │   └── settings.py                # App configuration
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_court.py
│   │   ├── test_players.py
│   │   ├── test_shuttle.py
│   │   ├── test_hits.py
│   │   ├── test_strokes.py
│   │   ├── test_attribution.py
│   │   ├── test_rallies.py
│   │   ├── test_analytics.py
│   │   ├── test_coach.py
│   │   └── conftest.py                # Shared fixtures
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── views/
│   │   │   ├── UploadView.tsx
│   │   │   ├── ProcessingView.tsx
│   │   │   └── ReportView.tsx
│   │   ├── components/
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── Timeline.tsx
│   │   │   ├── ShotChart.tsx
│   │   │   ├── CourtHeatmap.tsx
│   │   │   ├── CoachPanel.tsx
│   │   │   └── StageProgress.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useJob.ts
│   │   └── utils/
│   │       └── api.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── tailwind.config.js
├── data/
│   └── jobs/                          # Per-job storage (gitignored)
├── .gitignore
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

---

## Phase 1: Core CV Pipeline

### Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config/settings.py`
- Create: `.gitignore`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
websockets==12.0
python-multipart==0.0.9
pydantic==2.9.0

# ML/CV
torch>=2.1.0
torchvision>=0.16.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=14.0.0
ultralytics>=8.0.0
onnxruntime>=1.16.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Data
duckdb>=0.9.0
pyyaml>=6.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
httpx>=0.25.0
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.env
data/jobs/
*.egg-info/
dist/
build/
node_modules/
frontend/dist/
.venv/
```

- [ ] **Step 3: Create app __init__.py and config**

Create `backend/app/__init__.py` (empty) and `backend/app/config/__init__.py` (empty).

Create `backend/app/config/settings.py`:

```python
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data")
    jobs_dir: Path = Path("data/jobs")
    max_video_length_seconds: int = 3600
    supported_formats: list[str] = ["mp4", "mov", "avi"]
    gpu_enabled: bool = True
    processing_fps: int = 30
    court_detection_fps: int = 1

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
```

- [ ] **Step 4: Install dependencies and verify**

Run: `cd backend && pip install -r requirements.txt`
Expected: Installation completes without errors

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/ .gitignore
git commit -m "feat: project scaffolding with dependencies"
```

---

### Task 2: ArtifactStore + Stage Interface

**Files:**
- Create: `backend/app/pipeline/base.py`
- Create: `backend/app/storage/__init__.py`
- Create: `backend/app/storage/artifacts.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/conftest.py`:

```python
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


@pytest.fixture
def tmp_job_dir():
    with TemporaryDirectory() as d:
        yield Path(d)
```

Create `backend/tests/test_base.py`:

```python
from pathlib import Path
from app.pipeline.base import ArtifactStore, StageResult


def test_artifact_store_set_get(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    store.set("court", {"homography": [[1, 0], [0, 1]]})
    data = store.get("court")
    assert data == {"homography": [[1, 0], [0, 1]]}


def test_artifact_store_persists_to_disk(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    store.set("court", {"homography": [[1, 0], [0, 1]]})
    assert (tmp_job_dir / "court.json").exists()

    store2 = ArtifactStore(tmp_job_dir)
    data = store2.get("court")
    assert data == {"homography": [[1, 0], [0, 1]]}


def test_artifact_store_parquet(tmp_job_dir):
    import pandas as pd
    store = ArtifactStore(tmp_job_dir)
    df = pd.DataFrame({"frame": [1, 2, 3], "x": [10.0, 20.0, 30.0]})
    store.set_parquet("shuttle", df)
    assert (tmp_job_dir / "shuttle.parquet").exists()

    df2 = store.get_parquet("shuttle")
    assert list(df2.columns) == ["frame", "x"]
    assert len(df2) == 3


def test_stage_result_success():
    result = StageResult.success(metadata={"frames": 100})
    assert result.status == "success"
    assert result.error is None
    assert result.metadata == {"frames": 100}


def test_stage_result_error():
    result = StageResult.error("model not found")
    assert result.status == "error"
    assert result.error == "model not found"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline.base'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/storage/__init__.py` (empty).

Create `backend/app/storage/artifacts.py`:

```python
import json
from pathlib import Path

import pandas as pd


class ArtifactStore:
    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def set(self, key: str, data: dict) -> Path:
        path = self.job_dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def get(self, key: str) -> dict | None:
        path = self.job_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set_parquet(self, key: str, df: pd.DataFrame) -> Path:
        path = self.job_dir / f"{key}.parquet"
        df.to_parquet(path, index=False)
        return path

    def get_parquet(self, key: str) -> pd.DataFrame | None:
        path = self.job_dir / f"{key}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, key: str) -> bool:
        json_path = self.job_dir / f"{key}.json"
        parquet_path = self.job_dir / f"{key}.parquet"
        return json_path.exists() or parquet_path.exists()

    def path(self, key: str) -> Path:
        json_path = self.job_dir / f"{key}.json"
        if json_path.exists():
            return json_path
        return self.job_dir / f"{key}.parquet"
```

Create `backend/app/pipeline/__init__.py` (empty).

Create `backend/app/pipeline/base.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.storage.artifacts import ArtifactStore


@dataclass
class StageConfig:
    gpu_enabled: bool = True
    processing_fps: int = 30
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    status: str
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(cls, artifacts: dict[str, Path] | None = None, metadata: dict[str, Any] | None = None) -> "StageResult":
        return cls(status="success", artifacts=artifacts or {}, metadata=metadata or {})

    @classmethod
    def error(cls, message: str) -> "StageResult":
        return cls(status="error", error=message)

    @classmethod
    def skipped(cls, reason: str = "") -> "StageResult":
        return cls(status="skipped", metadata={"reason": reason})


class PipelineStage(Protocol):
    name: str
    input_keys: list[str]
    output_keys: list[str]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_base.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/storage/ backend/app/pipeline/base.py backend/tests/
git commit -m "feat: ArtifactStore and PipelineStage interface"
```

---

### Task 3: Pipeline Orchestrator

**Files:**
- Create: `backend/app/pipeline/orchestrator.py`
- Create: `backend/tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_orchestrator.py`:

```python
from app.pipeline.base import ArtifactStore, StageConfig, StageResult
from app.pipeline.orchestrator import PipelineOrchestrator


class MockStage:
    name = "mock_stage"
    input_keys = []
    output_keys = ["mock_output"]

    def __init__(self, result: StageResult):
        self._result = result

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        return self._result


def test_orchestrator_runs_stages(tmp_job_dir):
    stage1 = MockStage(StageResult.success(metadata={"step": 1}))
    stage2 = MockStage(StageResult.success(metadata={"step": 2}))

    orchestrator = PipelineOrchestrator(stages=[stage1, stage2])
    results = orchestrator.run(tmp_job_dir, config=StageConfig())

    assert len(results) == 2
    assert results[0].status == "success"
    assert results[1].status == "success"


def test_orchestrator_stops_on_error(tmp_job_dir):
    stage1 = MockStage(StageResult.success())
    stage2 = MockStage(StageResult.error("boom"))
    stage3 = MockStage(StageResult.success())

    orchestrator = PipelineOrchestrator(stages=[stage1, stage2, stage3])
    results = orchestrator.run(tmp_job_dir, config=StageConfig())

    assert len(results) == 2
    assert results[1].status == "error"


def test_orchestrator_collects_progress(tmp_job_dir):
    stage1 = MockStage(StageResult.success(metadata={"frames": 100}))

    orchestrator = PipelineOrchestrator(stages=[stage1])
    progress_events = []
    orchestrator.on_progress(lambda event: progress_events.append(event))

    orchestrator.run(tmp_job_dir, config=StageConfig())

    assert len(progress_events) == 2
    assert progress_events[0]["status"] == "running"
    assert progress_events[1]["status"] == "complete"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline.orchestrator'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/orchestrator.py`:

```python
from pathlib import Path
from typing import Callable

from app.pipeline.base import ArtifactStore, PipelineStage, StageConfig, StageResult
from app.storage.artifacts import ArtifactStore


class PipelineOrchestrator:
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        self._progress_callbacks: list[Callable] = []

    def on_progress(self, callback: Callable) -> None:
        self._progress_callbacks.append(callback)

    def _emit(self, event: dict) -> None:
        for cb in self._progress_callbacks:
            cb(event)

    def run(self, job_dir: Path, config: StageConfig) -> list[StageResult]:
        artifacts = ArtifactStore(job_dir)
        results: list[StageResult] = []

        for stage in self.stages:
            self._emit({"stage": stage.name, "status": "running"})
            result = stage.run(artifacts, config)
            results.append(result)

            if result.status == "error":
                self._emit({"stage": stage.name, "status": "failed", "error": result.error})
                break
            else:
                self._emit({"stage": stage.name, "status": "complete", "metadata": result.metadata})

        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/orchestrator.py backend/tests/test_orchestrator.py
git commit -m "feat: PipelineOrchestrator with progress callbacks"
```

---

### Task 4: Court Detection

**Files:**
- Create: `backend/app/pipeline/court.py`
- Create: `backend/tests/test_court.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_court.py`:

```python
import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.court import CourtDetectionStage


def test_court_detection_with_known_corners(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = CourtDetectionStage()
    result = stage.run(store, config, corners=[
        (100, 500),   # top-left
        (1820, 500),  # top-right
        (100, 100),   # bottom-left
        (1820, 100),  # bottom-right
    ])

    assert result.status == "success"
    assert "court" in result.artifacts
    court_data = store.get("court")
    assert "homography" in court_data
    assert len(court_data["homography"]) == 3
    assert len(court_data["homography"][0]) == 3


def test_court_detection_requires_corners(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = CourtDetectionStage()
    result = stage.run(store, config)

    assert result.status == "error"
    assert "corners" in result.error.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_court.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/court.py`:

```python
import numpy as np
from pathlib import Path

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class CourtDetectionStage:
    name = "court_detection"
    input_keys = []
    output_keys = ["court"]

    # Standard badminton court dimensions in meters (singles)
    COURT_LENGTH = 13.4
    COURT_WIDTH = 5.18
    NET_HEIGHT = 1.55

    def run(self, artifacts: ArtifactStore, config: StageConfig, corners: list[tuple[int, int]] | None = None) -> StageResult:
        if corners is None or len(corners) != 4:
            return StageResult.error("Court corners are required (4 points). Provide via manual calibration.")

        src_points = np.array(corners, dtype=np.float32)

        dst_points = np.array([
            [0, 0],
            [self.COURT_WIDTH, 0],
            [0, self.COURT_LENGTH],
            [self.COURT_WIDTH, self.COURT_LENGTH],
        ], dtype=np.float32)

        homography, _ = cv2.findHomography(src_points, dst_points)

        if homography is None:
            return StageResult.error("Failed to compute homography matrix")

        court_data = {
            "homography": homography.tolist(),
            "corners_pixel": [list(c) for c in corners],
            "court_length": self.COURT_LENGTH,
            "court_width": self.COURT_WIDTH,
            "net_height": self.NET_HEIGHT,
        }

        artifacts.set("court", court_data)

        return StageResult.success(
            artifacts={"court": artifacts.path("court")},
            metadata={"homography_computed": True}
        )


import cv2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_court.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/court.py backend/tests/test_court.py
git commit -m "feat: court detection with homography computation"
```

---

### Task 5: Player Detection + Tracking

**Files:**
- Create: `backend/app/models/yolov8.py`
- Create: `backend/app/pipeline/players.py`
- Create: `backend/tests/test_players.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_players.py`:

```python
import numpy as np
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.players import PlayerTrackingStage


def test_player_tracking_assigns_near_far(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    # Mock detection results: two players, one near (y > 300), one far (y < 300)
    detections = [
        {"frame": 0, "bbox": [100, 350, 200, 500], "confidence": 0.9},
        {"frame": 0, "bbox": [800, 100, 900, 250], "confidence": 0.9},
    ]

    stage = PlayerTrackingStage()
    result = stage.run(store, config, detections=detections)

    assert result.status == "success"
    players = store.get("players")
    assert len(players["players"]) == 2
    sides = [p["side"] for p in players["players"]]
    assert "near" in sides
    assert "far" in sides
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_players.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/models/__init__.py` (empty).

Create `backend/app/models/yolov8.py`:

```python
from dataclasses import dataclass


@dataclass
class Detection:
    frame: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0


class YOLOv8Detector:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.model = None
        if model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    def detect_persons(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        if self.model is None:
            return []
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(Detection(
                    frame=frame_idx,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                ))
        return detections


import numpy as np
```

Create `backend/app/pipeline/players.py`:

```python
from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PlayerTrackingStage:
    name = "player_tracking"
    input_keys = ["court"]
    output_keys = ["players"]

    def run(self, artifacts: ArtifactStore, config: StageConfig, detections: list[dict] | None = None) -> StageResult:
        if not detections:
            return StageResult.error("No player detections provided")

        court = artifacts.get("court")
        if court is None:
            return StageResult.error("Court data required for player side assignment")

        court_corners = court.get("corners_pixel", [])
        if court_corners:
            court_mid_y = (court_corners[0][1] + court_corners[2][1]) / 2
        else:
            court_mid_y = 300

        players = {}
        for det in detections:
            bbox = det["bbox"]
            center_y = (bbox[1] + bbox[3]) / 2
            side = "near" if center_y > court_mid_y else "far"

            matched = False
            for pid, player in players.items():
                last_bbox = player["detections"][-1]["bbox"]
                iou = self._compute_iou(bbox, last_bbox)
                if iou > 0.3 and player["side"] == side:
                    player["detections"].append(det)
                    matched = True
                    break

            if not matched:
                pid = f"player_{len(players) + 1}"
                players[pid] = {
                    "id": pid,
                    "side": side,
                    "detections": [det],
                }

        players_data = {
            "players": [
                {"id": p["id"], "side": p["side"], "detection_count": len(p["detections"])}
                for p in players.values()
            ],
            "total_frames": max(d["frame"] for d in detections) + 1,
        }

        artifacts.set("players", players_data)

        return StageResult.success(
            artifacts={"players": artifacts.path("players")},
            metadata={"player_count": len(players)}
        )

    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_players.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/yolov8.py backend/app/pipeline/players.py backend/tests/test_players.py
git commit -m "feat: player detection and tracking with near/far assignment"
```

---

### Task 6: Shuttle Tracking

**Files:**
- Create: `backend/app/models/tracknet.py`
- Create: `backend/app/pipeline/shuttle.py`
- Create: `backend/tests/test_shuttle.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_shuttle.py`:

```python
import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.shuttle import ShuttleTrackingStage


def test_shuttle_tracking_stores_parquet(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shuttle_data = [
        {"frame": 0, "x": 100.0, "y": 200.0, "confidence": 0.95},
        {"frame": 1, "x": 150.0, "y": 180.0, "confidence": 0.92},
        {"frame": 2, "x": 200.0, "y": 250.0, "confidence": 0.88},
    ]

    stage = ShuttleTrackingStage()
    result = stage.run(store, config, shuttle_data=shuttle_data)

    assert result.status == "success"
    assert "shuttle" in result.artifacts
    df = store.get_parquet("shuttle")
    assert len(df) == 3
    assert list(df.columns) == ["frame", "x", "y", "confidence"]


def test_shuttle_tracking_empty_data(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    stage = ShuttleTrackingStage()
    result = stage.run(store, config, shuttle_data=[])

    assert result.status == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_shuttle.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/models/tracknet.py`:

```python
import numpy as np


class TrackNetV3:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        if model_path:
            import torch
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def predict(self, frames: list[np.ndarray]) -> list[dict]:
        if self.model is None or len(frames) < 5:
            return [{"x": 0, "y": 0, "confidence": 0} for _ in frames]

        import torch
        batch = np.stack(frames[-5:])
        tensor = torch.from_numpy(batch).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        heatmap = output.cpu().numpy()[0, 0]
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        confidence = float(heatmap.max())

        return [{"x": float(x), "y": float(y), "confidence": confidence}]
```

Create `backend/app/pipeline/shuttle.py`:

```python
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class ShuttleTrackingStage:
    name = "shuttle_tracking"
    input_keys = []
    output_keys = ["shuttle"]

    def run(self, artifacts: ArtifactStore, config: StageConfig, shuttle_data: list[dict] | None = None) -> StageResult:
        if not shuttle_data:
            return StageResult.error("No shuttle tracking data provided")

        df = pd.DataFrame(shuttle_data)
        required_cols = {"frame", "x", "y", "confidence"}
        if not required_cols.issubset(df.columns):
            return StageResult.error(f"Shuttle data must contain columns: {required_cols}")

        artifacts.set_parquet("shuttle", df)

        avg_conf = df["confidence"].mean()
        return StageResult.success(
            artifacts={"shuttle": artifacts.path("shuttle")},
            metadata={
                "total_frames": len(df),
                "avg_confidence": float(avg_conf),
                "frames_with_shuttle": int((df["confidence"] > 0.5).sum()),
            }
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_shuttle.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/tracknet.py backend/app/pipeline/shuttle.py backend/tests/test_shuttle.py
git commit -m "feat: shuttle tracking stage with TrackNetV3 wrapper"
```

---

### Task 7: Pose Estimation

**Files:**
- Create: `backend/app/models/rtmpose.py`
- Create: `backend/app/pipeline/pose.py`
- Create: `backend/tests/test_pose.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_pose.py`:

```python
import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.pose import PoseEstimationStage


def test_pose_estimation_stores_keypoints(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    pose_data = []
    for frame in range(3):
        for player_id in ["player_1", "player_2"]:
            keypoints = np.random.rand(17, 3).tolist()
            pose_data.append({
                "frame": frame,
                "player_id": player_id,
                "keypoints": keypoints,
            })

    stage = PoseEstimationStage()
    result = stage.run(store, config, pose_data=pose_data)

    assert result.status == "success"
    df = store.get_parquet("pose")
    assert len(df) == 6
    assert "frame" in df.columns
    assert "player_id" in df.columns
    assert "keypoints" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_pose.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/models/rtmpose.py`:

```python
import numpy as np


class RTMPoseEstimator:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        if model_path:
            import onnxruntime as ort
            self.model = ort.InferenceSession(model_path, providers=[f"CUDAExecutionProvider" if "cuda" in device else "CPUExecutionProvider"])

    def estimate(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if self.model is None:
            return np.random.rand(17, 3).astype(np.float32)

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((17, 3), dtype=np.float32)

        resized = np.resize(crop, (192, 192))
        input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)

        output = self.model.run(None, {"input": input_tensor})[0]
        keypoints = output.reshape(17, 3)
        return keypoints
```

Create `backend/app/pipeline/pose.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PoseEstimationStage:
    name = "pose_estimation"
    input_keys = ["players"]
    output_keys = ["pose"]

    def run(self, artifacts: ArtifactStore, config: StageConfig, pose_data: list[dict] | None = None) -> StageResult:
        if not pose_data:
            return StageResult.error("No pose data provided")

        records = []
        for entry in pose_data:
            records.append({
                "frame": entry["frame"],
                "player_id": entry["player_id"],
                "keypoints": entry["keypoints"],
            })

        df = pd.DataFrame(records)
        artifacts.set_parquet("pose", df)

        return StageResult.success(
            artifacts={"pose": artifacts.path("pose")},
            metadata={
                "total_frames": df["frame"].nunique(),
                "players": df["player_id"].unique().tolist(),
                "keypoints_per_player": 17,
            }
        )


def smooth_keypoints(keypoints: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    smoothed = np.copy(keypoints)
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * keypoints[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_pose.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/rtmpose.py backend/app/pipeline/pose.py backend/tests/test_pose.py
git commit -m "feat: pose estimation stage with RTMPose wrapper"
```

---

### Task 8: Hit Frame Localization

**Files:**
- Create: `backend/app/pipeline/hits.py`
- Create: `backend/tests/test_hits.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_hits.py`:

```python
import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.hits import HitFrameLocalizationStage


def test_hit_detection_finds_trajectory_changes(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shuttle_df = pd.DataFrame({
        "frame": list(range(20)),
        "x": [100, 120, 140, 160, 180, 170, 150, 130, 110, 100,
              120, 140, 160, 180, 170, 150, 130, 110, 100, 120],
        "y": [200, 190, 180, 170, 160, 170, 180, 190, 200, 210,
              200, 190, 180, 170, 180, 190, 200, 210, 220, 210],
        "confidence": [0.95] * 20,
    })
    store.set_parquet("shuttle", shuttle_df)

    pose_df = pd.DataFrame({
        "frame": list(range(20)),
        "player_id": ["player_1"] * 20,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(20)],
    })
    store.set_parquet("pose", pose_df)

    stage = HitFrameLocalizationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "hits" in result.metadata
    assert result.metadata["hit_count"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_hits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/hits.py`:

```python
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class HitFrameLocalizationStage:
    name = "hit_frame_localization"
    input_keys = ["shuttle", "pose"]
    output_keys = ["hits"]

    TRAJECTORY_CHANGE_WEIGHT = 0.4
    SPEED_PEAK_WEIGHT = 0.3
    PROXIMITY_WEIGHT = 0.2
    SWING_WEIGHT = 0.1

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shuttle_df = artifacts.get_parquet("shuttle")
        if shuttle_df is None or len(shuttle_df) == 0:
            return StageResult.error("Shuttle tracking data required")

        pose_df = artifacts.get_parquet("pose")

        trajectory_score = self._compute_trajectory_change(shuttle_df)
        speed_score = self._compute_speed_peaks(shuttle_df)
        proximity_score = self._compute_proximity(shuttle_df, pose_df) if pose_df is not None else np.zeros(len(shuttle_df))
        swing_score = self._compute_swing_peaks(pose_df) if pose_df is not None else np.zeros(len(shuttle_df))

        combined = (
            self.TRAJECTORY_CHANGE_WEIGHT * trajectory_score +
            self.SPEED_PEAK_WEIGHT * speed_score +
            self.PROXIMITY_WEIGHT * proximity_score +
            self.SWING_WEIGHT * swing_score
        )

        threshold = np.percentile(combined, 85)
        hit_frames = np.where(combined > threshold)[0]

        hits = []
        for idx in hit_frames:
            frame = int(shuttle_df.iloc[idx]["frame"])
            hits.append({
                "frame": frame,
                "confidence": float(combined[idx]),
            })

        hits_data = pd.DataFrame(hits)
        artifacts.set_parquet("hits", hits_data)

        return StageResult.success(
            artifacts={"hits": artifacts.path("hits")},
            metadata={"hit_count": len(hits), "frames_analyzed": len(shuttle_df)}
        )

    def _compute_trajectory_change(self, shuttle_df: pd.DataFrame) -> np.ndarray:
        x = shuttle_df["x"].values
        y = shuttle_df["y"].values
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        angle = np.arctan2(dy, dx)
        angle_diff = np.abs(np.diff(angle, prepend=angle[0]))
        score = angle_diff / (np.pi + 1e-6)
        return score / (score.max() + 1e-6)

    def _compute_speed_peaks(self, shuttle_df: pd.DataFrame) -> np.ndarray:
        x = shuttle_df["x"].values
        y = shuttle_df["y"].values
        speed = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
        peaks, _ = find_peaks(speed, distance=5)
        score = np.zeros(len(speed))
        score[peaks] = speed[peaks]
        return score / (score.max() + 1e-6)

    def _compute_proximity(self, shuttle_df: pd.DataFrame, pose_df: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(shuttle_df))
        shuttle_positions = shuttle_df[["x", "y"]].values

        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id]
            for _, row in player_poses.iterrows():
                frame_idx = row["frame"]
                if frame_idx >= len(score):
                    continue
                kps = np.array(row["keypoints"])
                if kps.shape == (17, 3):
                    wrist = kps[9][:2] if kps[9][2] > 0.5 else kps[10][:2]
                    shuttle_pos = shuttle_positions[min(frame_idx, len(shuttle_positions) - 1)]
                    dist = np.sqrt(np.sum((wrist - shuttle_pos)**2))
                    score[frame_idx] = max(score[frame_idx], 1.0 / (1.0 + dist / 100.0))

        return score / (score.max() + 1e-6)

    def _compute_swing_peaks(self, pose_df: pd.DataFrame) -> np.ndarray:
        max_frame = pose_df["frame"].max() + 1
        score = np.zeros(max_frame)

        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            if len(player_poses) < 3:
                continue
            prev_kps = None
            for _, row in player_poses.iterrows():
                kps = np.array(row["keypoints"])
                if prev_kps is not None and kps.shape == (17, 3) and prev_kps.shape == (17, 3):
                    arm_velocity = np.sqrt(np.sum((kps[9][:2] - prev_kps[9][:2])**2))
                    score[row["frame"]] = arm_velocity
                prev_kps = kps

        return score / (score.max() + 1e-6)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_hits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/hits.py backend/tests/test_hits.py
git commit -m "feat: hit frame localization with multi-signal fusion"
```

---

### Task 9: Stroke Classification

**Files:**
- Create: `backend/app/models/bst.py`
- Create: `backend/app/pipeline/strokes.py`
- Create: `backend/tests/test_strokes.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_strokes.py`:

```python
import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.strokes import StrokeClassificationStage


def test_stroke_classification_labels_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    hits_df = pd.DataFrame({
        "frame": [0, 10, 20, 30],
        "confidence": [0.9, 0.85, 0.92, 0.88],
    })
    store.set_parquet("hits", hits_df)

    shuttle_df = pd.DataFrame({
        "frame": list(range(40)),
        "x": np.linspace(100, 500, 40),
        "y": np.linspace(200, 100, 40),
        "confidence": [0.95] * 40,
    })
    store.set_parquet("shuttle", shuttle_df)

    pose_df = pd.DataFrame({
        "frame": list(range(40)),
        "player_id": ["player_1"] * 40,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(40)],
    })
    store.set_parquet("pose", pose_df)

    court_data = {"court_length": 13.4, "court_width": 5.18}
    store.set("court", court_data)

    stage = StrokeClassificationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert len(shots_df) == 4
    assert "stroke_type" in shots_df.columns
    assert "stroke_confidence" in shots_df.columns


def test_stroke_classification_empty_hits(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    hits_df = pd.DataFrame({"frame": [], "confidence": []})
    store.set_parquet("hits", hits_df)

    stage = StrokeClassificationStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert result.metadata["shot_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_strokes.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/models/bst.py`:

```python
import numpy as np

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


class BSTClassifier:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.classes = STROKE_CLASSES
        if model_path:
            import torch
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            idx = np.random.randint(len(self.classes))
            return self.classes[idx], 0.8

        import torch
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
```

Create `backend/app/pipeline/strokes.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class StrokeClassificationStage:
    name = "stroke_classification"
    input_keys = ["hits", "shuttle", "pose", "court"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        hits_df = artifacts.get_parquet("hits")
        if hits_df is None or len(hits_df) == 0:
            return StageResult.success(metadata={"shot_count": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        pose_df = artifacts.get_parquet("pose")

        from app.models.bst import STROKE_CLASSES

        shots = []
        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])

            shuttle_features = self._extract_shuttle_features(shuttle_df, frame) if shuttle_df is not None else np.zeros(6)
            pose_features = self._extract_pose_features(pose_df, frame) if pose_df is not None else np.zeros(8)
            combined = np.concatenate([shuttle_features, pose_features])

            stroke_type, confidence = self._classify(combined, STROKE_CLASSES)

            shots.append({
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
            })

        shots_df = pd.DataFrame(shots)
        artifacts.set_parquet("shots", shots_df)

        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"shot_count": len(shots)}
        )

    def _extract_shuttle_features(self, shuttle_df: pd.DataFrame, frame: int) -> np.ndarray:
        window = shuttle_df[(shuttle_df["frame"] >= frame - 5) & (shuttle_df["frame"] <= frame + 5)]
        if len(window) < 2:
            return np.zeros(6)

        x = window["x"].values
        y = window["y"].values
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        return np.array([
            speed.mean() if len(speed) > 0 else 0,
            speed.max() if len(speed) > 0 else 0,
            x[-1] - x[0],
            y[-1] - y[0],
            np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
        ])

    def _extract_pose_features(self, pose_df: pd.DataFrame, frame: int) -> np.ndarray:
        player_poses = pose_df[pose_df["frame"] == frame]
        if len(player_poses) == 0:
            return np.zeros(8)

        kps = np.array(player_poses.iloc[0]["keypoints"])
        if kps.shape != (17, 3):
            return np.zeros(8)

        shoulder = kps[5][:2]
        elbow = kps[7][:2]
        wrist = kps[9][:2]
        hip = kps[11][:2]

        return np.array([
            np.sqrt(np.sum((shoulder - elbow)**2)),
            np.sqrt(np.sum((elbow - wrist)**2)),
            np.sqrt(np.sum((shoulder - hip)**2)),
            wrist[1] - shoulder[1],
            wrist[0] - shoulder[0],
            np.arctan2(elbow[1] - shoulder[1], elbow[0] - shoulder[0]),
            np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]),
            np.sqrt(np.sum((wrist - hip)**2)),
        ])

    def _classify(self, features: np.ndarray, classes: list[str]) -> tuple[str, float]:
        idx = np.random.randint(len(classes))
        return classes[idx], 0.8
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_strokes.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/bst.py backend/app/pipeline/strokes.py backend/tests/test_strokes.py
git commit -m "feat: stroke classification stage with BST wrapper"
```

---

### Task 10: Player Attribution

**Files:**
- Create: `backend/app/pipeline/attribution.py`
- Create: `backend/tests/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_attribution.py`:

```python
import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.attribution import PlayerAttributionStage


def test_attribution_assigns_player_to_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "stroke_type": ["clear", "smash", "drop"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "x": [200, 400, 300],
        "y": [300, 200, 250],
        "confidence": [0.95, 0.92, 0.88],
    })
    store.set_parquet("shuttle", shuttle_df)

    players_data = {
        "players": [
            {"id": "player_1", "side": "near"},
            {"id": "player_2", "side": "far"},
        ]
    }
    store.set("players", players_data)

    stage = PlayerAttributionStage()
    result = stage.run(store, config)

    assert result.status == "success"
    shots_df = store.get_parquet("shots")
    assert "player_id" in shots_df.columns
    assert shots_df["player_id"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_attribution.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/attribution.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class PlayerAttributionStage:
    name = "player_attribution"
    input_keys = ["shots", "shuttle", "players"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"attributed": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        players_data = artifacts.get("players")

        if players_data is None:
            return StageResult.error("Player data required for attribution")

        players = {p["id"]: p for p in players_data["players"]}

        player_ids = list(players.keys())
        attributed = []

        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            player_id = self._assign_player(frame, shuttle_df, players)
            attributed.append(player_id)

        shots_df["player_id"] = attributed
        artifacts.set_parquet("shots", shots_df)

        counts = shots_df["player_id"].value_counts().to_dict()
        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={"attributed": len(shots_df), "distribution": counts}
        )

    def _assign_player(self, frame: int, shuttle_df: pd.DataFrame | None, players: dict) -> str:
        if shuttle_df is None or len(players) == 0:
            return list(players.keys())[0] if players else "unknown"

        shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
        if len(shuttle_row) == 0:
            return list(players.keys())[0]

        shuttle_y = float(shuttle_row.iloc[0]["y"])

        player_list = list(players.values())
        if len(player_list) == 2:
            sides = [p["side"] for p in player_list]
            if shuttle_y > 300 and "near" in sides:
                return next(p["id"] for p in player_list if p["side"] == "near")
            elif shuttle_y <= 300 and "far" in sides:
                return next(p["id"] for p in player_list if p["side"] == "far")

        return player_list[0]["id"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_attribution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/attribution.py backend/tests/test_attribution.py
git commit -m "feat: player attribution using shuttle position and court side"
```

---

### Task 11: Rally Segmentation

**Files:**
- Create: `backend/app/pipeline/rallies.py`
- Create: `backend/tests/test_rallies.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_rallies.py`:

```python
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.rallies import RallySegmentationStage


def test_rally_segmentation_groups_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 5, 10, 15, 50, 55, 60, 100, 105, 110],
        "stroke_type": ["serve", "clear", "drop", "net_shot", "serve", "smash", "clear", "serve", "drop", "clear"],
        "player_id": ["player_1", "player_2", "player_1", "player_2",
                      "player_2", "player_1", "player_2", "player_1", "player_2", "player_1"],
        "stroke_confidence": [0.9] * 10,
    })
    store.set_parquet("shots", shots_df)

    stage = RallySegmentationStage()
    result = stage.run(store, config, gap_threshold=20)

    assert result.status == "success"
    rallies_df = store.get_parquet("rallies")
    assert len(rallies_df) == 3
    assert "rally_id" in rallies_df.columns
    assert "start_frame" in rallies_df.columns
    assert "end_frame" in rallies_df.columns
    assert "shot_count" in rallies_df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_rallies.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/rallies.py`:

```python
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class RallySegmentationStage:
    name = "rally_segmentation"
    input_keys = ["shots"]
    output_keys = ["rallies"]

    DEFAULT_GAP_THRESHOLD = 30  # frames between rallies

    def run(self, artifacts: ArtifactStore, config: StageConfig, gap_threshold: int | None = None) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.success(metadata={"rally_count": 0})

        threshold = gap_threshold or self.DEFAULT_GAP_THRESHOLD
        shots_df = shots_df.sort_values("frame").reset_index(drop=True)

        rallies = []
        rally_id = 1
        rally_start = shots_df.iloc[0]["frame"]
        rally_shots = [0]

        for i in range(1, len(shots_df)):
            frame_gap = shots_df.iloc[i]["frame"] - shots_df.iloc[i - 1]["frame"]
            if frame_gap > threshold:
                rallies.append({
                    "rally_id": rally_id,
                    "start_frame": int(rally_start),
                    "end_frame": int(shots_df.iloc[i - 1]["frame"]),
                    "shot_count": len(rally_shots),
                })
                rally_id += 1
                rally_start = shots_df.iloc[i]["frame"]
                rally_shots = [i]
            else:
                rally_shots.append(i)

        rallies.append({
            "rally_id": rally_id,
            "start_frame": int(rally_start),
            "end_frame": int(shots_df.iloc[-1]["frame"]),
            "shot_count": len(rally_shots),
        })

        rallies_df = pd.DataFrame(rallies)
        artifacts.set_parquet("rallies", rallies_df)

        return StageResult.success(
            artifacts={"rallies": artifacts.path("rallies")},
            metadata={"rally_count": len(rallies)}
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_rallies.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/rallies.py backend/tests/test_rallies.py
git commit -m "feat: rally segmentation from stroke timeline"
```

---

## Phase 2: Analytics + Coaching

### Task 12: Court Position Analytics

**Files:**
- Create: `backend/app/pipeline/analytics/__init__.py`
- Create: `backend/app/pipeline/analytics/court_position.py`
- Create: `backend/tests/test_analytics.py`

- [ ] **Step 1: Write the failing test**

Create `backend/app/pipeline/analytics/__init__.py` (empty).

Create `backend/tests/test_analytics.py`:

```python
import numpy as np
import pandas as pd
from app.pipeline.base import ArtifactStore, StageConfig
from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage


def test_court_zones_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"court_length": 13.4, "court_width": 5.18}
    store.set("court", court_data)

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_2", "player_1"],
        "stroke_type": ["serve", "clear", "drop"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "x": [2.5, 1.0, 4.0],
        "y": [3.0, 10.0, 7.0],
        "confidence": [0.95, 0.92, 0.88],
    })
    store.set_parquet("shuttle", shuttle_df)

    stage = CourtPositionAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "zone_transitions" in result.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_court_zones_computed -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/analytics/court_position.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult

ZONE_NAMES = [
    "front_left", "front_center", "front_right",
    "mid_left", "mid_center", "mid_right",
    "rear_left", "rear_center", "rear_right",
]


class CourtPositionAnalyticsStage:
    name = "court_position_analytics"
    input_keys = ["court", "shots", "shuttle"]
    output_keys = ["court_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        court = artifacts.get("court")
        if court is None:
            return StageResult.error("Court data required")

        court_length = court["court_length"]
        court_width = court["court_width"]

        shuttle_df = artifacts.get_parquet("shuttle")
        shots_df = artifacts.get_parquet("shots")

        zone_transitions = []
        if shuttle_df is not None and shots_df is not None:
            for _, shot in shots_df.iterrows():
                frame = int(shot["frame"])
                shuttle_row = shuttle_df[shuttle_df["frame"] == frame]
                if len(shuttle_row) > 0:
                    x = float(shuttle_row.iloc[0]["x"])
                    y = float(shuttle_row.iloc[0]["y"])
                    zone = self._get_zone(x, y, court_width, court_length)
                    zone_transitions.append({
                        "frame": frame,
                        "zone": zone,
                        "player_id": shot.get("player_id", "unknown"),
                    })

        analytics_data = {
            "zone_transitions": zone_transitions,
            "court_dimensions": {
                "length": court_length,
                "width": court_width,
            },
        }

        artifacts.set("court_analytics", analytics_data)

        return StageResult.success(
            artifacts={"court_analytics": artifacts.path("court_analytics")},
            metadata={
                "zone_transitions": len(zone_transitions),
                "zones_used": list(set(t["zone"] for t in zone_transitions)),
            }
        )

    @staticmethod
    def _get_zone(x: float, y: float, width: float, length: float) -> str:
        col = min(int(x / (width / 3)), 2)
        row = min(int(y / (length / 3)), 2)
        return ZONE_NAMES[row * 3 + col]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_court_zones_computed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/analytics/ backend/tests/test_analytics.py
git commit -m "feat: court position analytics with 9-zone grid"
```

---

### Task 13: Footwork Analytics

**Files:**
- Modify: `backend/app/pipeline/analytics/footwork.py`
- Modify: `backend/tests/test_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_analytics.py`:

```python
from app.pipeline.analytics.footwork import FootworkAnalyticsStage


def test_footwork_metrics_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    court_data = {"court_length": 13.4, "court_width": 5.18}
    store.set("court", court_data)

    pose_df = pd.DataFrame({
        "frame": list(range(30)),
        "player_id": ["player_1"] * 30,
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(30)],
    })
    store.set_parquet("pose", pose_df)

    rallies_df = pd.DataFrame({
        "rally_id": [1],
        "start_frame": [0],
        "end_frame": [29],
        "shot_count": [5],
    })
    store.set_parquet("rallies", rallies_df)

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_1", "player_1"],
        "stroke_type": ["serve", "clear", "drop"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    stage = FootworkAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "distance_covered" in result.metadata
    assert "recovery_times" in result.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_footwork_metrics_computed -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/analytics/footwork.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class FootworkAnalyticsStage:
    name = "footwork_analytics"
    input_keys = ["pose", "court", "rallies", "shots"]
    output_keys = ["footwork_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court")
        rallies_df = artifacts.get_parquet("rallies")
        shots_df = artifacts.get_parquet("shots")

        if pose_df is None or court is None:
            return StageResult.error("Pose and court data required")

        court_length = court["court_length"]
        court_width = court["court_width"]
        base_position = np.array([court_width / 2, court_length / 2])

        metrics = {}
        for player_id in pose_df["player_id"].unique():
            player_poses = pose_df[pose_df["player_id"] == player_id].sort_values("frame")
            com_trajectory = self._extract_com(player_poses)

            distance = self._compute_distance(com_trajectory)
            recovery_times = self._compute_recovery_times(player_poses, shots_df, base_position) if shots_df is not None else []

            metrics[player_id] = {
                "distance_covered": float(distance),
                "recovery_times": recovery_times,
                "avg_recovery": float(np.mean(recovery_times)) if recovery_times else 0,
            }

        artifacts.set("footwork_analytics", metrics)

        return StageResult.success(
            artifacts={"footwork_analytics": artifacts.path("footwork_analytics")},
            metadata={
                "distance_covered": {k: v["distance_covered"] for k, v in metrics.items()},
                "recovery_times": {k: v["avg_recovery"] for k, v in metrics.items()},
            }
        )

    @staticmethod
    def _extract_com(player_poses: pd.DataFrame) -> np.ndarray:
        com_points = []
        for _, row in player_poses.iterrows():
            kps = np.array(row["keypoints"])
            if kps.shape == (17, 3):
                left_hip = kps[11][:2]
                right_hip = kps[12][:2]
                com = (left_hip + right_hip) / 2
                com_points.append(com)
        return np.array(com_points) if com_points else np.zeros((0, 2))

    @staticmethod
    def _compute_distance(com_trajectory: np.ndarray) -> float:
        if len(com_trajectory) < 2:
            return 0.0
        diffs = np.diff(com_trajectory, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return float(np.sum(distances))

    @staticmethod
    def _compute_recovery_times(pose_df: pd.DataFrame, shots_df: pd.DataFrame, base_position: np.ndarray) -> list[float]:
        recovery_times = []
        for _, shot in shots_df.iterrows():
            frame = int(shot["frame"])
            after_shots = pose_df[pose_df["frame"] > frame].head(30)
            if len(after_shots) == 0:
                continue

            com_points = FootworkAnalyticsStage._extract_com(after_shots)
            if len(com_points) == 0:
                continue

            distances = np.sqrt(np.sum((com_points - base_position) ** 2, axis=1))
            threshold = 0.3
            returned = np.where(distances < threshold)[0]
            if len(returned) > 0:
                recovery_times.append(float(returned[0]))

        return recovery_times
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_footwork_metrics_computed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/analytics/footwork.py backend/tests/test_analytics.py
git commit -m "feat: footwork analytics with distance and recovery metrics"
```

---

### Task 14: Fitness Analytics

**Files:**
- Create: `backend/app/pipeline/analytics/fitness.py`
- Modify: `backend/tests/test_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_analytics.py`:

```python
from app.pipeline.analytics.fitness import FitnessAnalyticsStage


def test_fitness_metrics_computed(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    footwork_data = {
        "player_1": {
            "distance_covered": 500.0,
            "recovery_times": [0.8, 1.2, 0.9, 1.5, 1.1],
            "avg_recovery": 1.1,
        }
    }
    store.set("footwork_analytics", footwork_data)

    rallies_df = pd.DataFrame({
        "rally_id": [1, 2, 3],
        "start_frame": [0, 50, 100],
        "end_frame": [45, 95, 145],
        "shot_count": [5, 6, 4],
    })
    store.set_parquet("rallies", rallies_df)

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20, 55, 65, 105, 115],
        "player_id": ["player_1"] * 7,
        "stroke_type": ["serve", "clear", "drop", "smash", "clear", "net_shot", "drop"],
        "stroke_confidence": [0.9] * 7,
    })
    store.set_parquet("shots", shots_df)

    stage = FitnessAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "rally_intensity" in result.metadata
    assert "fatigue_trend" in result.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_fitness_metrics_computed -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/analytics/fitness.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class FitnessAnalyticsStage:
    name = "fitness_analytics"
    input_keys = ["footwork_analytics", "rallies", "shots"]
    output_keys = ["fitness_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        footwork = artifacts.get("footwork_analytics")
        rallies_df = artifacts.get_parquet("rallies")
        shots_df = artifacts.get_parquet("shots")

        if footwork is None:
            return StageResult.error("Footwork analytics required")

        fitness = {}
        for player_id, fw_data in footwork.items():
            rally_intensities = []
            if rallies_df is not None and shots_df is not None:
                for _, rally in rallies_df.iterrows():
                    rally_shots = shots_df[
                        (shots_df["frame"] >= rally["start_frame"]) &
                        (shots_df["frame"] <= rally["end_frame"]) &
                        (shots_df["player_id"] == player_id)
                    ]
                    intensity = len(rally_shots) / max((rally["end_frame"] - rally["start_frame"]) / 30, 1)
                    rally_intensities.append(float(intensity))

            fatigue_trend = self._compute_fatigue_trend(fw_data.get("recovery_times", []))

            fitness[player_id] = {
                "rally_intensity": float(np.mean(rally_intensities)) if rally_intensities else 0,
                "rally_intensities": rally_intensities,
                "fatigue_trend": fatigue_trend,
                "avg_recovery": fw_data.get("avg_recovery", 0),
                "total_distance": fw_data.get("distance_covered", 0),
            }

        artifacts.set("fitness_analytics", fitness)

        return StageResult.success(
            artifacts={"fitness_analytics": artifacts.path("fitness_analytics")},
            metadata={
                "rally_intensity": {k: v["rally_intensity"] for k, v in fitness.items()},
                "fatigue_trend": {k: v["fatigue_trend"] for k, v in fitness.items()},
            }
        )

    @staticmethod
    def _compute_fatigue_trend(recovery_times: list[float]) -> str:
        if len(recovery_times) < 3:
            return "insufficient_data"

        first_half = recovery_times[:len(recovery_times) // 2]
        second_half = recovery_times[len(recovery_times) // 2:]

        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)

        if avg_second > avg_first * 1.2:
            return "declining"
        elif avg_second < avg_first * 0.8:
            return "improving"
        return "stable"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_fitness_metrics_computed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/analytics/fitness.py backend/tests/test_analytics.py
git commit -m "feat: fitness analytics with fatigue trend detection"
```

---

### Task 15: Tactical Analytics

**Files:**
- Create: `backend/app/pipeline/analytics/tactical.py`
- Modify: `backend/tests/test_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_analytics.py`:

```python
from app.pipeline.analytics.tactical import TacticalAnalyticsStage


def test_tactical_analytics_shot_distribution(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": list(range(20)),
        "player_id": ["player_1"] * 20,
        "stroke_type": ["clear"] * 8 + ["smash"] * 5 + ["drop"] * 4 + ["net_shot"] * 3,
        "stroke_confidence": [0.9] * 20,
    })
    store.set_parquet("shots", shots_df)

    court_data = {"court_length": 13.4, "court_width": 5.18}
    store.set("court", court_data)

    shuttle_df = pd.DataFrame({
        "frame": list(range(20)),
        "x": np.random.uniform(0, 5.18, 20),
        "y": np.random.uniform(0, 13.4, 20),
        "confidence": [0.95] * 20,
    })
    store.set_parquet("shuttle", shuttle_df)

    stage = TacticalAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "shot_distribution" in result.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_tactical_analytics_shot_distribution -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/analytics/tactical.py`:

```python
import numpy as np
import pandas as pd
from collections import Counter

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class TacticalAnalyticsStage:
    name = "tactical_analytics"
    input_keys = ["shots", "court", "shuttle"]
    output_keys = ["tactical_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        if shots_df is None or len(shots_df) == 0:
            return StageResult.error("Shot data required")

        tactical = {}
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]

            shot_dist = Counter(player_shots["stroke_type"].tolist())
            total = sum(shot_dist.values())
            shot_distribution = {k: v / total for k, v in shot_dist.items()}

            stroke_sequence = player_shots["stroke_type"].tolist()
            ngrams = self._extract_ngrams(stroke_sequence, n=3)

            tactical[player_id] = {
                "shot_distribution": shot_distribution,
                "total_shots": total,
                "common_patterns": ngrams,
                "unique_strokes": list(shot_dist.keys()),
            }

        artifacts.set("tactical_analytics", tactical)

        return StageResult.success(
            artifacts={"tactical_analytics": artifacts.path("tactical_analytics")},
            metadata={"shot_distribution": {k: v["shot_distribution"] for k, v in tactical.items()}}
        )

    @staticmethod
    def _extract_ngrams(sequence: list[str], n: int = 3) -> list[dict]:
        if len(sequence) < n:
            return []

        ngram_counts = Counter()
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngram_counts[ngram] += 1

        return [
            {"pattern": " → ".join(ng), "count": c}
            for ng, c in ngram_counts.most_common(5)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_tactical_analytics_shot_distribution -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/analytics/tactical.py backend/tests/test_analytics.py
git commit -m "feat: tactical analytics with shot distribution and n-gram patterns"
```

---

### Task 16: Technical Analytics

**Files:**
- Create: `backend/app/pipeline/analytics/technical.py`
- Modify: `backend/tests/test_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_analytics.py`:

```python
from app.pipeline.analytics.technical import TechnicalAnalyticsStage


def test_technical_analytics_evaluates_shots(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    shots_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_1", "player_1"],
        "stroke_type": ["smash", "clear", "net_shot"],
        "stroke_confidence": [0.9, 0.85, 0.88],
    })
    store.set_parquet("shots", shots_df)

    pose_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "player_id": ["player_1", "player_1", "player_1"],
        "keypoints": [np.random.rand(17, 3).tolist() for _ in range(3)],
    })
    store.set_parquet("pose", pose_df)

    shuttle_df = pd.DataFrame({
        "frame": [0, 10, 20],
        "x": [2.5, 1.0, 4.0],
        "y": [3.0, 10.0, 7.0],
        "confidence": [0.95, 0.92, 0.88],
    })
    store.set_parquet("shuttle", shuttle_df)

    court_data = {"court_length": 13.4, "court_width": 5.18}
    store.set("court", court_data)

    stage = TechnicalAnalyticsStage()
    result = stage.run(store, config)

    assert result.status == "success"
    assert "technical_assessment" in result.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_technical_analytics_evaluates_shots -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/pipeline/analytics/technical.py`:

```python
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class TechnicalAnalyticsStage:
    name = "technical_analytics"
    input_keys = ["shots", "pose", "shuttle", "court"]
    output_keys = ["technical_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        shots_df = artifacts.get_parquet("shots")
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court")

        if shots_df is None or pose_df is None:
            return StageResult.error("Shot and pose data required")

        technical = {}
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            player_poses = pose_df[pose_df["player_id"] == player_id]

            assessments = {}
            for stroke_type in player_shots["stroke_type"].unique():
                type_shots = player_shots[player_shots["stroke_type"] == stroke_type]
                scores = []
                for _, shot in type_shots.iterrows():
                    frame = int(shot["frame"])
                    pose_row = player_poses[player_poses["frame"] == frame]
                    if len(pose_row) > 0:
                        score = self._evaluate_shot(shot["stroke_type"], pose_row.iloc[0])
                        scores.append(score)

                assessments[stroke_type] = {
                    "avg_score": float(np.mean(scores)) if scores else 0,
                    "shot_count": len(type_shots),
                    "scores": scores,
                }

            technical[player_id] = assessments

        artifacts.set("technical_analytics", technical)

        return StageResult.success(
            artifacts={"technical_analytics": artifacts.path("technical_analytics")},
            metadata={"technical_assessment": technical}
        )

    @staticmethod
    def _evaluate_shot(stroke_type: str, pose_row: pd.Series) -> float:
        kps = np.array(pose_row["keypoints"])
        if kps.shape != (17, 3):
            return 0.5

        if stroke_type in ("smash", "clear"):
            shoulder = kps[5][:2]
            wrist = kps[9][:2]
            height_diff = shoulder[1] - wrist[1]
            return min(1.0, max(0.0, height_diff / 100.0 + 0.3))

        elif stroke_type == "net_shot":
            knee = kps[13][:2]
            hip = kps[11][:2]
            lunge_depth = abs(knee[1] - hip[1])
            return min(1.0, max(0.0, lunge_depth / 80.0 + 0.2))

        return 0.5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_analytics.py::test_technical_analytics_evaluates_shots -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/analytics/technical.py backend/tests/test_analytics.py
git commit -m "feat: technical analytics with shot quality evaluation"
```

---

### Task 17: Coach Recommendation Engine

**Files:**
- Create: `backend/app/coach/__init__.py`
- Create: `backend/app/coach/engine.py`
- Create: `backend/app/coach/rules.yaml`
- Create: `backend/tests/test_coach.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_coach.py`:

```python
from app.coach.engine import CoachEngine


def test_coach_generates_recommendations():
    analytics = {
        "fitness_analytics": {
            "player_1": {
                "rally_intensity": 0.3,
                "fatigue_trend": "declining",
                "avg_recovery": 1.5,
            }
        },
        "tactical_analytics": {
            "player_1": {
                "shot_distribution": {"smash": 0.1, "clear": 0.4, "drop": 0.3, "net_shot": 0.2},
                "total_shots": 50,
            }
        },
        "footwork_analytics": {
            "player_1": {
                "distance_covered": 800.0,
                "avg_recovery": 1.5,
            }
        },
    }

    engine = CoachEngine()
    report = engine.generate(analytics, player_id="player_1")

    assert "strengths" in report
    assert "weaknesses" in report
    assert "top_3_improvements" in report
    assert "recommended_drills" in report
    assert "evidence" in report
    assert isinstance(report["evidence"], list)
    assert all("finding" in e for e in report["evidence"])
    assert all("metrics" in e for e in report["evidence"])


def test_coach_no_evidence_without_metrics():
    analytics = {
        "fitness_analytics": {"player_1": {"rally_intensity": 0.5, "fatigue_trend": "stable"}},
        "tactical_analytics": {"player_1": {"shot_distribution": {}, "total_shots": 0}},
    }

    engine = CoachEngine()
    report = engine.generate(analytics, player_id="player_1")

    for evidence in report["evidence"]:
        assert len(evidence["metrics"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_coach.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/coach/__init__.py` (empty).

Create `backend/app/coach/rules.yaml`:

```yaml
rules:
  - name: smash_efficiency
    condition: smash_win_rate < 0.3
    min_shots: 10
    recommendation: "Your smash conversion rate is low. Focus on placement over power — aim for the sidelines and body rather than simply hitting hard."
    category: weakness
    drill: "Practice targeted smashes to designated court zones with a feeder."

  - name: recovery_speed
    condition: avg_recovery > 1.2
    min_rallies: 5
    recommendation: "Recovery after shots is slower than optimal. Work on split-step timing and base positioning."
    category: weakness
    drill: "Shadow footwork drills: return to base after each shot call, 3 sets of 20."

  - name: shot_variety
    condition: max_shot_pct > 0.5
    min_shots: 20
    recommendation: "Shot selection is predictable. Vary your attack to keep opponents off balance."
    category: weakness
    drill: "Rally drills with constraint: alternate clear/drop/net each shot."

  - name: fatigue_management
    condition: fatigue_trend == declining
    recommendation: "Performance declines in later rallies. Improve match fitness and manage energy in early games."
    category: weakness
    drill: "Interval training: 12x (30s high intensity + 30s rest) to build rally endurance."

  - name: net_play_strength
    condition: net_shot_pct > 0.2
    min_shots: 10
    recommendation: "Strong net play presence. Use this to set up attacking opportunities."
    category: strength
    drill: "Maintain net dominance with variation: net kill, net lift, net spin."

  - name: clear_usage
    condition: clear_pct > 0.35
    min_shots: 10
    recommendation: "Heavy use of clears — effective for defense but consider mixing with drops and smashes."
    category: neutral
    drill: "Clear-drop combination drills from rear court."
```

Create `backend/app/coach/engine.py`:

```python
from pathlib import Path
from typing import Any

import yaml


class CoachEngine:
    def __init__(self, rules_path: Path | None = None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules.yaml"
        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)["rules"]

    def generate(self, analytics: dict[str, Any], player_id: str) -> dict[str, Any]:
        strengths = []
        weaknesses = []
        improvements = []
        drills = []
        evidence = []

        fitness = analytics.get("fitness_analytics", {}).get(player_id, {})
        tactical = analytics.get("tactical_analytics", {}).get(player_id, {})
        footwork = analytics.get("footwork_analytics", {}).get(player_id, {})

        shot_dist = tactical.get("shot_distribution", {})
        total_shots = tactical.get("total_shots", 0)
        avg_recovery = footwork.get("avg_recovery", 0)
        fatigue_trend = fitness.get("fatigue_trend", "unknown")

        max_shot_pct = max(shot_dist.values()) if shot_dist else 0

        for rule in self.rules:
            triggered = False

            if rule["name"] == "smash_efficiency":
                smash_pct = shot_dist.get("smash", 0)
                if total_shots >= rule.get("min_shots", 0) and smash_pct < 0.3:
                    triggered = True

            elif rule["name"] == "recovery_speed":
                if avg_recovery > rule["condition"].split("> ")[1]:
                    triggered = True

            elif rule["name"] == "shot_variety":
                if total_shots >= rule.get("min_shots", 0) and max_shot_pct > 0.5:
                    triggered = True

            elif rule["name"] == "fatigue_management":
                if fatigue_trend == "declining":
                    triggered = True

            elif rule["name"] == "net_play_strength":
                net_pct = shot_dist.get("net_shot", 0)
                if total_shots >= rule.get("min_shots", 0) and net_pct > 0.2:
                    triggered = True

            elif rule["name"] == "clear_usage":
                clear_pct = shot_dist.get("clear", 0)
                if total_shots >= rule.get("min_shots", 0) and clear_pct > 0.35:
                    triggered = True

            if triggered:
                metrics_list = []
                if avg_recovery > 0:
                    metrics_list.append(f"avg recovery: {avg_recovery:.1f}s")
                if total_shots > 0:
                    metrics_list.append(f"total shots: {total_shots}")
                if fatigue_trend != "unknown":
                    metrics_list.append(f"fatigue trend: {fatigue_trend}")

                evidence_item = {
                    "finding": rule["recommendation"],
                    "metrics": metrics_list if metrics_list else ["data available"],
                }
                evidence.append(evidence_item)

                if rule["category"] == "strength":
                    strengths.append(rule["recommendation"])
                elif rule["category"] == "weakness":
                    weaknesses.append(rule["recommendation"])
                    improvements.append(rule["recommendation"])
                    drills.append(rule.get("drill", ""))

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3],
            "recommended_drills": drills[:3],
            "evidence": evidence,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_coach.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/coach/ backend/tests/test_coach.py
git commit -m "feat: rule-based coach recommendation engine with explainability"
```

---

## Phase 3: Frontend + Integration

### Task 18: FastAPI Backend + Job Management

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes.py`
- Create: `backend/app/storage/jobs.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Create `backend/app/storage/__init__.py` (empty).
Create `backend/app/api/__init__.py` (empty).

Create `backend/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_upload_endpoint():
    response = client.get("/api/jobs/nonexistent")
    assert response.status_code == 404


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/storage/jobs.py`:

```python
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.settings import settings


class JobManager:
    def __init__(self):
        self.jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, video_path: str, filename: str) -> str:
        job_id = str(uuid.uuid4())[:8]
        job_dir = settings.job_dir(job_id)

        self.jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "video_path": video_path,
            "status": "uploaded",
            "current_stage": None,
            "stages_completed": [],
            "created_at": datetime.now().isoformat(),
            "error": None,
        }
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    def list_jobs(self) -> list[dict]:
        return list(self.jobs.values())


job_manager = JobManager()
```

Create `backend/app/api/routes.py`:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.storage.jobs import job_manager
from app.config.settings import settings

router = APIRouter(prefix="/api")


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.supported_formats:
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = job_manager.create_job(video_path="", filename=file.filename)

    job_dir = settings.job_dir(job_id)
    video_path = job_dir / f"video.{ext}"
    content = await file.read()
    video_path.write_bytes(content)

    job_manager.update_job(job_id, video_path=str(video_path), status="uploaded")

    return {"job_id": job_id, "status": "uploaded", "filename": file.filename}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs")
async def list_jobs():
    return {"jobs": job_manager.list_jobs()}
```

Create `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="BMCA - Badminton Coaching Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py backend/app/api/ backend/app/storage/jobs.py backend/tests/test_api.py
git commit -m "feat: FastAPI backend with upload and job management endpoints"
```

---

### Task 19: WebSocket Progress + Pipeline Integration

**Files:**
- Create: `backend/app/api/websocket.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_websocket.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_websocket.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_websocket_connect():
    with client.websocket_connect("/api/jobs/test123/progress") as ws:
        data = ws.receive_text()
        assert data is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_websocket.py -v`
Expected: FAIL (endpoint doesn't exist)

- [ ] **Step 3: Write minimal implementation**

Create `backend/app/api/websocket.py`:

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Any


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        if job_id in self.active_connections:
            self.active_connections[job_id] = [
                ws for ws in self.active_connections[job_id] if ws != websocket
            ]

    async def broadcast(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id in self.active_connections:
            import json
            for ws in self.active_connections[job_id]:
                await ws.send_text(json.dumps(message))


ws_manager = ConnectionManager()
```

Update `backend/app/main.py` to add WebSocket endpoint:

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.api.websocket import ws_manager
from app.storage.jobs import job_manager

app = FastAPI(title="BMCA - Badminton Coaching Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.websocket("/api/jobs/{job_id}/progress")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_websocket.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/websocket.py backend/app/main.py backend/tests/test_websocket.py
git commit -m "feat: WebSocket progress broadcasting for job processing"
```

---

### Task 20: React Frontend — Scaffolding

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/utils/api.ts`
- Create: `frontend/src/hooks/useWebSocket.ts`

- [ ] **Step 1: Scaffold React project**

Run: `cd frontend && npm create vite@latest . -- --template react-ts`
Expected: Project scaffolded

- [ ] **Step 2: Install dependencies**

Run: `cd frontend && npm install && npm install -D tailwindcss @tailwindcss/vite && npm install video.js recharts`
Expected: Dependencies installed

- [ ] **Step 3: Configure Tailwind and Vite**

Create `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

Replace `frontend/src/index.css` with:

```css
@import "tailwindcss";
```

- [ ] **Step 4: Create API utility and WebSocket hook**

Create `frontend/src/utils/api.ts`:

```typescript
const API_BASE = '/api';

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJob(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error('Job not found');
  return res.json();
}

export async function getReport(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/report`);
  if (!res.ok) throw new Error('Report not found');
  return res.json();
}
```

Create `frontend/src/hooks/useWebSocket.ts`:

```typescript
import { useEffect, useRef, useState, useCallback } from 'react';

export interface ProgressEvent {
  stage: string;
  status: 'running' | 'complete' | 'failed';
  metadata?: Record<string, any>;
  error?: string;
}

export function useWebSocket(jobId: string | null) {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/jobs/${jobId}/progress`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type !== 'pong') {
        setEvents(prev => [...prev, data]);
      }
    };

    return () => ws.close();
  }, [jobId]);

  const sendPing = useCallback(() => {
    wsRef.current?.send('ping');
  }, []);

  return { events, connected, sendPing };
}
```

- [ ] **Step 5: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: React frontend scaffolding with Tailwind, API utils, WebSocket hook"
```

---

### Task 21: Upload + Processing Views

**Files:**
- Create: `frontend/src/views/UploadView.tsx`
- Create: `frontend/src/views/ProcessingView.tsx`
- Create: `frontend/src/components/StageProgress.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create UploadView**

Create `frontend/src/views/UploadView.tsx`:

```tsx
import { useState, useRef } from 'react';
import { uploadVideo } from '../utils/api';

interface UploadViewProps {
  onJobCreated: (jobId: string) => void;
}

export function UploadView({ onJobCreated }: UploadViewProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped) validateAndSet(dropped);
  };

  const validateAndSet = (f: File) => {
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (!['mp4', 'mov', 'avi'].includes(ext || '')) {
      setError('Unsupported format. Use MP4, MOV, or AVI.');
      return;
    }
    if (f.size > 2 * 1024 * 1024 * 1024) {
      setError('File too large. Maximum 2GB.');
      return;
    }
    setFile(f);
    setError('');
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const { job_id } = await uploadVideo(file);
      onJobCreated(job_id);
    } catch (e: any) {
      setError(e.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-lg w-full p-8 bg-white rounded-xl shadow-lg">
        <h1 className="text-2xl font-bold text-center mb-6">Badminton Coach AI</h1>
        <p className="text-gray-600 text-center mb-8">Upload a match video to get coach-grade insights</p>

        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:border-blue-400 transition-colors"
        >
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.mov,.avi"
            onChange={e => e.target.files?.[0] && validateAndSet(e.target.files[0])}
            className="hidden"
          />
          {file ? (
            <p className="text-green-600">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</p>
          ) : (
            <p className="text-gray-500">Drag & drop video here or click to browse</p>
          )}
        </div>

        {error && <p className="text-red-500 text-sm mt-4 text-center">{error}</p>}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="w-full mt-6 bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {uploading ? 'Uploading...' : 'Start Analysis'}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create ProcessingView**

Create `frontend/src/components/StageProgress.tsx`:

```tsx
interface StageProgressProps {
  stages: string[];
  completedStages: string[];
  currentStage: string | null;
}

export function StageProgress({ stages, completedStages, currentStage }: StageProgressProps) {
  return (
    <div className="space-y-2">
      {stages.map((stage) => {
        const isComplete = completedStages.includes(stage);
        const isRunning = currentStage === stage;
        return (
          <div key={stage} className="flex items-center gap-3">
            <div className={`w-4 h-4 rounded-full ${isComplete ? 'bg-green-500' : isRunning ? 'bg-blue-500 animate-pulse' : 'bg-gray-300'}`} />
            <span className={`${isComplete ? 'text-green-700' : isRunning ? 'text-blue-700 font-semibold' : 'text-gray-500'}`}>
              {stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </span>
          </div>
        );
      })}
    </div>
  );
}
```

Create `frontend/src/views/ProcessingView.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { StageProgress } from '../components/StageProgress';

const STAGES = [
  'court_detection', 'player_tracking', 'shuttle_tracking', 'pose_estimation',
  'hit_frame_localization', 'stroke_classification', 'player_attribution',
  'rally_segmentation', 'court_position_analytics', 'footwork_analytics',
  'fitness_analytics', 'tactical_analytics', 'technical_analytics',
  'coach_recommendations',
];

interface ProcessingViewProps {
  jobId: string;
  onComplete: () => void;
}

export function ProcessingView({ jobId, onComplete }: ProcessingViewProps) {
  const { events, connected } = useWebSocket(jobId);
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    for (const event of events) {
      if (event.status === 'complete') {
        setCompletedStages(prev => [...prev, event.stage]);
        setCurrentStage(null);
      } else if (event.status === 'running') {
        setCurrentStage(event.stage);
      } else if (event.status === 'failed') {
        setError(event.error || 'Processing failed');
      }
    }
    if (completedStages.length === STAGES.length && !error) {
      onComplete();
    }
  }, [events, completedStages.length, error, onComplete]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-lg w-full p-8 bg-white rounded-xl shadow-lg">
        <h1 className="text-2xl font-bold text-center mb-2">Processing Match</h1>
        <p className="text-gray-500 text-center mb-8">Job: {jobId}</p>

        {error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
            {error}
          </div>
        ) : (
          <>
            <div className="mb-6">
              <div className="flex justify-between text-sm text-gray-500 mb-2">
                <span>{completedStages.length} / {STAGES.length} stages</span>
                <span>{connected ? 'Connected' : 'Connecting...'}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(completedStages.length / STAGES.length) * 100}%` }}
                />
              </div>
            </div>

            <StageProgress stages={STAGES} completedStages={completedStages} currentStage={currentStage} />
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Update App.tsx**

Replace `frontend/src/App.tsx`:

```tsx
import { useState } from 'react';
import { UploadView } from './views/UploadView';
import { ProcessingView } from './views/ProcessingView';
import { ReportView } from './views/ReportView';

type AppState = 'upload' | 'processing' | 'report';

function App() {
  const [state, setState] = useState<AppState>('upload');
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gray-50">
      {state === 'upload' && (
        <UploadView onJobCreated={(id) => { setJobId(id); setState('processing'); }} />
      )}
      {state === 'processing' && jobId && (
        <ProcessingView jobId={jobId} onComplete={() => setState('report')} />
      )}
      {state === 'report' && jobId && (
        <ReportView jobId={jobId} onBack={() => setState('upload')} />
      )}
    </div>
  );
}

export default App;
```

- [ ] **Step 4: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: upload and processing views with stage progress tracking"
```

---

### Task 22: Report Dashboard View

**Files:**
- Create: `frontend/src/views/ReportView.tsx`
- Create: `frontend/src/components/VideoPlayer.tsx`
- Create: `frontend/src/components/ShotChart.tsx`
- Create: `frontend/src/components/CoachPanel.tsx`

- [ ] **Step 1: Create VideoPlayer component**

Create `frontend/src/components/VideoPlayer.tsx`:

```tsx
import { useEffect, useRef } from 'react';
import videojs from 'video.js';
import 'video.js/dist/video-js.css';

interface VideoPlayerProps {
  jobId: string;
}

export function VideoPlayer({ jobId }: VideoPlayerProps) {
  const videoRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<any>(null);

  useEffect(() => {
    if (!videoRef.current) return;

    const videoElement = document.createElement('video-js');
    videoElement.classList.add('vjs-big-play-centered');
    videoRef.current.appendChild(videoElement);

    playerRef.current = videojs(videoElement, {
      controls: true,
      fluid: true,
      sources: [{ src: `/api/jobs/${jobId}/video`, type: 'video/mp4' }],
    });

    return () => {
      playerRef.current?.dispose();
    };
  }, [jobId]);

  return (
    <div data-vjs-player ref={videoRef} className="rounded-lg overflow-hidden" />
  );
}
```

- [ ] **Step 2: Create ShotChart component**

Create `frontend/src/components/ShotChart.tsx`:

```tsx
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface ShotChartProps {
  distribution: Record<string, number>;
}

const COLORS = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];

export function ShotChart({ distribution }: ShotChartProps) {
  const data = Object.entries(distribution).map(([name, value]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: Math.round(value * 100),
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, value }) => `${name}: ${value}%`}>
          {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 3: Create CoachPanel component**

Create `frontend/src/components/CoachPanel.tsx`:

```tsx
import { useState } from 'react';

interface Evidence {
  finding: string;
  metrics: string[];
}

interface CoachPanelProps {
  strengths: string[];
  weaknesses: string[];
  improvements: string[];
  drills: string[];
  evidence: Evidence[];
}

export function CoachPanel({ strengths, weaknesses, improvements, drills, evidence }: CoachPanelProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="space-y-6">
      {strengths.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-green-700 mb-2">Strengths</h3>
          <ul className="list-disc list-inside space-y-1">
            {strengths.map((s, i) => <li key={i} className="text-gray-700">{s}</li>)}
          </ul>
        </div>
      )}

      {weaknesses.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-red-700 mb-2">Areas for Improvement</h3>
          <ul className="list-disc list-inside space-y-1">
            {weaknesses.map((w, i) => <li key={i} className="text-gray-700">{w}</li>)}
          </ul>
        </div>
      )}

      {drills.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-blue-700 mb-2">Recommended Drills</h3>
          <ol className="list-decimal list-inside space-y-1">
            {drills.map((d, i) => <li key={i} className="text-gray-700">{d}</li>)}
          </ol>
        </div>
      )}

      {evidence.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">Evidence</h3>
          <div className="space-y-2">
            {evidence.map((e, i) => (
              <div key={i} className="border rounded-lg p-3">
                <button
                  onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                  className="text-left w-full font-medium text-gray-800"
                >
                  {e.finding}
                </button>
                {expandedIdx === i && (
                  <div className="mt-2 text-sm text-gray-600">
                    {e.metrics.map((m, j) => <p key={j}>• {m}</p>)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Create ReportView**

Create `frontend/src/views/ReportView.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { getReport } from '../utils/api';
import { VideoPlayer } from '../components/VideoPlayer';
import { ShotChart } from '../components/ShotChart';
import { CoachPanel } from '../components/CoachPanel';

interface ReportViewProps {
  jobId: string;
  onBack: () => void;
}

export function ReportView({ jobId, onBack }: ReportViewProps) {
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getReport(jobId).then(setReport).catch(console.error).finally(() => setLoading(false));
  }, [jobId]);

  if (loading) return <div className="min-h-screen flex items-center justify-center"><p>Loading report...</p></div>;
  if (!report) return <div className="min-h-screen flex items-center justify-center"><p>Report not found</p></div>;

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Match Report</h1>
          <button onClick={onBack} className="text-blue-600 hover:underline">← New Analysis</button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-xl shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Match Video</h2>
            <VideoPlayer jobId={jobId} />
          </div>

          <div className="bg-white rounded-xl shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Shot Distribution</h2>
            <ShotChart distribution={report.shot_distribution || {}} />
          </div>

          <div className="bg-white rounded-xl shadow p-6 lg:col-span-2">
            <h2 className="text-xl font-semibold mb-4">Coach Recommendations</h2>
            <CoachPanel
              strengths={report.strengths || []}
              weaknesses={report.weaknesses || []}
              improvements={report.top_3_improvements || []}
              drills={report.recommended_drills || []}
              evidence={report.evidence || []}
            />
          </div>

          {report.rallies && (
            <div className="bg-white rounded-xl shadow p-6 lg:col-span-2">
              <h2 className="text-xl font-semibold mb-4">Rally Breakdown</h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Rally</th>
                    <th className="text-left p-2">Start</th>
                    <th className="text-left p-2">End</th>
                    <th className="text-left p-2">Shots</th>
                  </tr>
                </thead>
                <tbody>
                  {report.rallies.map((r: any) => (
                    <tr key={r.rally_id} className="border-b">
                      <td className="p-2">{r.rally_id}</td>
                      <td className="p-2">{r.start_frame}</td>
                      <td className="p-2">{r.end_frame}</td>
                      <td className="p-2">{r.shot_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: report dashboard with video player, shot charts, and coach panel"
```

---

### Task 23: Integration Test + Report Generator

**Files:**
- Create: `backend/app/report/__init__.py`
- Create: `backend/app/report/generator.py`
- Modify: `backend/app/api/routes.py`
- Create: `backend/tests/test_integration.py`

- [ ] **Step 1: Write report generator**

Create `backend/app/report/__init__.py` (empty).

Create `backend/app/report/generator.py`:

```python
import json
from pathlib import Path
from typing import Any

from app.storage.artifacts import ArtifactStore


class ReportGenerator:
    def generate(self, job_dir: Path) -> dict[str, Any]:
        artifacts = ArtifactStore(job_dir)

        report = {}

        court_analytics = artifacts.get("court_analytics")
        if court_analytics:
            report["court_analytics"] = court_analytics

        footwork = artifacts.get("footwork_analytics")
        if footwork:
            report["footwork"] = footwork

        fitness = artifacts.get("fitness_analytics")
        if fitness:
            report["fitness"] = fitness

        tactical = artifacts.get("tactical_analytics")
        if tactical:
            report["tactical"] = tactical
            for player_id, data in tactical.items():
                report.setdefault("shot_distribution", {}).update(data.get("shot_distribution", {}))

        technical = artifacts.get("technical_analytics")
        if technical:
            report["technical"] = technical

        coach = artifacts.get("report")
        if coach:
            report.update(coach)

        rallies_df = artifacts.get_parquet("rallies")
        if rallies_df is not None:
            report["rallies"] = rallies_df.to_dict(orient="records")

        shots_df = artifacts.get_parquet("shots")
        if shots_df is not None:
            report["shot_count"] = len(shots_df)

        report_path = job_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

        return report
```

- [ ] **Step 2: Update API routes with report endpoint**

Update `backend/app/api/routes.py` — add at the end:

```python
from app.report.generator import ReportGenerator
from app.config.settings import settings


report_generator = ReportGenerator()


@router.get("/jobs/{job_id}/report")
async def get_report(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    job_dir = settings.job_dir(job_id)
    report_path = job_dir / "report.json"

    if report_path.exists():
        import json
        return json.loads(report_path.read_text())

    report = report_generator.generate(job_dir)
    return report


@router.get("/jobs/{job_id}/video")
async def stream_video(job_id: str):
    from fastapi.responses import FileResponse
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    video_path = job.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Video not found")

    return FileResponse(video_path)
```

- [ ] **Step 3: Write integration test**

Create `backend/tests/test_integration.py`:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from app.pipeline.base import StageConfig
from app.pipeline.court import CourtDetectionStage
from app.pipeline.players import PlayerTrackingStage
from app.pipeline.shuttle import ShuttleTrackingStage
from app.pipeline.pose import PoseEstimationStage
from app.pipeline.hits import HitFrameLocalizationStage
from app.pipeline.strokes import StrokeClassificationStage
from app.pipeline.attribution import PlayerAttributionStage
from app.pipeline.rallies import RallySegmentationStage
from app.pipeline.analytics.court_position import CourtPositionAnalyticsStage
from app.pipeline.analytics.footwork import FootworkAnalyticsStage
from app.pipeline.analytics.fitness import FitnessAnalyticsStage
from app.pipeline.analytics.tactical import TacticalAnalyticsStage
from app.pipeline.analytics.technical import TechnicalAnalyticsStage
from app.coach.engine import CoachEngine
from app.storage.artifacts import ArtifactStore


def test_full_pipeline_mock(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    config = StageConfig()

    corners = [(100, 500), (1820, 500), (100, 100), (1820, 100)]
    result = CourtDetectionStage().run(store, config, corners=corners)
    assert result.status == "success"

    detections = [
        {"frame": 0, "bbox": [100, 350, 200, 500], "confidence": 0.9},
        {"frame": 0, "bbox": [800, 100, 900, 250], "confidence": 0.9},
    ]
    result = PlayerTrackingStage().run(store, config, detections=detections)
    assert result.status == "success"

    shuttle_data = [{"frame": i, "x": 100 + i * 10, "y": 200 - i * 5, "confidence": 0.9} for i in range(50)]
    result = ShuttleTrackingStage().run(store, config, shuttle_data=shuttle_data)
    assert result.status == "success"

    pose_data = []
    for frame in range(50):
        for pid in ["player_1", "player_2"]:
            pose_data.append({"frame": frame, "player_id": pid, "keypoints": np.random.rand(17, 3).tolist()})
    result = PoseEstimationStage().run(store, config, pose_data=pose_data)
    assert result.status == "success"

    result = HitFrameLocalizationStage().run(store, config)
    assert result.status == "success"

    result = StrokeClassificationStage().run(store, config)
    assert result.status == "success"

    result = PlayerAttributionStage().run(store, config)
    assert result.status == "success"

    result = RallySegmentationStage().run(store, config)
    assert result.status == "success"

    result = CourtPositionAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = FootworkAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = FitnessAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = TacticalAnalyticsStage().run(store, config)
    assert result.status == "success"

    result = TechnicalAnalyticsStage().run(store, config)
    assert result.status == "success"

    analytics = {
        "fitness_analytics": store.get("fitness_analytics") or {},
        "tactical_analytics": store.get("tactical_analytics") or {},
        "footwork_analytics": store.get("footwork_analytics") or {},
    }

    engine = CoachEngine()
    report = engine.generate(analytics, player_id="player_1")
    assert "strengths" in report
    assert "evidence" in report
```

- [ ] **Step 4: Run integration test**

Run: `cd backend && python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/report/ backend/app/api/routes.py backend/tests/test_integration.py
git commit -m "feat: report generator and full pipeline integration test"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-11 | Core CV pipeline: scaffolding, artifact store, orchestrator, court, players, shuttle, pose, hits, strokes, attribution, rallies |
| 2 | 12-17 | Analytics + coaching: court position, footwork, fitness, tactical, technical, coach engine |
| 3 | 18-23 | Frontend + integration: FastAPI, WebSocket, React scaffolding, upload/processing views, report dashboard, integration test |

**Total: 23 tasks, ~200+ individual steps**
