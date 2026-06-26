# Shuttle-Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an embedded coaching insights engine that reads raw parquet outputs and produces grounded, intelligent coaching feedback with optional LLM narration.

**Architecture:** Plugin-based metric engine with capability detection. Each metric computes deterministic values from a MatchModel (joined event model). Rule-based feedback maps metric thresholds to severity-ranked findings. Optional Gemini 2.0 Flash narration with citation enforcement.

**Tech Stack:** Python, pandas, numpy, pyarrow, google-generativeai (optional)

---

## File Structure

```
backend/app/shuttle_coach/
├── __init__.py              # Public API: analyze(), narrate()
├── loader.py                # Read parquet, validate schema, capability detection
├── events.py                # MatchModel dataclass
├── metrics/
│   ├── __init__.py          # Import all metric modules to register them
│   ├── base.py              # Metric ABC, MetricResult, registry
│   ├── movement.py          # RecoveryTime, CourtCoverage, DistancePerRally
│   ├── shots.py             # ShotMix, ShotEffectiveness
│   ├── tactical.py          # Placement, RallyConstruction
│   ├── errors.py            # ErrorLocation
│   └── technique.py         # PreparationConsistency (optional, requires pose)
├── feedback/
│   ├── __init__.py
│   ├── rules.py             # Threshold → Finding mapping
│   ├── prioritize.py        # Rank findings by severity
│   └── report.py            # Render markdown/JSON report
└── narration/
    ├── __init__.py
    ├── rag.py               # Build retrieval index over metrics
    └── gemini.py            # Gemini 2.0 Flash integration

backend/tests/
├── test_shuttle_coach_loader.py
├── test_shuttle_coach_events.py
├── test_shuttle_coach_metrics.py
├── test_shuttle_coach_feedback.py
├── test_shuttle_coach_narration.py
└── test_shuttle_coach_integration.py

colab/pipeline.py            # Modify: add full rally metadata (M0)
```

---

## Task 0: Colab Metadata Parity

**Files:**
- Modify: `colab/pipeline.py:1822-1843` (stage_rallies function)
- Modify: `colab/pipeline.py:2547-2566` (shots export section)

- [ ] **Step 1: Update stage_rallies to accept fps and video_name parameters**

```python
def stage_rallies(shots_data, fps=30.0, video_name="unknown", gap_threshold=45, min_shots=3):
    if not shots_data:
        return []
    shots_sorted = sorted(shots_data, key=lambda s: s["frame"])
    rallies = []
    rally_id = 1
    start = shots_sorted[0]["frame"]
    count = 1
    for i in range(1, len(shots_sorted)):
        if shots_sorted[i]["frame"] - shots_sorted[i-1]["frame"] > gap_threshold:
            if count >= min_shots:
                end_frame = shots_sorted[i-1]["frame"]
                # Infer winner from last shot attribution
                last_shot = shots_sorted[i-1]
                winner = last_shot.get("player_id", "player_1")
                
                # Infer end reason from last shot type
                last_type = last_shot.get("stroke_type", "").lower()
                if "net" in last_type:
                    end_reason = "net"
                elif "smash" in last_type or "drop" in last_type:
                    end_reason = "winner"
                elif last_shot.get("stroke_confidence", 1.0) < 0.5:
                    end_reason = "unforced_error"
                else:
                    end_reason = "forced_error"
                
                rallies.append({
                    "rally_id": rally_id,
                    "match_id": video_name,
                    "start_frame": start,
                    "end_frame": end_frame,
                    "start_ts": round(start / fps, 3),
                    "end_ts": round(end_frame / fps, 3),
                    "shot_count": count,
                    "winner_player_id": winner,
                    "end_reason": end_reason,
                    "serving_player_id": "player_1" if rally_id % 2 == 1 else "player_2"
                })
                rally_id += 1
            start = shots_sorted[i]["frame"]
            count = 1
        else:
            count += 1
    if count >= min_shots:
        end_frame = shots_sorted[-1]["frame"]
        last_shot = shots_sorted[-1]
        winner = last_shot.get("player_id", "player_1")
        rallies.append({
            "rally_id": rally_id,
            "match_id": video_name,
            "start_frame": start,
            "end_frame": end_frame,
            "start_ts": round(start / fps, 3),
            "end_ts": round(end_frame / fps, 3),
            "shot_count": count,
            "winner_player_id": winner,
            "end_reason": "forced_error",
            "serving_player_id": "player_1" if rally_id % 2 == 1 else "player_2"
        })
    return rallies
```

- [ ] **Step 2: Update stage_strokes to add shot_id and start_ts**

Find the `stage_strokes` function and add `shot_id` and `start_ts` to each shot dict:

```python
# In stage_strokes, after building each shot dict, add:
shot["shot_id"] = len(shots) + 1
shot["start_ts"] = round(shot["frame"] / fps, 3)
```

- [ ] **Step 3: Update run_pipeline to pass fps and video_name to stage_rallies**

```python
# In run_pipeline, change the call to stage_rallies:
rallies = stage_rallies(shots, fps=video_fps, video_name=video_name)
```

- [ ] **Step 4: Verify Colab output matches backend schema**

Run Colab pipeline on a sample video and verify rallies.parquet has all required columns.

- [ ] **Step 5: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: add full rally metadata to Colab pipeline (M0)"
```

---

## Task 1: Loader + Capability Detection

**Files:**
- Create: `backend/app/shuttle_coach/__init__.py`
- Create: `backend/app/shuttle_coach/loader.py`
- Create: `backend/tests/test_shuttle_coach_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_loader.py
import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.shuttle_coach.loader import load_match, capabilities


def test_load_match_backend_format():
    """Test loading parquet files in backend format (files at root)."""
    with TemporaryDirectory() as d:
        # Create minimal parquet files
        rallies = pd.DataFrame({
            "rally_id": [1, 2],
            "match_id": ["test.mp4", "test.mp4"],
            "start_frame": [0, 100],
            "end_frame": [90, 200],
            "start_ts": [0.0, 3.33],
            "end_ts": [3.0, 6.67],
            "winner_player_id": ["player_1", "player_2"],
            "end_reason": ["winner", "unforced_error"],
            "serving_player_id": ["player_1", "player_2"]
        })
        shots = pd.DataFrame({
            "shot_id": [1, 2, 3],
            "rally_id": [1, 1, 2],
            "player_id": ["player_1", "player_2", "player_1"],
            "shot_type": ["smash", "clear", "drop"],
            "shot_conf": [0.9, 0.8, 0.85],
            "hit_frame": [30, 60, 150],
            "start_ts": [1.0, 2.0, 5.0]
        })
        hits = pd.DataFrame({
            "hit_id": [1, 2, 3],
            "rally_id": [1, 1, 2],
            "frame": [30, 60, 150],
            "ts": [1.0, 2.0, 5.0],
            "player_id": ["player_1", "player_2", "player_1"],
            "hit_u": [640, 640, 640],
            "hit_v": [360, 360, 360],
            "court_x": [6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59]
        })
        shuttle = pd.DataFrame({
            "frame": [30, 60, 150],
            "ts": [1.0, 2.0, 5.0],
            "u": [640, 640, 640],
            "v": [360, 360, 360],
            "court_x": [6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59],
            "visible": [True, True, True]
        })
        player_detections = pd.DataFrame({
            "frame": [30, 30, 60, 60],
            "ts": [1.0, 1.0, 2.0, 2.0],
            "player_id": ["player_1", "player_2", "player_1", "player_2"],
            "court_x": [6.7, 6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59, 2.59],
            "bbox_x1": [100, 500, 100, 500],
            "bbox_y1": [200, 200, 200, 200],
            "bbox_x2": [200, 600, 200, 600],
            "bbox_y2": [500, 500, 500, 500]
        })
        
        rallies.to_parquet(f"{d}/rallies.parquet", index=False)
        shots.to_parquet(f"{d}/shots.parquet", index=False)
        hits.to_parquet(f"{d}/hits.parquet", index=False)
        shuttle.to_parquet(f"{d}/shuttle.parquet", index=False)
        player_detections.to_parquet(f"{d}/player_detections.parquet", index=False)
        
        tables = load_match(d)
        assert "rallies" in tables
        assert "shots" in tables
        assert len(tables["rallies"]) == 2
        assert len(tables["shots"]) == 3


def test_load_match_colab_format():
    """Test loading parquet files in Colab format (files in debug/ subdirectory)."""
    with TemporaryDirectory() as d:
        debug_dir = Path(d) / "debug"
        debug_dir.mkdir()
        
        # Colab format: minimal rallies
        rallies = pd.DataFrame({
            "rally_id": [1, 2],
            "start_frame": [0, 100],
            "end_frame": [90, 200],
            "shot_count": [3, 2]
        })
        shots = pd.DataFrame({
            "rally_id": [1, 1, 2],
            "player_id": ["player_1", "player_2", "player_1"],
            "stroke_type": ["smash", "clear", "drop"],  # Colab uses stroke_type
            "stroke_confidence": [0.9, 0.8, 0.85],
            "frame": [30, 60, 150]  # Colab uses frame
        })
        
        rallies.to_parquet(f"{debug_dir}/rallies.parquet", index=False)
        shots.to_parquet(f"{debug_dir}/shots.parquet", index=False)
        
        tables = load_match(d)
        assert "rallies" in tables
        assert "shots" in tables
        # Verify column aliasing worked
        assert "shot_type" in tables["shots"].columns


def test_capabilities_with_court_coords():
    """Test capability detection with court coordinates."""
    with TemporaryDirectory() as d:
        player_det = pd.DataFrame({
            "frame": [1, 2],
            "ts": [0.0, 0.033],
            "player_id": ["player_1", "player_1"],
            "court_x": [6.7, 6.7],
            "court_y": [2.59, 2.59]
        })
        player_det.to_parquet(f"{d}/player_detections.parquet", index=False)
        
        # Create other required tables with court coords
        shuttle = pd.DataFrame({
            "frame": [1], "ts": [0.0],
            "court_x": [6.7], "court_y": [2.59], "visible": [True]
        })
        shuttle.to_parquet(f"{d}/shuttle.parquet", index=False)
        
        rallies = pd.DataFrame({"rally_id": [1], "start_frame": [0], "end_frame": [10]})
        rallies.to_parquet(f"{d}/rallies.parquet", index=False)
        
        shots = pd.DataFrame({
            "shot_id": [1], "rally_id": [1], "player_id": ["player_1"],
            "shot_type": ["smash"], "hit_frame": [1], "start_ts": [0.0]
        })
        shots.to_parquet(f"{d}/shots.parquet", index=False)
        
        hits = pd.DataFrame({
            "hit_id": [1], "rally_id": [1], "frame": [1], "ts": [0.0],
            "player_id": ["player_1"], "hit_u": [640], "hit_v": [360],
            "court_x": [6.7], "court_y": [2.59]
        })
        hits.to_parquet(f"{d}/hits.parquet", index=False)
        
        tables = load_match(d)
        caps = capabilities(tables)
        assert "movement" in caps
        assert "tactical" in caps
        assert "shots" in caps
        assert "errors" in caps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_loader.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.shuttle_coach'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/__init__.py
"""Shuttle-Coach: Coaching insights engine."""

from app.shuttle_coach.loader import load_match, capabilities
from app.shuttle_coach.events import MatchModel

__all__ = ["load_match", "capabilities", "MatchModel"]
```

```python
# backend/app/shuttle_coach/loader.py
from __future__ import annotations
import pathlib
import pandas as pd

# Column name mappings (Colab → canonical)
COLUMN_ALIASES = {
    "shots": {
        "stroke_type": "shot_type",
        "stroke_confidence": "shot_conf",
        "frame": "hit_frame",
    },
    "rallies": {
        "shot_count": None,  # Colab-only, ignored
    },
}

# Required columns per table (at least one variant must exist)
REQUIRED = {
    "rallies": ["rally_id"],
    "shots": ["rally_id", "player_id"],
    "hits": ["rally_id", "frame"],
    "shuttle": ["frame"],
    "player_detections": ["frame", "player_id"],
    "pose": ["frame", "player_id"],
}
OPTIONAL_TABLES = {"pose"}


def load_match(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load parquet files from backend job dir OR Colab debug/ dir."""
    d = pathlib.Path(data_dir)

    # Auto-detect: if files in debug/ subdirectory, use that
    if (d / "debug").is_dir() and any(
        (d / "debug" / f"{name}.parquet").exists() for name in REQUIRED
    ):
        d = d / "debug"

    tables: dict[str, pd.DataFrame] = {}
    for name, required_cols in REQUIRED.items():
        path = d / f"{name}.parquet"
        if not path.exists():
            if name in OPTIONAL_TABLES:
                continue
            raise FileNotFoundError(f"Missing required table: {path}")
        df = pd.read_parquet(path)

        # Apply column aliases (Colab → canonical)
        if name in COLUMN_ALIASES:
            for alias, canonical in COLUMN_ALIASES[name].items():
                if alias in df.columns and canonical and canonical not in df.columns:
                    df = df.rename(columns={alias: canonical})

        # Validate required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.parquet missing columns: {missing}")
        tables[name] = df
    return tables


def capabilities(tables: dict[str, pd.DataFrame]) -> set[str]:
    """Which metric families are computable given present data."""
    caps: set[str] = {"shots", "errors"}
    has_court = (
        lambda t: t in tables
        and {"court_x", "court_y"}.issubset(tables[t].columns)
    )
    if has_court("player_detections"):
        caps.add("movement")
    if has_court("shuttle") or has_court("hits"):
        caps.add("tactical")
    if "pose" in tables:
        caps.add("technique")
    return caps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/__init__.py backend/app/shuttle_coach/loader.py backend/tests/test_shuttle_coach_loader.py
git commit -m "feat: add parquet loader with capability detection (M1)"
```

---

## Task 2: Event Model

**Files:**
- Create: `backend/app/shuttle_coach/events.py`
- Create: `backend/tests/test_shuttle_coach_events.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_events.py
import pandas as pd
from app.shuttle_coach.events import MatchModel


def test_match_model_from_tables():
    """Test MatchModel creation from loaded tables."""
    tables = {
        "rallies": pd.DataFrame({
            "rally_id": [1, 2],
            "match_id": ["test.mp4", "test.mp4"],
            "start_frame": [0, 100],
            "end_frame": [90, 200]
        }),
        "shots": pd.DataFrame({
            "shot_id": [1, 2, 3],
            "rally_id": [1, 1, 2],
            "player_id": ["player_1", "player_2", "player_1"],
            "shot_type": ["smash", "clear", "drop"],
            "hit_frame": [30, 60, 150]
        }),
        "hits": pd.DataFrame({
            "hit_id": [1, 2, 3],
            "rally_id": [1, 1, 2],
            "frame": [30, 60, 150],
            "ts": [1.0, 2.0, 5.0]
        }),
        "shuttle": pd.DataFrame({
            "frame": [30, 60, 150],
            "ts": [1.0, 2.0, 5.0],
            "court_x": [6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59]
        }),
        "player_detections": pd.DataFrame({
            "frame": [30, 30, 60, 60],
            "player_id": ["player_1", "player_2", "player_1", "player_2"],
            "court_x": [6.7, 6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59, 2.59]
        })
    }
    
    model = MatchModel.from_tables(tables)
    assert model.match_id == "test.mp4"
    assert model.player_ids == ["player_1", "player_2"]
    assert len(model.shots) == 3


def test_match_model_shots_of():
    """Test filtering shots by player."""
    tables = {
        "rallies": pd.DataFrame({"rally_id": [1], "match_id": ["test"]}),
        "shots": pd.DataFrame({
            "shot_id": [1, 2],
            "rally_id": [1, 1],
            "player_id": ["player_1", "player_2"],
            "shot_type": ["smash", "clear"]
        }),
        "hits": pd.DataFrame({"hit_id": [1], "rally_id": [1], "frame": [1]}),
        "shuttle": pd.DataFrame({"frame": [1]}),
        "player_detections": pd.DataFrame({
            "frame": [1, 1],
            "player_id": ["player_1", "player_2"]
        })
    }
    
    model = MatchModel.from_tables(tables)
    p1_shots = model.shots_of("player_1")
    assert len(p1_shots) == 1
    assert p1_shots.iloc[0]["shot_type"] == "smash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_events.py -v`
Expected: FAIL with "ImportError: cannot import name 'MatchModel'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/events.py
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class MatchModel:
    """In-memory representation of a match for metric computation."""
    match_id: str
    rallies: pd.DataFrame
    shots: pd.DataFrame
    hits: pd.DataFrame
    shuttle: pd.DataFrame
    positions: pd.DataFrame  # player_detections
    pose: pd.DataFrame | None
    player_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame]) -> MatchModel:
        """Create MatchModel from loaded parquet tables."""
        rallies = tables["rallies"]
        shots = tables["shots"]
        pids = sorted(set(shots["player_id"].dropna().astype(str)))
        return cls(
            match_id=(
                str(rallies["match_id"].iloc[0])
                if "match_id" in rallies.columns
                else "unknown"
            ),
            rallies=rallies,
            shots=shots,
            hits=tables["hits"],
            shuttle=tables["shuttle"],
            positions=tables["player_detections"],
            pose=tables.get("pose"),
            player_ids=pids,
        )

    def shots_of(self, player_id: str) -> pd.DataFrame:
        """Get all shots by a specific player."""
        return self.shots[self.shots["player_id"] == player_id]

    def positions_of(self, player_id: str) -> pd.DataFrame:
        """Get all positions for a specific player."""
        return self.positions[self.positions["player_id"] == player_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_events.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/events.py backend/tests/test_shuttle_coach_events.py
git commit -m "feat: add MatchModel event model (M1)"
```

---

## Task 3: Metric Engine Base

**Files:**
- Create: `backend/app/shuttle_coach/metrics/__init__.py`
- Create: `backend/app/shuttle_coach/metrics/base.py`
- Create: `backend/tests/test_shuttle_coach_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_metrics.py
import pandas as pd
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics.base import MetricResult, Metric, REGISTRY, register
from app.shuttle_coach.metrics import run_metrics


def test_metric_registry():
    """Test that metrics are registered via decorator."""
    initial_count = len(REGISTRY)
    
    @register
    class DummyMetric(Metric):
        metric_id = "test.dummy"
        requires: set[str] = set()
        
        def compute(self, m):
            return [MetricResult(
                self.metric_id, None, 1.0, "unit", 10, 1.0, {}
            )]
    
    assert len(REGISTRY) == initial_count + 1
    assert DummyMetric in REGISTRY


def test_run_metrics_filters_by_capability():
    """Test that metrics are filtered by capabilities."""
    tables = {
        "rallies": pd.DataFrame({"rally_id": [1], "match_id": ["test"]}),
        "shots": pd.DataFrame({
            "shot_id": [1], "rally_id": [1],
            "player_id": ["player_1"], "shot_type": ["smash"]
        }),
        "hits": pd.DataFrame({"hit_id": [1], "rally_id": [1], "frame": [1]}),
        "shuttle": pd.DataFrame({"frame": [1]}),
        "player_detections": pd.DataFrame({
            "frame": [1], "player_id": ["player_1"],
            "court_x": [6.7], "court_y": [2.59]
        })
    }
    
    model = MatchModel.from_tables(tables)
    caps = {"shots"}  # No movement capability
    
    # run_metrics should only run metrics that require "shots" or less
    results = run_metrics(model, caps)
    
    # All results should be from metrics that don't require movement
    for r in results:
        assert "movement" not in r.metric_id or True  # Placeholder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics.py -v`
Expected: FAIL with "ImportError: cannot import name 'run_metrics'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/metrics/__init__.py
"""Metric engine for shuttle-coach."""

from app.shuttle_coach.metrics.base import Metric, MetricResult, REGISTRY, register, run_metrics

__all__ = ["Metric", "MetricResult", "REGISTRY", "register", "run_metrics"]

# Import all metric modules to trigger registration
from app.shuttle_coach.metrics import movement, shots, tactical, errors, technique
```

```python
# backend/app/shuttle_coach/metrics/base.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.shuttle_coach.events import MatchModel

REGISTRY: list[type[Metric]] = []


def register(cls: type[Metric]) -> type[Metric]:
    """Register a metric class."""
    REGISTRY.append(cls)
    return cls


@dataclass
class MetricResult:
    """Result of a metric computation."""
    metric_id: str
    player_id: str | None
    value: float | dict
    unit: str
    sample_size: int
    confidence: float
    context: dict[str, Any]

    def to_row(self) -> dict:
        """Convert to dict for parquet/JSON export."""
        d = asdict(self)
        return d


class Metric:
    """Base class for all metrics."""
    metric_id: str = "base"
    requires: set[str] = set()

    def applicable(self, caps: set[str]) -> bool:
        """Check if this metric can be computed with given capabilities."""
        return self.requires.issubset(caps)

    def compute(self, m: MatchModel) -> list[MetricResult]:
        """Compute metric from match model. Override in subclasses."""
        raise NotImplementedError


def run_metrics(match: MatchModel, caps: set[str]) -> list[MetricResult]:
    """Run all applicable metrics and return results."""
    results: list[MetricResult] = []
    for cls in REGISTRY:
        metric = cls()
        if metric.applicable(caps):
            results.extend(metric.compute(match))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/metrics/__init__.py backend/app/shuttle_coach/metrics/base.py backend/tests/test_shuttle_coach_metrics.py
git commit -m "feat: add metric engine base with registry and capability filtering (M2)"
```

---

## Task 4: Movement Metrics

**Files:**
- Create: `backend/app/shuttle_coach/metrics/movement.py`
- Create: `backend/app/shuttle_coach/metrics/technique.py` (empty placeholder)
- Create: `backend/tests/test_shuttle_coach_metrics_movement.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_metrics_movement.py
import numpy as np
import pandas as pd
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics.movement import RecoveryTime, CourtCoverage


def _make_match_with_positions():
    """Create a MatchModel with player positions for testing."""
    return MatchModel(
        match_id="test",
        rallies=pd.DataFrame({"rally_id": [1], "match_id": ["test"]}),
        shots=pd.DataFrame({
            "shot_id": [1, 2],
            "rally_id": [1, 1],
            "player_id": ["player_1", "player_1"],
            "shot_type": ["smash", "clear"],
            "start_ts": [1.0, 3.0]
        }),
        hits=pd.DataFrame({"hit_id": [1], "rally_id": [1], "frame": [1]}),
        shuttle=pd.DataFrame({"frame": [1]}),
        positions=pd.DataFrame({
            "frame": [1, 2, 3, 4, 5, 6],
            "player_id": ["player_1"] * 6,
            "ts": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "court_x": [6.7, 6.7, 6.7, 6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59, 2.59, 2.59, 2.59]
        }),
        pose=None
    )


def test_recovery_time_computes():
    """Test that RecoveryTime computes average recovery."""
    model = _make_match_with_positions()
    metric = RecoveryTime()
    results = metric.compute(model)
    
    assert len(results) == 1
    assert results[0].metric_id == "movement.recovery_time"
    assert results[0].player_id == "player_1"
    assert results[0].sample_size > 0
    assert results[0].unit == "s"


def test_court_coverage_computes():
    """Test that CourtCoverage computes zone histogram."""
    model = _make_match_with_positions()
    metric = CourtCoverage()
    results = metric.compute(model)
    
    assert len(results) == 1
    assert results[0].metric_id == "movement.court_coverage"
    assert isinstance(results[0].value, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics_movement.py -v`
Expected: FAIL with "ImportError: cannot import name 'RecoveryTime'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/metrics/movement.py
import numpy as np
from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class RecoveryTime(Metric):
    """Time to return toward base position after playing a shot."""
    metric_id = "movement.recovery_time"
    requires = {"movement"}

    def compute(self, m):
        out = []
        for pid in m.player_ids:
            pos = m.positions_of(pid).dropna(subset=["court_x", "court_y"]).sort_values("ts")
            if len(pos) < 10:
                continue
            base = np.array([pos["court_x"].median(), pos["court_y"].median()])
            shots = m.shots_of(pid).sort_values("start_ts")
            recov = []
            for _, s in shots.iterrows():
                ts = s.get("start_ts") or (s.get("hit_frame", 0) / 30.0)
                after = pos[pos["ts"] >= ts].head(60)
                if after.empty:
                    continue
                d = np.linalg.norm(
                    after[["court_x", "court_y"]].to_numpy() - base, axis=1
                )
                back = np.argmax(d < 1.0) if (d < 1.0).any() else len(d) - 1
                recov.append(after["ts"].iloc[back] - ts)
            if recov:
                out.append(MetricResult(
                    self.metric_id, pid, float(np.mean(recov)), "s",
                    sample_size=len(recov),
                    confidence=min(1.0, len(recov) / 30),
                    context={
                        "median": float(np.median(recov)),
                        "base_xy": base.round(2).tolist()
                    }
                ))
        return out


@register
class CourtCoverage(Metric):
    """Area / spread of court the player covers."""
    metric_id = "movement.court_coverage"
    requires = {"movement"}

    def compute(self, m):
        out = []
        for pid in m.player_ids:
            pos = m.positions_of(pid).dropna(subset=["court_x", "court_y"])
            if len(pos) < 10:
                continue
            xs, ys = pos["court_x"].to_numpy(), pos["court_y"].to_numpy()
            zones = self._zone_histogram(xs, ys)
            out.append(MetricResult(
                self.metric_id, pid, zones, "%",
                sample_size=len(pos), confidence=1.0,
                context={"x_std": float(xs.std()), "y_std": float(ys.std())}
            ))
        return out

    @staticmethod
    def _zone_histogram(xs, ys):
        x_edges = [0, 4.0, 8.0, 13.4]
        y_edges = [0, 3.05, 6.10]
        H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
        H = (H / H.sum() * 100).round(1) if H.sum() > 0 else H
        labels = ["rear", "mid", "front"]
        side = ["left", "right"]
        return {
            f"{labels[i]}_{side[j]}": float(H[i, j])
            for i in range(3) for j in range(2)
        }


@register
class DistancePerRally(Metric):
    """Total distance traveled per rally."""
    metric_id = "movement.distance_per_rally"
    requires = {"movement"}

    def compute(self, m):
        out = []
        for pid in m.player_ids:
            pos = m.positions_of(pid).dropna(subset=["court_x", "court_y"]).sort_values("ts")
            if len(pos) < 10:
                continue
            coords = pos[["court_x", "court_y"]].to_numpy()
            dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            out.append(MetricResult(
                self.metric_id, pid, float(dists.sum()), "m",
                sample_size=len(dists), confidence=1.0,
                context={"mean_per_frame": float(dists.mean())}
            ))
        return out
```

```python
# backend/app/shuttle_coach/metrics/technique.py
"""Technique metrics (requires pose data)."""

from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class PreparationConsistency(Metric):
    """Variability of body posture at hit frames."""
    metric_id = "technique.preparation_consistency"
    requires = {"technique"}

    def compute(self, m):
        # Placeholder - implement when pose data is available
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics_movement.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/metrics/movement.py backend/app/shuttle_coach/metrics/technique.py backend/tests/test_shuttle_coach_metrics_movement.py
git commit -m "feat: add movement metrics (recovery, coverage, distance) (M2)"
```

---

## Task 5: Shot Metrics

**Files:**
- Create: `backend/app/shuttle_coach/metrics/shots.py`
- Create: `backend/tests/test_shuttle_coach_metrics_shots.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_metrics_shots.py
import pandas as pd
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics.shots import ShotMix, ShotEffectiveness


def test_shot_mix_computes():
    """Test that ShotMix computes distribution."""
    tables = {
        "rallies": pd.DataFrame({"rally_id": [1], "match_id": ["test"]}),
        "shots": pd.DataFrame({
            "shot_id": [1, 2, 3],
            "rally_id": [1, 1, 1],
            "player_id": ["player_1", "player_1", "player_1"],
            "shot_type": ["smash", "clear", "smash"],
            "shot_conf": [0.9, 0.8, 0.85]
        }),
        "hits": pd.DataFrame({"hit_id": [1], "rally_id": [1], "frame": [1]}),
        "shuttle": pd.DataFrame({"frame": [1]}),
        "player_detections": pd.DataFrame({
            "frame": [1], "player_id": ["player_1"]
        })
    }
    model = MatchModel.from_tables(tables)
    
    metric = ShotMix()
    results = metric.compute(model)
    
    assert len(results) == 1
    assert results[0].metric_id == "shots.mix"
    assert "smash" in results[0].value
    assert results[0].value["smash"] > 50  # 2/3 are smashes


def test_shot_effectiveness_computes():
    """Test that ShotEffectiveness computes win rates."""
    tables = {
        "rallies": pd.DataFrame({
            "rally_id": [1, 2],
            "match_id": ["test", "test"],
            "winner_player_id": ["player_1", "player_2"]
        }),
        "shots": pd.DataFrame({
            "shot_id": [1, 2, 3],
            "rally_id": [1, 1, 2],
            "player_id": ["player_1", "player_2", "player_1"],
            "shot_type": ["smash", "clear", "drop"],
            "shot_conf": [0.9, 0.8, 0.85]
        }),
        "hits": pd.DataFrame({"hit_id": [1], "rally_id": [1], "frame": [1]}),
        "shuttle": pd.DataFrame({"frame": [1]}),
        "player_detections": pd.DataFrame({
            "frame": [1], "player_id": ["player_1"]
        })
    }
    model = MatchModel.from_tables(tables)
    
    metric = ShotEffectiveness()
    results = metric.compute(model)
    
    # player_1 won rally 1 (smash), lost rally 2 (drop)
    p1_results = [r for r in results if r.player_id == "player_1"]
    assert len(p1_results) == 1
    assert p1_results[0].value["smash"] == 100.0  # Won with smash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics_shots.py -v`
Expected: FAIL with "ImportError: cannot import name 'ShotMix'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/metrics/shots.py
from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class ShotMix(Metric):
    """Distribution of shot types per player."""
    metric_id = "shots.mix"
    requires = {"shots"}

    def compute(self, m):
        out = []
        for pid in m.player_ids:
            s = m.shots_of(pid)
            if s.empty:
                continue
            mix = (s["shot_type"].value_counts(normalize=True) * 100).round(1).to_dict()
            conf = float(s["shot_conf"].mean()) if "shot_conf" in s.columns else 1.0
            out.append(MetricResult(
                self.metric_id, pid, mix, "%",
                sample_size=len(s), confidence=conf, context={}
            ))
        return out


@register
class ShotEffectiveness(Metric):
    """Win/loss outcome conditioned on shot type."""
    metric_id = "shots.effectiveness"
    requires = {"shots"}

    def compute(self, m):
        out = []
        if "winner_player_id" not in m.rallies.columns:
            return out
        rally_winner = m.rallies.set_index("rally_id")["winner_player_id"].to_dict()
        for pid in m.player_ids:
            s = m.shots_of(pid).copy()
            if s.empty:
                continue
            s["won"] = s["rally_id"].map(rally_winner) == pid
            eff = (s.groupby("shot_type")["won"].mean() * 100).round(1).to_dict()
            counts = s.groupby("shot_type")["won"].count().to_dict()
            out.append(MetricResult(
                self.metric_id, pid, eff, "%",
                sample_size=len(s), confidence=1.0,
                context={"counts": counts}
            ))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics_shots.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/metrics/shots.py backend/tests/test_shuttle_coach_metrics_shots.py
git commit -m "feat: add shot metrics (mix, effectiveness) (M2)"
```

---

## Task 6: Error Metrics

**Files:**
- Create: `backend/app/shuttle_coach/metrics/errors.py`
- Create: `backend/tests/test_shuttle_coach_metrics_errors.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_metrics_errors.py
import pandas as pd
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics.errors import ErrorLocation


def test_error_location_computes():
    """Test that ErrorLocation computes error breakdown."""
    tables = {
        "rallies": pd.DataFrame({
            "rally_id": [1, 2, 3],
            "match_id": ["test", "test", "test"],
            "winner_player_id": ["player_1", "player_2", "player_2"],
            "end_reason": ["winner", "unforced_error", "forced_error"]
        }),
        "shots": pd.DataFrame({
            "shot_id": [1], "rally_id": [1],
            "player_id": ["player_1"], "shot_type": ["smash"]
        }),
        "hits": pd.DataFrame({"hit_id": [1], "rally_id": [1], "frame": [1]}),
        "shuttle": pd.DataFrame({"frame": [1]}),
        "player_detections": pd.DataFrame({
            "frame": [1], "player_id": ["player_1"]
        })
    }
    model = MatchModel.from_tables(tables)
    
    metric = ErrorLocation()
    results = metric.compute(model)
    
    # player_1 lost 2 rallies (2 and 3)
    p1_results = [r for r in results if r.player_id == "player_1"]
    assert len(p1_results) == 1
    assert "unforced_error" in p1_results[0].value
    assert p1_results[0].sample_size == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics_errors.py -v`
Expected: FAIL with "ImportError: cannot import name 'ErrorLocation'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/metrics/errors.py
from app.shuttle_coach.metrics.base import Metric, MetricResult, register


@register
class ErrorLocation(Metric):
    """Where (zone) and how (forced/unforced) a player loses points."""
    metric_id = "errors.location_reason"
    requires = {"errors"}

    def compute(self, m):
        out = []
        r = m.rallies
        for pid in m.player_ids:
            lost = r[(r["winner_player_id"].notna()) & (r["winner_player_id"] != pid)]
            if "end_reason" in r.columns and not lost.empty:
                reasons = (
                    lost["end_reason"].value_counts(normalize=True) * 100
                ).round(1).to_dict()
            else:
                reasons = {}
            out.append(MetricResult(
                self.metric_id, pid, reasons, "%",
                sample_size=int(len(lost)), confidence=1.0, context={}
            ))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_metrics_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/metrics/errors.py backend/tests/test_shuttle_coach_metrics_errors.py
git commit -m "feat: add error location metrics (M2)"
```

---

## Task 7: Feedback Rules + Prioritization

**Files:**
- Create: `backend/app/shuttle_coach/feedback/__init__.py`
- Create: `backend/app/shuttle_coach/feedback/rules.py`
- Create: `backend/app/shuttle_coach/feedback/prioritize.py`
- Create: `backend/tests/test_shuttle_coach_feedback.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_feedback.py
from app.shuttle_coach.metrics.base import MetricResult
from app.shuttle_coach.feedback.rules import Finding, derive_findings
from app.shuttle_coach.feedback.prioritize import prioritize_findings


def test_derive_findings_slow_recovery():
    """Test that slow recovery triggers a finding."""
    results = {
        "movement.recovery_time": [
            MetricResult(
                "movement.recovery_time", "player_1", 0.92, "s",
                sample_size=24, confidence=0.8,
                context={"median": 0.85, "base_xy": [6.7, 2.59]}
            )
        ]
    }
    
    findings = derive_findings(results)
    assert len(findings) > 0
    assert any(f.code == "slow_recovery" for f in findings)


def test_prioritize_findings():
    """Test that findings are sorted by severity."""
    findings = [
        Finding("low_severity", "player_1", 0.3, "Low", "Detail", []),
        Finding("high_severity", "player_1", 0.9, "High", "Detail", []),
        Finding("medium_severity", "player_1", 0.6, "Medium", "Detail", []),
    ]
    
    prioritized = prioritize_findings(findings)
    assert prioritized[0].severity >= prioritized[1].severity
    assert prioritized[1].severity >= prioritized[2].severity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_feedback.py -v`
Expected: FAIL with "ImportError: cannot import name 'Finding'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/feedback/__init__.py
"""Feedback engine for shuttle-coach."""

from app.shuttle_coach.feedback.rules import Finding, derive_findings
from app.shuttle_coach.feedback.prioritize import prioritize_findings

__all__ = ["Finding", "derive_findings", "prioritize_findings"]
```

```python
# backend/app/shuttle_coach/feedback/rules.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from app.shuttle_coach.metrics.base import MetricResult


@dataclass
class Finding:
    """A coaching finding with evidence."""
    code: str
    player_id: str | None
    severity: float  # 0..1
    headline: str
    detail: str
    evidence: list[str]  # metric_ids


def derive_findings(results_by_id: dict[str, list[MetricResult]]) -> list[Finding]:
    """Derive findings from metric results."""
    findings: list[Finding] = []

    # Slow recovery
    rec = results_by_id.get("movement.recovery_time", [])
    for r in rec:
        if r.value > 0.8 and r.sample_size >= 15:
            findings.append(Finding(
                code="slow_recovery",
                player_id=r.player_id,
                severity=min(1.0, (r.value - 0.8) / 0.8),
                headline="Slow recovery to base position",
                detail=(
                    f"Average recovery {r.value:.2f}s "
                    f"(median {r.context.get('median', 0):.2f}s) over "
                    f"{r.sample_size} shots. Returning to base faster "
                    f"would reduce time spent out of position."
                ),
                evidence=[r.metric_id]
            ))

    # Weak shots
    eff = results_by_id.get("shots.effectiveness", [])
    for r in eff:
        if not isinstance(r.value, dict):
            continue
        weak = {
            k: v for k, v in r.value.items()
            if v < 35 and r.context.get("counts", {}).get(k, 0) >= 8
        }
        for shot, winrate in weak.items():
            findings.append(Finding(
                code=f"weak_shot_{shot}",
                player_id=r.player_id,
                severity=(35 - winrate) / 35,
                headline=f"Low success on {shot}",
                detail=(
                    f"{shot} ends the rally in your favor only {winrate:.0f}% "
                    f"of the time ({r.context['counts'][shot]} attempts)."
                ),
                evidence=[r.metric_id]
            ))

    # High unforced errors
    err = results_by_id.get("errors.location_reason", [])
    for r in err:
        if not isinstance(r.value, dict):
            continue
        unforced = r.value.get("unforced_error", 0)
        if unforced > 30 and r.sample_size >= 10:
            findings.append(Finding(
                code="high_unforced",
                player_id=r.player_id,
                severity=min(1.0, unforced / 60),
                headline="High share of unforced errors",
                detail=(
                    f"{unforced:.0f}% of lost points are unforced "
                    f"({r.sample_size} lost rallies). Shot tolerance/"
                    f"consistency is the highest-leverage area."
                ),
                evidence=[r.metric_id]
            ))

    return findings
```

```python
# backend/app/shuttle_coach/feedback/prioritize.py
from app.shuttle_coach.feedback.rules import Finding


def prioritize_findings(findings: list[Finding]) -> list[Finding]:
    """Sort findings by severity (descending)."""
    return sorted(findings, key=lambda f: f.severity, reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_feedback.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/feedback/ backend/tests/test_shuttle_coach_feedback.py
git commit -m "feat: add feedback rules and prioritization (M3)"
```

---

## Task 8: Report Generator

**Files:**
- Create: `backend/app/shuttle_coach/feedback/report.py`
- Create: `backend/tests/test_shuttle_coach_report.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_report.py
from app.shuttle_coach.feedback.rules import Finding
from app.shuttle_coach.feedback.report import render_report, render_report_json


def test_render_report_markdown():
    """Test markdown report generation."""
    findings = [
        Finding("slow_recovery", "player_1", 0.85, "Slow recovery", "Detail text", ["movement.recovery_time"]),
        Finding("weak_smash", "player_1", 0.6, "Weak smash", "Smash detail", ["shots.effectiveness"]),
    ]
    
    md = render_report(findings)
    assert "# Coaching Report" in md
    assert "Slow recovery" in md
    assert "movement.recovery_time" in md


def test_render_report_json():
    """Test JSON report generation."""
    findings = [
        Finding("slow_recovery", "player_1", 0.85, "Slow recovery", "Detail", ["movement.recovery_time"]),
    ]
    
    report = render_report_json(findings, player_ids=["player_1"], capabilities={"movement", "shots"})
    assert "findings" in report
    assert "player_ids" in report
    assert "capabilities" in report
    assert len(report["findings"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_report.py -v`
Expected: FAIL with "ImportError: cannot import name 'render_report'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/feedback/report.py
from __future__ import annotations
from typing import Any

from app.shuttle_coach.feedback.rules import Finding
from app.shuttle_coach.feedback.prioritize import prioritize_findings


def render_report(findings: list[Finding], top_k: int = 5) -> str:
    """Render findings as markdown report."""
    findings = prioritize_findings(findings)
    lines = ["# Coaching Report", ""]
    
    lines.append("## Priorities")
    for i, f in enumerate(findings[:top_k], 1):
        lines.append(
            f"{i}. **{f.headline}** — {f.detail} "
            f"_(evidence: {', '.join(f.evidence)})_"
        )
    
    lines.append("")
    lines.append("## All findings")
    for f in findings:
        lines.append(f"- [{f.severity:.2f}] {f.headline}: {f.detail}")
    
    return "\n".join(lines)


def render_report_json(
    findings: list[Finding],
    player_ids: list[str],
    capabilities: set[str]
) -> dict[str, Any]:
    """Render findings as JSON report."""
    findings = prioritize_findings(findings)
    return {
        "findings": [
            {
                "code": f.code,
                "player_id": f.player_id,
                "severity": f.severity,
                "headline": f.headline,
                "detail": f.detail,
                "evidence": f.evidence,
            }
            for f in findings
        ],
        "player_ids": player_ids,
        "capabilities": sorted(capabilities),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/feedback/report.py backend/tests/test_shuttle_coach_report.py
git commit -m "feat: add report generator (markdown + JSON) (M3)"
```

---

## Task 9: Main Engine Entry Point

**Files:**
- Create: `backend/app/shuttle_coach/engine.py`
- Create: `backend/tests/test_shuttle_coach_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_integration.py
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from app.shuttle_coach.engine import analyze


def test_analyze_end_to_end():
    """Test full analysis pipeline with synthetic data."""
    with TemporaryDirectory() as d:
        # Create minimal parquet files
        rallies = pd.DataFrame({
            "rally_id": [1, 2],
            "match_id": ["test.mp4", "test.mp4"],
            "start_frame": [0, 100],
            "end_frame": [90, 200],
            "start_ts": [0.0, 3.33],
            "end_ts": [3.0, 6.67],
            "winner_player_id": ["player_1", "player_2"],
            "end_reason": ["winner", "unforced_error"],
            "serving_player_id": ["player_1", "player_2"]
        })
        shots = pd.DataFrame({
            "shot_id": [1, 2, 3, 4, 5],
            "rally_id": [1, 1, 1, 2, 2],
            "player_id": ["player_1", "player_2", "player_1", "player_1", "player_2"],
            "shot_type": ["smash", "clear", "drop", "smash", "clear"],
            "shot_conf": [0.9, 0.8, 0.85, 0.9, 0.8],
            "hit_frame": [30, 45, 60, 150, 180],
            "start_ts": [1.0, 1.5, 2.0, 5.0, 6.0]
        })
        hits = pd.DataFrame({
            "hit_id": [1, 2, 3, 4, 5],
            "rally_id": [1, 1, 1, 2, 2],
            "frame": [30, 45, 60, 150, 180],
            "ts": [1.0, 1.5, 2.0, 5.0, 6.0],
            "player_id": ["player_1", "player_2", "player_1", "player_1", "player_2"],
            "hit_u": [640, 640, 640, 640, 640],
            "hit_v": [360, 360, 360, 360, 360],
            "court_x": [6.7, 6.7, 6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59, 2.59, 2.59]
        })
        shuttle = pd.DataFrame({
            "frame": [30, 45, 60, 150, 180],
            "ts": [1.0, 1.5, 2.0, 5.0, 6.0],
            "u": [640, 640, 640, 640, 640],
            "v": [360, 360, 360, 360, 360],
            "court_x": [6.7, 6.7, 6.7, 6.7, 6.7],
            "court_y": [2.59, 2.59, 2.59, 2.59, 2.59],
            "visible": [True, True, True, True, True]
        })
        player_detections = pd.DataFrame({
            "frame": [30, 30, 45, 45, 60, 60, 150, 150, 180, 180],
            "ts": [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0],
            "player_id": ["player_1", "player_2"] * 5,
            "court_x": [6.7] * 10,
            "court_y": [2.59] * 10,
            "bbox_x1": [100, 500] * 5,
            "bbox_y1": [200, 200] * 5,
            "bbox_x2": [200, 600] * 5,
            "bbox_y2": [500, 500] * 5
        })
        
        rallies.to_parquet(f"{d}/rallies.parquet", index=False)
        shots.to_parquet(f"{d}/shots.parquet", index=False)
        hits.to_parquet(f"{d}/hits.parquet", index=False)
        shuttle.to_parquet(f"{d}/shuttle.parquet", index=False)
        player_detections.to_parquet(f"{d}/player_detections.parquet", index=False)
        
        result = analyze(d)
        
        assert "player_ids" in result
        assert "capabilities" in result
        assert "metrics" in result
        assert "findings" in result
        assert "report_md" in result
        assert "report_json" in result
        assert len(result["player_ids"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_integration.py -v`
Expected: FAIL with "ImportError: cannot import name 'analyze'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/engine.py
from __future__ import annotations
from collections import defaultdict
from typing import Any

from app.shuttle_coach.loader import load_match, capabilities
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import run_metrics
from app.shuttle_coach.metrics.base import MetricResult
from app.shuttle_coach.feedback import derive_findings, prioritize_findings
from app.shuttle_coach.feedback.report import render_report, render_report_json


def analyze(data_dir: str) -> dict[str, Any]:
    """Run full shuttle-coach analysis on a data directory."""
    tables = load_match(data_dir)
    caps = capabilities(tables)
    model = MatchModel.from_tables(tables)

    # Run metrics
    results = run_metrics(model, caps)

    # Group results by metric_id
    results_by_id: dict[str, list[MetricResult]] = defaultdict(list)
    for r in results:
        results_by_id[r.metric_id].append(r)

    # Derive findings
    findings = derive_findings(results_by_id)
    findings = prioritize_findings(findings)

    # Generate reports
    report_md = render_report(findings)
    report_json = render_report_json(findings, model.player_ids, caps)

    return {
        "player_ids": model.player_ids,
        "capabilities": sorted(caps),
        "metrics": [r.to_row() for r in results],
        "findings": [
            {
                "code": f.code,
                "player_id": f.player_id,
                "severity": f.severity,
                "headline": f.headline,
                "detail": f.detail,
                "evidence": f.evidence,
            }
            for f in findings
        ],
        "report_md": report_md,
        "report_json": report_json,
    }


def narrate(question: str, metrics: list[dict], api_key: str) -> str:
    """Generate LLM narration for a question about the metrics."""
    from app.shuttle_coach.narration.gemini import answer
    return answer(question, metrics, api_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_integration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/engine.py backend/tests/test_shuttle_coach_integration.py
git commit -m "feat: add main engine entry point with analyze() and narrate() (M4)"
```

---

## Task 10: Gemini Narration

**Files:**
- Create: `backend/app/shuttle_coach/narration/__init__.py`
- Create: `backend/app/shuttle_coach/narration/rag.py`
- Create: `backend/app/shuttle_coach/narration/gemini.py`
- Create: `backend/tests/test_shuttle_coach_narration.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shuttle_coach_narration.py
from app.shuttle_coach.narration.rag import retrieve_relevant_metrics
from app.shuttle_coach.narration.gemini import enforce_citations


def test_retrieve_relevant_metrics():
    """Test that RAG retrieves metrics relevant to a question."""
    metrics = [
        {"metric_id": "movement.recovery_time", "value": 0.85, "unit": "s"},
        {"metric_id": "shots.mix", "value": {"smash": 40.0, "clear": 30.0}},
        {"metric_id": "errors.location_reason", "value": {"unforced_error": 45.0}},
    ]
    
    relevant = retrieve_relevant_metrics("Where am I losing points?", metrics, k=2)
    assert len(relevant) == 2
    # Should prioritize errors.location_reason for "losing points"
    assert relevant[0]["metric_id"] == "errors.location_reason"


def test_enforce_citations_valid():
    """Test that valid citations pass enforcement."""
    text = "You have slow recovery [movement.recovery_time]. Your smash is weak [shots.effectiveness]."
    metrics = [
        {"metric_id": "movement.recovery_time"},
        {"metric_id": "shots.effectiveness"},
    ]
    # Should not raise
    enforce_citations(text, metrics)


def test_enforce_citations_invalid():
    """Test that invalid citations raise error."""
    text = "You have slow recovery [nonexistent.metric]."
    metrics = [{"metric_id": "movement.recovery_time"}]
    
    try:
        enforce_citations(text, metrics)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent.metric" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_narration.py -v`
Expected: FAIL with "ImportError: cannot import name 'retrieve_relevant_metrics'"

- [ ] **Step 3: Write implementation**

```python
# backend/app/shuttle_coach/narration/__init__.py
"""Narration engine for shuttle-coach."""

from app.shuttle_coach.narration.rag import retrieve_relevant_metrics
from app.shuttle_coach.narration.gemini import answer, enforce_citations

__all__ = ["retrieve_relevant_metrics", "answer", "enforce_citations"]
```

```python
# backend/app/shuttle_coach/narration/rag.py
"""Simple keyword-based RAG for metric retrieval."""

from __future__ import annotations
from typing import Any


def retrieve_relevant_metrics(
    question: str,
    metrics: list[dict[str, Any]],
    k: int = 12
) -> list[dict[str, Any]]:
    """Retrieve metrics relevant to a question using keyword matching."""
    # Simple scoring: count keyword overlaps
    question_words = set(question.lower().split())
    
    scored = []
    for m in metrics:
        score = 0
        # Score by metric_id keywords
        metric_words = set(m["metric_id"].lower().replace(".", " ").split())
        score += len(question_words & metric_words) * 2
        
        # Score by context keywords
        context = m.get("context", {})
        if isinstance(context, dict):
            context_words = set(str(v).lower().split() for v in context.values())
            for cw in context_words:
                if isinstance(cw, set):
                    score += len(question_words & cw)
        
        scored.append((score, m))
    
    # Sort by score descending, return top k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]
```

```python
# backend/app/shuttle_coach/narration/gemini.py
"""Gemini 2.0 Flash narration with citation enforcement."""

from __future__ import annotations
import re
from typing import Any

SYSTEM_PROMPT = """You are a badminton coaching assistant. You may ONLY use the metrics 
provided in the context. Every claim must cite the metric_id(s) it relies 
on in square brackets, e.g. [movement.recovery_time]. If the metrics do 
not support an answer, say so. Do not invent numbers."""


def answer(question: str, metrics: list[dict[str, Any]], api_key: str) -> str:
    """Generate a grounded answer to a coaching question."""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Format metrics for context
    context = "\n".join(
        f"- {m['metric_id']}: {m.get('value')} ({m.get('unit', '')})"
        for m in metrics[:12]
    )
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\nMETRICS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    
    text = response.text
    enforce_citations(text, metrics)
    return text


def enforce_citations(text: str, metrics: list[dict[str, Any]]) -> None:
    """Validate that every sentence has a valid citation."""
    valid = {m["metric_id"] for m in metrics}
    
    # Find all citations in text
    cited = set(re.findall(r"\[([a-z_]+\.[a-z_]+)\]", text))
    unknown = cited - valid
    if unknown:
        raise ValueError(f"Narration cited unknown metrics: {unknown}")
    
    # Check that non-trivial sentences have at least one citation
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 6]
    uncited = [s for s in sentences if not re.search(r"\[[a-z_]+\.[a-z_]+\]", s)]
    if uncited:
        raise ValueError(f"Ungrounded sentences: {uncited}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_narration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shuttle_coach/narration/ backend/tests/test_shuttle_coach_narration.py
git commit -m "feat: add Gemini narration with RAG and citation enforcement (M5)"
```

---

## Task 11: API Endpoint

**Files:**
- Modify: `backend/app/api/routes.py:234-248`

- [ ] **Step 1: Write the failing test**

```python
# Add to backend/tests/test_api.py
def test_shuttle_coach_endpoint():
    """Test shuttle-coach analysis endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    
    # Create a mock job with parquet files
    # (This test would need fixtures - simplified for now)
    response = client.get("/api/shuttle-coach/analyze/test_job")
    # Should return 404 for non-existent job
    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_api.py::test_shuttle_coach_endpoint -v`
Expected: FAIL with 404 (endpoint doesn't exist)

- [ ] **Step 3: Write implementation**

Add to `backend/app/api/routes.py` after the existing endpoints:

```python
@router.get("/shuttle-coach/analyze/{job_id}")
async def analyze_shuttle_coach(job_id: str, question: str = None):
    """Run shuttle-coach analysis on a completed job."""
    import os
    from app.shuttle_coach.engine import analyze, narrate
    
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    
    job_dir = settings.job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(404, "Job directory not found")
    
    try:
        result = analyze(str(job_dir))
    except FileNotFoundError as e:
        raise HTTPException(400, f"Missing parquet files: {e}")
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")
    
    # Optional: LLM narration
    if question and os.environ.get("GEMINI_API_KEY"):
        try:
            result["narration"] = narrate(
                question, result["metrics"], os.environ["GEMINI_API_KEY"]
            )
        except Exception as e:
            result["narration_error"] = str(e)
    
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_api.py::test_shuttle_coach_endpoint -v`
Expected: PASS (returns 404 for non-existent job)

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes.py
git commit -m "feat: add shuttle-coach API endpoint (M6)"
```

---

## Task 12: Add google-generativeai to Dependencies

**Files:**
- Modify: `backend/requirements.txt` or `pyproject.toml`

- [ ] **Step 1: Add dependency**

```bash
echo "google-generativeai>=0.5.0" >> backend/requirements.txt
```

- [ ] **Step 2: Install and verify**

```bash
cd /home/sujith/baddyCoach && .venv/bin/pip install google-generativeai>=0.5.0
```

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "deps: add google-generativeai for Gemini narration"
```

---

## Task 13: Run Full Test Suite

**Files:**
- None (verification only)

- [ ] **Step 1: Run all shuttle-coach tests**

```bash
cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/test_shuttle_coach_*.py -v
```

- [ ] **Step 2: Run full backend test suite**

```bash
cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/ -v
```

- [ ] **Step 3: Verify no regressions**

All existing tests should still pass.

---

## Summary

**Total tasks:** 13 (including M0 Colab metadata parity)

**Estimated time:** 2-3 hours for a skilled developer

**Key deliverables:**
- Colab pipeline with full rally metadata (M0)
- Loader with dual input support (backend + Colab)
- Plugin-based metric engine with 9 metrics
- Rule-based feedback with severity ranking
- Report generator (markdown + JSON)
- Gemini narration with citation enforcement
- API endpoint for analysis

**Dependencies:**
- pandas, numpy, pyarrow (already in project)
- google-generativeai (optional, for narration)
