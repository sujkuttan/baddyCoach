# Shuttle-Coach: Coaching Insights Engine — Design Spec

**Date:** 2026-06-20
**Status:** Approved
**Source document:** `docs/badminton_coaching_feedback_prd_techdesign.md`

---

## 1. Overview

Shuttle-coach is an embedded Python library within the BMCA backend that reads raw parquet outputs from the ML pipeline and produces grounded, intelligent coaching feedback. It computes its own metrics from scratch (ignoring existing analytics stages), generates rule-based findings, and optionally produces natural-language narration via Gemini 2.0 Flash.

**Design principle (from PRD):** "The intelligence lives in the analytics layer, not the language layer." An LLM handed raw coordinates invents tactics; an LLM handed "average recovery time 0.8s vs opponent 0.5s" produces grounded, useful feedback.

## 2. Integration Approach

**Chosen:** Backend library (embedded in existing FastAPI codebase)

- `backend/app/shuttle_coach/` as a Python package
- Reads parquet from `ArtifactStore` (backend) or `debug/` directory (Colab)
- Exposed via new API endpoint (`/api/v1/shuttle-coach/analyze`)
- Existing rule-based engine (`engine.py`) kept as fallback

**Dual input support:**
- **Backend mode:** `shuttle_coach.analyze(job_dir)` — reads `shots.parquet`, `rallies.parquet`, etc. from job root
- **Colab mode:** `shuttle_coach.analyze(colab_debug_dir)` — reads from `debug/` subdirectory
- **Auto-detection:** if files at root → backend mode; if files in `debug/` → Colab mode

## 3. Module Structure

```
backend/app/shuttle_coach/
├── __init__.py
├── loader.py          # Read parquet, validate schema, capability detection
├── events.py          # MatchModel dataclass
├── metrics/
│   ├── __init__.py
│   ├── base.py        # Metric ABC, MetricResult, registry
│   ├── movement.py    # RecoveryTime, CourtCoverage, DistancePerRally
│   ├── shots.py       # ShotMix, ShotEffectiveness
│   ├── tactical.py    # Placement, RallyConstruction
│   ├── errors.py      # ErrorLocation
│   └── technique.py   # PreparationConsistency (optional, requires pose)
├── feedback/
│   ├── __init__.py
│   ├── rules.py       # Threshold → Finding mapping
│   ├── prioritize.py  # Rank findings by severity
│   └── report.py      # Render markdown/JSON report
├── narration/
│   ├── __init__.py
│   ├── rag.py         # Build retrieval index over metrics
│   └── gemini.py      # Gemini 2.0 Flash integration
└── engine.py          # Main entry point
```

## 4. Data Flow

```
Input: job_dir (parquet files at root) OR colab_dir (parquet in debug/)
    ↓
loader.py → validate schema, detect capabilities
    ↓
events.py → MatchModel (in-memory representation)
    ↓
metrics/ → List[MetricResult] (deterministic, no network)
    ↓
feedback/rules.py → List[Finding] (severity-ranked)
    ↓
feedback/report.py → report.md + report.json
    ↓
narration/gemini.py → natural language coaching (optional, requires API key)
```

## 5. Parquet Schema

### rallies.parquet
| column | type | notes | source |
|--------|------|-------|--------|
| `rally_id` | int | unique | both |
| `start_frame`, `end_frame` | int | frame range | both |
| `shot_count` | int | number of shots in rally | colab |
| `match_id` | str | | backend |
| `start_ts`, `end_ts` | float | seconds | backend |
| `winner_player_id` | str\|null | who won the rally | backend |
| `end_reason` | str\|null | e.g. winner / forced_error / unforced_error / out / net | backend |
| `serving_player_id` | str\|null | | backend |

**Note:** Colab rallies only have `rally_id`, `start_frame`, `end_frame`, `shot_count`. Backend rallies have full metadata. Loader handles both.

### shots.parquet
| column | type | notes |
|--------|------|-------|
| `shot_id` | int | unique |
| `rally_id` | int | FK |
| `player_id` | str | who hit it ("player_1" or "player_2") |
| `stroke_type` | str | BST class (smash, clear, drop, net, drive, lift, serve, …) |
| `stroke_confidence` | float | BST confidence |
| `frame` | int | FK to hits (note: Colab uses `frame`, backend uses `hit_frame`) |
| `start_ts` | float | |

**Note:** Column names vary between Colab (`stroke_type`, `frame`) and backend (`shot_type`, `hit_frame`). Loader must handle both conventions.

### hits.parquet
| column | type | notes |
|--------|------|-------|
| `hit_id` | int | |
| `rally_id` | int | |
| `frame` | int | |
| `ts` | float | |
| `player_id` | int\|null | hitting player |
| `hit_u`, `hit_v` | float | image-space contact point |
| `court_x`, `court_y` | float\|null | contact point in court metres |

### shuttle.parquet
| column | type | notes |
|--------|------|-------|
| `frame` | int | |
| `ts` | float | |
| `u`, `v` | float\|null | image position |
| `court_x`, `court_y` | float\|null | court metres (if homography applied) |
| `visible` | bool | TrackNet visibility |

### player_detections.parquet
| column | type | notes |
|--------|------|-------|
| `frame` | int | |
| `ts` | float | |
| `player_id` | int | |
| `court_x`, `court_y` | float\|null | court metres (foot point) |
| `bbox_x1..y2` | float | image box |

### pose.parquet (optional)
| column | type | notes |
|--------|------|-------|
| `frame` | int | |
| `player_id` | int | |
| `kpt_{i}_x`, `kpt_{i}_y`, `kpt_{i}_conf` | float | i = 0..16 (COCO-17) |

## 6. Loader + Capability Detection

```python
# shuttle_coach/loader.py
# Column name mappings (Colab → canonical)
COLUMN_ALIASES = {
    "shots": {
        "stroke_type": "shot_type",
        "stroke_confidence": "shot_conf",
        "frame": "hit_frame",  # Colab uses 'frame', backend uses 'hit_frame'
    },
    "rallies": {
        "shot_count": None,  # Colab-only, ignored
    },
}

# Required columns per table (at least one variant must exist)
REQUIRED = {
    "rallies": ["rally_id"],  # Minimal: just rally_id
    "shots":   ["rally_id", "player_id"],  # Minimal: rally + player
    "hits":    ["rally_id", "frame"],
    "shuttle": ["frame"],
    "player_detections": ["frame", "player_id"],
    "pose":    ["frame", "player_id"],
}
OPTIONAL_TABLES = {"pose"}

def load_match(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load parquet files from backend job dir OR Colab debug/ dir."""
    d = pathlib.Path(data_dir)
    
    # Auto-detect: if files in debug/ subdirectory, use that
    if (d / "debug").is_dir() and any((d / "debug" / f"{name}.parquet").exists() 
                                       for name in REQUIRED):
        d = d / "debug"
    
    tables = {}
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
        
        # Validate required columns (check both original and aliased)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.parquet missing columns: {missing}")
        tables[name] = df
    return tables

def capabilities(tables: dict[str, pd.DataFrame]) -> set[str]:
    """Which metric families are computable given present data."""
    caps = {"shots", "errors"}
    has_court = lambda t: t in tables and {"court_x", "court_y"}.issubset(tables[t].columns)
    if has_court("player_detections"):
        caps.add("movement")
    if has_court("shuttle") or has_court("hits"):
        caps.add("tactical")
    if "pose" in tables:
        caps.add("technique")
    return caps
```

## 7. Event Model

**Player ID convention:** Both Colab and backend use string player IDs (`"player_1"`, `"player_2"`). The MatchModel uses `str` for player_id throughout.

```python
# shuttle_coach/events.py
@dataclass
class MatchModel:
    match_id: str
    rallies: pd.DataFrame
    shots: pd.DataFrame
    hits: pd.DataFrame
    shuttle: pd.DataFrame
    positions: pd.DataFrame          # player_detections
    pose: pd.DataFrame | None
    player_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame]) -> "MatchModel":
        rallies = tables["rallies"]
        shots = tables["shots"]
        pids = sorted(set(shots["player_id"].dropna().astype(str)))
        return cls(
            match_id=str(rallies["match_id"].iloc[0]) if "match_id" in rallies.columns else "unknown",
            rallies=rallies, shots=shots, hits=tables["hits"],
            shuttle=tables["shuttle"], positions=tables["player_detections"],
            pose=tables.get("pose"), player_ids=pids,
        )

    def shots_of(self, player_id: str) -> pd.DataFrame:
        return self.shots[self.shots["player_id"] == player_id]

    def positions_of(self, player_id: str) -> pd.DataFrame:
        return self.positions[self.positions["player_id"] == player_id]
```

## 8. Metric Engine

### Base Classes
```python
# shuttle_coach/metrics/base.py
REGISTRY: list[type["Metric"]] = []

def register(cls):
    REGISTRY.append(cls)
    return cls

@dataclass
class MetricResult:
    metric_id: str
    player_id: int | None
    value: float | dict
    unit: str
    sample_size: int
    confidence: float
    context: dict[str, Any]

class Metric:
    metric_id: str = "base"
    requires: set[str] = set()
    def applicable(self, caps: set[str]) -> bool:
        return self.requires.issubset(caps)
    def compute(self, m: MatchModel) -> list[MetricResult]:
        raise NotImplementedError
```

### Metric Catalogue

| metric_id | family | needs | computes |
|-----------|--------|-------|----------|
| `movement.recovery_time` | movement | positions(court) | Time to return to base (1.0m threshold) after each shot |
| `movement.court_coverage` | movement | positions(court) | Zone histogram: front/mid/rear × left/right (6 zones) |
| `movement.distance_per_rally` | movement | positions(court) | Total distance traveled per rally |
| `shots.mix` | shots | shots | Shot type distribution (%) per player |
| `shots.effectiveness` | shots | shots+rallies | Win rate conditioned on shot type |
| `tactical.placement` | tactical | hits/shuttle(court) | Depth & width control metrics |
| `tactical.rally_construction` | tactical | shots(seq) | Shot sequences preceding winners |
| `errors.location_reason` | errors | rallies | Where/how points are lost (unforced vs forced) |
| `technique.preparation_consistency` | technique | pose | Body posture variability at hit frames |

### Key Metrics Implementation

**Recovery Time:**
```python
@register
class RecoveryTime(Metric):
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
                # Use 'start_ts' if available, else compute from frame
                ts = s.get("start_ts") or (s["frame"] / 30.0)  # fallback: frame/30
                after = pos[pos["ts"] >= ts].head(60)
                if after.empty:
                    continue
                d = np.linalg.norm(after[["court_x", "court_y"]].to_numpy() - base, axis=1)
                back = np.argmax(d < 1.0) if (d < 1.0).any() else len(d) - 1
                recov.append(after["ts"].iloc[back] - ts)
            if recov:
                out.append(MetricResult(
                    self.metric_id, pid, float(np.mean(recov)), "s",
                    sample_size=len(recov), confidence=min(1.0, len(recov) / 30),
                    context={"median": float(np.median(recov)),
                             "base_xy": base.round(2).tolist()}))
        return out
```

**Shot Effectiveness:**
```python
@register
class ShotEffectiveness(Metric):
    metric_id = "shots.effectiveness"
    requires = {"shots"}

    def compute(self, m):
        out = []
        # Handle missing winner_player_id (Colab rallies don't have it)
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
            out.append(MetricResult(self.metric_id, pid, eff, "%",
                       sample_size=len(s), confidence=1.0,
                       context={"counts": counts}))
        return out
```

## 9. Feedback System

### Finding Dataclass
```python
@dataclass
class Finding:
    code: str
    player_id: int | None
    severity: float            # 0..1
    headline: str
    detail: str
    evidence: list[str]        # metric_ids
```

### Rule Thresholds

| rule | condition | severity calc |
|------|-----------|---------------|
| `slow_recovery` | `movement.recovery_time > 0.8s` | `min(1.0, (value - 0.8) / 0.8)` |
| `weak_shot_{type}` | `shots.effectiveness[type] < 35%` | `(35 - winrate) / 35` |
| `high_unforced` | `errors.location_reason[unforced] > 30%` | `min(1.0, unforced / 60)` |
| `unbalanced_court` | `movement.court_coverage[rear] > 60%` | `(rear_pct - 60) / 40` |
| `low_variety` | `shots.mix[max] > 45%` | `(max_pct - 45) / 55` |

**Minimum sample size:** Rules require ≥10-15 events before firing (varies by rule).

### Prioritization
Findings sorted by `severity` (descending). Top 5 presented as priorities, rest as "All findings".

### Report Output

**report.md:**
```markdown
# Coaching Report

## Priorities
1. **Slow recovery to base position** — Average recovery 0.92s (median 0.85s) over 24 shots. Returning to base faster would reduce time spent out of position. _(evidence: movement.recovery_time)_
2. **Low success on smash** — Smash ends the rally in your favor only 28% of the time (12 attempts). _(evidence: shots.effectiveness)_

## All findings
- [0.85] Slow recovery to base position: ...
- [0.62] Low success on smash: ...
```

**report.json:**
```json
{
  "findings": [...],
  "metrics": [...],
  "player_ids": [1, 2],
  "capabilities": ["movement", "shots", "errors"]
}
```

## 10. LLM Narration (Gemini 2.0 Flash)

### System Prompt
```
You are a badminton coaching assistant. You may ONLY use the metrics 
provided in the context. Every claim must cite the metric_id(s) it relies 
on in square brackets, e.g. [movement.recovery_time]. If the metrics do 
not support an answer, say so. Do not invent numbers.
```

### Implementation
```python
# narration/gemini.py
import google.generativeai as genai

def answer(question: str, metrics: list[MetricResult], api_key: str) -> str:
    genai.configure(api_key=api_key)
    context = format_metrics_for_rag(metrics, question)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\nMETRICS:\n{context}\n\nQUESTION: {question}"
    )
    enforce_citations(response.text, metrics)
    return response.text
```

### Citation Enforcement
- Every sentence >6 words must contain at least one `[metric_id]` citation
- Reject sentences with unknown metric references
- Raise `ValueError` if ungrounded content detected

### RAG (Simple Keyword Matcher)
- Build index from metric_id tags + context keys
- Retrieve top-12 metrics relevant to the question
- Upgrade to embeddings only if needed

### Example Interaction
```
Q: "Where am I losing most of my points?"
A: "You're losing most points in the rear court zone [movement.court_coverage]. 
    45% of your lost rallies end with unforced errors [errors.location_reason], 
    particularly on backhand clears [shots.effectiveness]."
```

## 11. API Integration

### Dependencies
```
google-generativeai>=0.5.0  # Gemini API client (optional, for narration)
```

### New Endpoint
```python
# backend/app/api/routes.py
import os
from pathlib import Path
from app.shuttle_coach import engine as shuttle_coach

@router.post("/shuttle-coach/analyze")
async def analyze_shuttle_coach(job_id: str, question: str = None):
    job_dir = Path(f"data/jobs/{job_id}")
    result = shuttle_coach.analyze(str(job_dir))
    
    # Optional: LLM narration via Gemini
    if question and os.environ.get("GEMINI_API_KEY"):
        result["narration"] = shuttle_coach.narrate(
            question, result["metrics"], os.environ["GEMINI_API_KEY"]
        )
    
    return result
```

### Response Format
```json
{
  "job_id": "abc123",
  "player_ids": ["player_1", "player_2"],
  "capabilities": ["movement", "shots", "errors", "technique"],
  "metrics": [
    {
      "metric_id": "movement.recovery_time",
      "player_id": "player_1",
      "value": 0.85,
      "unit": "s",
      "sample_size": 24,
      "confidence": 0.8
    }
  ],
  "findings": [
    {
      "code": "slow_recovery",
      "player_id": "player_1",
      "severity": 0.62,
      "headline": "Slow recovery to base position",
      "detail": "Average recovery 0.85s over 24 shots...",
      "evidence": ["movement.recovery_time"]
    }
  ],
  "report_md": "# Coaching Report\n...",
  "report_json": {...},
  "narration": "Optional LLM response..."
}
```

## 12. Testing Strategy

- **Unit:** Each metric against synthetic `MatchModel` with known answer
- **Property:** Metrics never emit result with `sample_size == 0`
- **Groundedness:** Automated check that every finding's `evidence` IDs exist in metrics store
- **Integration:** Full pipeline test with sample parquet files from Colab output
- **Graceful degradation:** Metrics requiring missing data (e.g., `winner_player_id`) return empty results, not errors

## 13. Build Order (Milestones)

0. **M0** — Colab pipeline metadata parity: Add full rally metadata to Colab `stage_rallies()` so rallies.parquet matches backend schema (add `match_id`, `start_ts`, `end_ts`, `winner_player_id`, `end_reason`, `serving_player_id`). Also ensure shots.parquet has `shot_id` and `start_ts` columns. This eliminates schema divergence between Colab and backend.
1. **M1** — Loader + capability detection + event model; round-trip real parquet folder
2. **M2** — Movement + shots + errors metrics; metrics store output
3. **M3** — Rule-based findings + prioritized report.md
4. **M4** — Tactical + technique (pose) metrics behind capability flags
5. **M5** — Gemini narration with citation enforcement
6. **M6** — API endpoint + frontend integration

## 14. Colab Metadata Parity (M0 Detail)

The Colab pipeline currently produces minimal rally metadata. To achieve schema parity with the backend, `stage_rallies()` must be updated to include:

### Current Colab rallies.parquet
```python
{"rally_id": 1, "start_frame": 100, "end_frame": 450, "shot_count": 12}
```

### Target (matching backend)
```python
{
    "rally_id": 1,
    "match_id": video_name,           # from video filename
    "start_frame": 100,
    "end_frame": 450,
    "start_ts": 3.33,                 # start_frame / fps
    "end_ts": 15.0,                   # end_frame / fps
    "shot_count": 12,
    "winner_player_id": "player_1",   # inferred from last shot attribution
    "end_reason": "unforced_error",   # inferred from shot pattern
    "serving_player_id": "player_2"   # first server (alternates each rally)
}
```

### Changes required in `colab/pipeline.py`

1. **`stage_rallies()`** — Accept `fps` and `video_name` parameters. Compute `start_ts`/`end_ts` from frame numbers. Return full metadata dict.

2. **Winner inference** — After attribution (stage 8), determine rally winner by who hit the last shot. Add `winner_player_id` to rally dict.

3. **End reason heuristic** — Infer from last shot type: `net` for net shots, `out` for clears/drives that went long, `unforced_error` for low-confidence shots, `winner` for smashes/drops with high confidence.

4. **First server** — Player 1 serves first, alternates each rally. Set `serving_player_id` accordingly.

5. **Shots metadata** — Add `shot_id` (sequential) and `start_ts` (frame / fps) to each shot dict in `stage_strokes()`.

### Backend parity check
After M0, verify that both Colab and backend produce identical schemas for:
- `rallies.parquet` — all columns present
- `shots.parquet` — `shot_id`, `stroke_type`/`shot_type`, `frame`/`hit_frame`, `start_ts`
- `hits.parquet` — `hit_id`, `court_x`, `court_y`
- `player_detections.parquet` — `court_x`, `court_y`
- `shuttle.parquet` — `court_x`, `court_y`, `visible`

## 15. Open Questions

- **Court coordinates:** Are they in metres (0-13.4 x 0-5.18) or normalised (0-1)? Zone edges in `court_coverage` must match.
- **Ground-truth labelled match:** Needed for golden-test fixture and validating shot-effectiveness logic.
- **Column name variations:** Backend uses `shot_type`/`hit_frame`, Colab uses `stroke_type`/`frame`. Loader handles both via aliases. After M0, both should use consistent names.
