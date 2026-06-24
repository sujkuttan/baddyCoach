# Badminton Coaching Feedback Layer — PRD + Technical Design

**Component name (working):** `shuttle-coach`
**One-liner:** A standalone Python application that reads the pipeline's parquet outputs (shots, rallies, player_detections, pose, hits, shuttle) and produces grounded, intelligent coaching feedback — first as deterministic analytics, then as natural-language narration over those analytics.

> This is a sibling component to the court-position extraction design. It assumes that upstream detection/classification has already run and persisted its results as parquet. It is a normal Python application (CLI + library), **not** a notebook/Colab build.

---

## Part 1 — Product Requirements (PRD)

### 1.1 Problem

Detection and classification models (TrackNetV3 for the shuttle, BST for shot type, SoloShuttlePose for court positions and pose) emit large volumes of low-level, per-frame and per-event data. None of it is coaching advice. A coach or player cannot look at a parquet file of shuttle coordinates and shot labels and know *what to work on*. We need a layer that converts these structured detections into interpretable performance metrics, and then into clear, trustworthy, actionable feedback.

### 1.2 Goal

Given a directory of parquet files for one or more matches, produce: (a) a structured analytics report of per-player and per-rally metrics, and (b) natural-language coaching feedback grounded strictly in those metrics, with every statement traceable back to a number.

### 1.3 Design principle (the load-bearing one)

**The intelligence lives in the analytics layer, not the language layer.** An LLM handed raw coordinates invents tactics; an LLM handed "average recovery time 0.8 s vs opponent 0.5 s; 70% of points lost in the backhand rear court" produces grounded, useful feedback. So the bulk of engineering effort goes into computing correct, interpretable metrics. The narration layer only *verbalizes* metrics it is given — it never sees raw detections and never computes its own statistics.

### 1.4 Non-goals

- No detection/tracking/classification — strictly consumes upstream parquet.
- No real-time/live feedback — offline batch over completed-match data.
- No medical/injury or strength-and-conditioning prescription.
- No claims requiring 3D (shuttle height, true racket-head speed) unless those fields are present in the input.

### 1.5 Users / consumers

- **Player / coach:** wants a readable report and the ability to ask questions.
- **Analyst / developer:** wants the structured metrics table to build dashboards or feed other tools.

### 1.6 User stories

1. As a coach, I point the app at a match's parquet folder and get a written report of the player's strengths, weaknesses, and 3–5 prioritized things to work on.
2. As a player, I ask "where am I losing most of my points?" and get an answer grounded in actual rally outcomes.
3. As an analyst, I get a metrics parquet/JSON I can load into my own dashboards.
4. As a developer, I can add a new metric without touching the narration layer.

### 1.7 Functional requirements

- **FR1** — Load and validate the input parquet set (shots, rallies, player_detections, pose, hits, shuttle).
- **FR2** — Join/align the tables into a coherent per-rally, per-shot event model.
- **FR3** — Compute a defined battery of metrics (movement, shot, tactical, error metrics) per player and per rally.
- **FR4** — Persist metrics as structured output (parquet + JSON).
- **FR5** — Generate a deterministic rule-based feedback report from thresholds.
- **FR6** — Optionally generate LLM narration grounded via RAG over the computed metrics, with citations to metric IDs.
- **FR7** — Run as a CLI (`shuttle-coach analyze <data_dir>`) and as an importable library.

### 1.8 Non-functional requirements

- **NFR1 — Groundedness:** every feedback statement references a metric; no statement without a supporting number. LLM output is constrained to the supplied metrics.
- **NFR2 — Determinism of analytics:** same parquet in → same metrics out (LLM phrasing may vary; numbers may not).
- **NFR3 — Robustness to missing data:** if `pose` is absent, technique metrics are skipped with a clear note, not faked.
- **NFR4 — Extensibility:** metrics are plugins; adding one shouldn't ripple.
- **NFR5 — No network dependency for the analytics path.** LLM narration is an optional, clearly isolated add-on.

### 1.9 Success metrics

- **Coverage:** ≥ 95% of rallies in the input contribute to at least one metric.
- **Groundedness audit:** 100% of generated feedback sentences map to a metric ID (automated check).
- **Coach usefulness (qualitative):** a domain coach rates the prioritized recommendations as actionable on a small review set.

### 1.10 Risks & mitigations

| Risk | Mitigation |
|------|------------|
| LLM hallucinates tactics not in the data | Narration sees *only* the metrics JSON; enforce metric-ID citations; validate every sentence maps to a metric |
| Upstream label noise (wrong shot type) propagates | Confidence-weight metrics; report data-quality flags; let low-confidence events be excluded |
| Missing tables (e.g. no pose) | Capability detection: compute only metrics whose inputs exist; declare skipped metrics |
| Court coordinates inconsistent with training convention | Validate court coordinate range on load; fail loudly |
| Over-confident advice from small sample | Attach sample sizes; suppress recommendations below a min-events threshold |

---

## Part 2 — Technical Design

### 2.1 Inputs — parquet schema (assumed)

The app consumes these tables. Exact columns should be reconciled with the upstream pipeline; below is the assumed contract. Unknown extra columns are ignored; missing required columns raise a clear validation error.

**`rallies.parquet`** — one row per rally
| column | type | notes |
|--------|------|-------|
| `rally_id` | int | unique |
| `match_id` | str | |
| `start_frame`, `end_frame` | int | |
| `start_ts`, `end_ts` | float | seconds |
| `winner_player_id` | int\|null | who won the rally |
| `end_reason` | str\|null | e.g. winner / forced_error / unforced_error / out / net |
| `serving_player_id` | int\|null | |

**`shots.parquet`** — one row per shot (BST output lives here)
| column | type | notes |
|--------|------|-------|
| `shot_id` | int | unique |
| `rally_id` | int | FK |
| `player_id` | int | who hit it |
| `shot_type` | str | BST class (smash, clear, drop, net, drive, lift, serve, …) |
| `shot_conf` | float | BST confidence |
| `hit_frame` | int | FK to hits |
| `start_ts` | float | |

**`hits.parquet`** — one row per detected hit/contact
| column | type | notes |
|--------|------|-------|
| `hit_id` | int | |
| `rally_id` | int | |
| `frame` | int | |
| `ts` | float | |
| `player_id` | int\|null | hitting player |
| `hit_u`, `hit_v` | float | image-space contact point |
| `court_x`, `court_y` | float\|null | contact point in court metres (if available) |

**`shuttle.parquet`** — per-frame shuttle track (TrackNetV3 output)
| column | type | notes |
|--------|------|-------|
| `frame` | int | |
| `ts` | float | |
| `u`, `v` | float\|null | image position |
| `court_x`, `court_y` | float\|null | court metres (if homography applied) |
| `visible` | bool | TrackNet visibility |

**`player_detections.parquet`** — per-frame per-player position
| column | type | notes |
|--------|------|-------|
| `frame` | int | |
| `ts` | float | |
| `player_id` | int | |
| `court_x`, `court_y` | float\|null | court metres (foot point) |
| `bbox_x1..y2` | float | image box |

**`pose.parquet`** — per-frame per-player keypoints (SoloShuttlePose / HRNet-style)
| column | type | notes |
|--------|------|-------|
| `frame` | int | |
| `player_id` | int | |
| `kpt_{i}_x`, `kpt_{i}_y`, `kpt_{i}_conf` | float | i = 0..16 (COCO-17) |

> Capability detection (NFR3): the app inspects which tables/columns are present and enables only the metrics whose inputs exist. Court-coordinate metrics require `court_x/court_y`; technique metrics require `pose`.

### 2.2 Architecture

```
data_dir/*.parquet
      │
      ▼
[A] Loader + validator ............ schema check, dtype coercion, FK integrity
      │
      ▼
[B] Event model builder ........... join rallies⇄shots⇄hits⇄shuttle⇄positions
      │                              into a typed in-memory model (one match)
      ▼
[C] Metric engine ................. registry of metric plugins, each consuming
      │                              the event model → MetricResult objects
      ▼
[D] Metrics store ................. metrics.parquet + metrics.json (per player,
      │                              per rally, per match) with sample sizes/conf
      ├───────────────► [E] Rule-based feedback (deterministic, no network)
      │                        thresholds → prioritized findings → report.md
      │
      └───────────────► [F] LLM narration (optional, RAG over metrics.json)
                               grounded answers + citations to metric IDs
```

### 2.3 Project layout (standalone Python app)

```
shuttle-coach/
├── pyproject.toml
├── shuttle_coach/
│   ├── __init__.py
│   ├── cli.py                 # entrypoint: argparse / typer
│   ├── io/
│   │   ├── loader.py          # read parquet, validate schema
│   │   └── schema.py          # column contracts + capability detection
│   ├── model/
│   │   └── events.py          # MatchModel, Rally, Shot, Track dataclasses
│   ├── metrics/
│   │   ├── base.py            # Metric ABC, MetricResult, registry
│   │   ├── movement.py        # coverage, recovery time, distance
│   │   ├── shots.py           # shot mix, effectiveness
│   │   ├── tactical.py        # placement, rally construction
│   │   ├── errors.py          # where/why points are lost
│   │   └── technique.py       # pose-based (optional)
│   ├── feedback/
│   │   ├── rules.py           # threshold → finding mapping
│   │   ├── prioritize.py      # rank findings by impact
│   │   └── report.py          # render report.md / report.json
│   └── narration/
│       ├── rag.py             # build retrieval index over metrics
│       └── llm.py             # grounded generation, citation enforcement
└── tests/
```

### 2.4 Loader + capability detection

```python
# shuttle_coach/io/loader.py
from __future__ import annotations
import pathlib
import pandas as pd

REQUIRED = {
    "rallies": ["rally_id", "match_id", "start_ts", "end_ts"],
    "shots":   ["shot_id", "rally_id", "player_id", "shot_type"],
    "hits":    ["hit_id", "rally_id", "frame", "ts"],
    "shuttle": ["frame", "ts"],
    "player_detections": ["frame", "ts", "player_id"],
    "pose":    ["frame", "player_id"],
}
OPTIONAL_TABLES = {"pose"}  # absence disables technique metrics, not a hard error

def load_match(data_dir: str) -> dict[str, pd.DataFrame]:
    d = pathlib.Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, required_cols in REQUIRED.items():
        path = d / f"{name}.parquet"
        if not path.exists():
            if name in OPTIONAL_TABLES:
                continue
            raise FileNotFoundError(f"Missing required table: {path}")
        df = pd.read_parquet(path)
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

### 2.5 Event model

```python
# shuttle_coach/model/events.py
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class MatchModel:
    match_id: str
    rallies: pd.DataFrame
    shots: pd.DataFrame
    hits: pd.DataFrame
    shuttle: pd.DataFrame
    positions: pd.DataFrame          # player_detections
    pose: pd.DataFrame | None
    player_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame]) -> "MatchModel":
        rallies = tables["rallies"]
        shots = tables["shots"]
        pids = sorted(set(shots["player_id"].dropna().astype(int)))
        return cls(
            match_id=str(rallies["match_id"].iloc[0]),
            rallies=rallies, shots=shots, hits=tables["hits"],
            shuttle=tables["shuttle"], positions=tables["player_detections"],
            pose=tables.get("pose"), player_ids=pids,
        )

    def shots_of(self, player_id: int) -> pd.DataFrame:
        return self.shots[self.shots["player_id"] == player_id]

    def positions_of(self, player_id: int) -> pd.DataFrame:
        return self.positions[self.positions["player_id"] == player_id]
```

### 2.6 Metric engine (plugin registry)

```python
# shuttle_coach/metrics/base.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
from shuttle_coach.model.events import MatchModel

REGISTRY: list[type["Metric"]] = []

def register(cls):
    REGISTRY.append(cls)
    return cls

@dataclass
class MetricResult:
    metric_id: str          # stable id, e.g. "movement.recovery_time"
    player_id: int | None   # None = match-level
    value: float | dict     # scalar or breakdown
    unit: str               # "s", "m", "%", "count", ...
    sample_size: int        # n events behind the value
    confidence: float       # 0..1 (e.g. mean upstream conf, or sample-based)
    context: dict[str, Any] # extra fields for narration/rules
    def to_row(self) -> dict:
        d = asdict(self); d["value"] = d["value"]; return d

class Metric:
    metric_id: str = "base"
    requires: set[str] = set()        # capability tags needed
    def applicable(self, caps: set[str]) -> bool:
        return self.requires.issubset(caps)
    def compute(self, m: MatchModel) -> list[MetricResult]:
        raise NotImplementedError
```

```python
# shuttle_coach/metrics/movement.py
import numpy as np
from shuttle_coach.metrics.base import Metric, MetricResult, register

BASE_POSITION = None  # computed per player as their median court position

@register
class RecoveryTime(Metric):
    """Time to return toward base position after playing a shot.
    Late recovery → out of position → conceding court control."""
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
                after = pos[pos["ts"] >= s["start_ts"]].head(60)  # ~window after hit
                if after.empty:
                    continue
                d = np.linalg.norm(after[["court_x", "court_y"]].to_numpy() - base, axis=1)
                # time until within 1.0 m of base
                back = np.argmax(d < 1.0) if (d < 1.0).any() else len(d) - 1
                recov.append(after["ts"].iloc[back] - s["start_ts"])
            if recov:
                out.append(MetricResult(
                    self.metric_id, pid, float(np.mean(recov)), "s",
                    sample_size=len(recov), confidence=min(1.0, len(recov) / 30),
                    context={"median": float(np.median(recov)),
                             "base_xy": base.round(2).tolist()}))
        return out

@register
class CourtCoverage(Metric):
    """Area / spread of court the player covers; and time spent per zone."""
    metric_id = "movement.court_coverage"
    requires = {"movement"}

    def compute(self, m):
        out = []
        for pid in m.player_ids:
            pos = m.positions_of(pid).dropna(subset=["court_x", "court_y"])
            if len(pos) < 10:
                continue
            xs, ys = pos["court_x"].to_numpy(), pos["court_y"].to_numpy()
            # 6-zone grid: front/mid/rear × left/right (court split at its midlines)
            zones = self._zone_histogram(xs, ys)
            out.append(MetricResult(
                self.metric_id, pid, zones, "%",
                sample_size=len(pos), confidence=1.0,
                context={"x_std": float(xs.std()), "y_std": float(ys.std())}))
        return out

    @staticmethod
    def _zone_histogram(xs, ys):
        # placeholder zone edges; align to canonical court model in metres
        import numpy as np
        x_edges = [0, 4.0, 8.0, 13.4]   # rear / mid / front along length (own half logic upstream)
        y_edges = [0, 3.05, 6.10]       # left / right across width
        H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
        H = (H / H.sum() * 100).round(1)
        labels = ["rear", "mid", "front"]
        side = ["left", "right"]
        return {f"{labels[i]}_{side[j]}": float(H[i, j])
                for i in range(3) for j in range(2)}
```

```python
# shuttle_coach/metrics/shots.py
from shuttle_coach.metrics.base import Metric, MetricResult, register

@register
class ShotMix(Metric):
    """Distribution of shot types per player (style fingerprint)."""
    metric_id = "shots.mix"
    requires = {"shots"}
    def compute(self, m):
        out = []
        for pid in m.player_ids:
            s = m.shots_of(pid)
            if s.empty:
                continue
            mix = (s["shot_type"].value_counts(normalize=True) * 100).round(1).to_dict()
            out.append(MetricResult(self.metric_id, pid, mix, "%",
                       sample_size=len(s), confidence=float(s["shot_conf"].mean())
                       if "shot_conf" in s else 1.0, context={}))
        return out

@register
class ShotEffectiveness(Metric):
    """Win/loss outcome conditioned on shot type (which shots win points)."""
    metric_id = "shots.effectiveness"
    requires = {"shots"}
    def compute(self, m):
        out = []
        rally_winner = m.rallies.set_index("rally_id")["winner_player_id"].to_dict()
        for pid in m.player_ids:
            s = m.shots_of(pid).copy()
            if s.empty:
                continue
            # last shot of a rally by this player vs rally outcome
            s["won"] = s["rally_id"].map(rally_winner) == pid
            eff = (s.groupby("shot_type")["won"].mean() * 100).round(1).to_dict()
            counts = s.groupby("shot_type")["won"].count().to_dict()
            out.append(MetricResult(self.metric_id, pid, eff, "%",
                       sample_size=len(s), confidence=1.0,
                       context={"counts": counts}))
        return out
```

```python
# shuttle_coach/metrics/errors.py
from shuttle_coach.metrics.base import Metric, MetricResult, register

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
                reasons = (lost["end_reason"].value_counts(normalize=True) * 100).round(1).to_dict()
            else:
                reasons = {}
            out.append(MetricResult(self.metric_id, pid, reasons, "%",
                       sample_size=int(len(lost)), confidence=1.0, context={}))
        return out
```

```python
# shuttle_coach/metrics/technique.py  (optional; requires pose)
import numpy as np
from shuttle_coach.metrics.base import Metric, MetricResult, register

@register
class PreparationConsistency(Metric):
    """Variability of body posture at hit frames for a given shot type.
    High variance in a 'should-be-grooved' shot suggests inconsistent technique."""
    metric_id = "technique.preparation_consistency"
    requires = {"technique"}
    def compute(self, m):
        if m.pose is None:
            return []
        out = []
        hits = m.hits.dropna(subset=["player_id"])
        for pid in m.player_ids:
            ph = hits[hits["player_id"] == pid]
            frames = m.pose[(m.pose["player_id"] == pid) &
                            (m.pose["frame"].isin(ph["frame"]))]
            if len(frames) < 5:
                continue
            # example: trunk lean angle from shoulders/hips keypoints
            angles = self._trunk_angles(frames)
            out.append(MetricResult(self.metric_id, pid, float(np.std(angles)), "deg",
                       sample_size=len(angles), confidence=1.0,
                       context={"mean_angle": float(np.mean(angles))}))
        return out

    @staticmethod
    def _trunk_angles(frames):
        # COCO indices: shoulders 5/6, hips 11/12
        sx = (frames["kpt_5_x"] + frames["kpt_6_x"]) / 2
        sy = (frames["kpt_5_y"] + frames["kpt_6_y"]) / 2
        hx = (frames["kpt_11_x"] + frames["kpt_12_x"]) / 2
        hy = (frames["kpt_11_y"] + frames["kpt_12_y"]) / 2
        return np.degrees(np.arctan2((sx - hx), (sy - hy))).to_numpy()
```

### 2.7 Running the engine + metrics store

```python
# shuttle_coach/metrics/run.py
import pandas as pd
from shuttle_coach.metrics.base import REGISTRY

def run_metrics(match, caps):
    results = []
    for cls in REGISTRY:
        metric = cls()
        if metric.applicable(caps):
            results.extend(metric.compute(match))
    return results

def metrics_to_frame(results) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])

def save_metrics(results, out_dir):
    import json, pathlib
    p = pathlib.Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    df = metrics_to_frame(results)
    df.to_parquet(p / "metrics.parquet")
    with open(p / "metrics.json", "w") as f:
        json.dump([r.to_row() for r in results], f, indent=2, default=str)
    return p / "metrics.json"
```

### 2.8 Rule-based feedback (deterministic, offline)

This path needs no network and is the trustworthy default. Rules map metric values to *findings*; findings are ranked by estimated impact (points lost, frequency, sample size).

```python
# shuttle_coach/feedback/rules.py
from dataclasses import dataclass

@dataclass
class Finding:
    code: str
    player_id: int | None
    severity: float          # 0..1, used for prioritization
    headline: str
    detail: str
    evidence: list[str]      # metric_ids that justify this finding

def derive_findings(results_by_id) -> list[Finding]:
    findings = []

    rec = results_by_id.get("movement.recovery_time", [])
    for r in rec:
        if r.value > 0.8 and r.sample_size >= 15:   # threshold in seconds
            findings.append(Finding(
                code="slow_recovery", player_id=r.player_id,
                severity=min(1.0, (r.value - 0.8) / 0.8),
                headline="Slow recovery to base position",
                detail=(f"Average recovery {r.value:.2f}s "
                        f"(median {r.context.get('median'):.2f}s) over "
                        f"{r.sample_size} shots. Returning to base faster "
                        f"would reduce time spent out of position."),
                evidence=[r.metric_id]))

    eff = results_by_id.get("shots.effectiveness", [])
    for r in eff:
        weak = {k: v for k, v in r.value.items()
                if v < 35 and r.context.get("counts", {}).get(k, 0) >= 8}
        for shot, winrate in weak.items():
            findings.append(Finding(
                code=f"weak_shot_{shot}", player_id=r.player_id,
                severity=(35 - winrate) / 35,
                headline=f"Low success on {shot}",
                detail=(f"{shot} ends the rally in your favor only {winrate:.0f}% "
                        f"of the time ({r.context['counts'][shot]} attempts)."),
                evidence=[r.metric_id]))

    err = results_by_id.get("errors.location_reason", [])
    for r in err:
        unforced = r.value.get("unforced_error", 0)
        if unforced > 30 and r.sample_size >= 10:
            findings.append(Finding(
                code="high_unforced", player_id=r.player_id,
                severity=min(1.0, unforced / 60),
                headline="High share of unforced errors",
                detail=(f"{unforced:.0f}% of lost points are unforced "
                        f"({r.sample_size} lost rallies). Shot tolerance/"
                        f"consistency is the highest-leverage area."),
                evidence=[r.metric_id]))

    return findings
```

```python
# shuttle_coach/feedback/report.py
def render_report(findings, top_k=5) -> str:
    findings = sorted(findings, key=lambda f: f.severity, reverse=True)
    lines = ["# Coaching Report", ""]
    lines.append("## Priorities")
    for i, f in enumerate(findings[:top_k], 1):
        lines.append(f"{i}. **{f.headline}** — {f.detail} "
                     f"_(evidence: {', '.join(f.evidence)})_")
    lines.append("")
    lines.append("## All findings")
    for f in findings:
        lines.append(f"- [{f.severity:.2f}] {f.headline}: {f.detail}")
    return "\n".join(lines)
```

### 2.9 LLM narration via RAG over metrics (optional)

The narration layer answers free-form questions ("where am I losing points?", "what should I drill this week?") but is constrained to the computed metrics. It never sees raw detections. Retrieval pulls the relevant `MetricResult` records into the prompt; generation must cite the `metric_id`s it used, and a post-check rejects any sentence with no citation.

```python
# shuttle_coach/narration/llm.py
import json

SYSTEM = (
    "You are a badminton coaching assistant. You may ONLY use the metrics "
    "provided in the context. Every claim must cite the metric_id(s) it relies "
    "on in square brackets, e.g. [movement.recovery_time]. If the metrics do "
    "not support an answer, say so. Do not invent numbers."
)

def build_context(results, question, retriever):
    """Select the metrics most relevant to the question."""
    selected = retriever.search(question, results, k=12)
    return json.dumps([r.to_row() for r in selected], default=str, indent=2)

def answer(question, results, retriever, llm_call):
    context = build_context(results, question, retriever)
    prompt = (f"{SYSTEM}\n\nMETRICS:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
    text = llm_call(prompt)              # any backend; injected dependency
    _enforce_citations(text, results)    # raises/repairs if ungrounded
    return text

def _enforce_citations(text, results):
    valid = {r.metric_id for r in results}
    import re
    cited = set(re.findall(r"\[([a-z_]+\.[a-z_]+)\]", text))
    unknown = cited - valid
    if unknown:
        raise ValueError(f"Narration cited unknown metrics: {unknown}")
    # NFR1: ensure non-trivial sentences carry at least one citation
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 6]
    uncited = [s for s in sentences if not re.search(r"\[[a-z_]+\.[a-z_]+\]", s)]
    if uncited:
        raise ValueError(f"Ungrounded sentences: {uncited}")
```

`retriever` can start as a trivial keyword/tag matcher over `metric_id` + `context` keys; upgrade to embeddings only if needed. `llm_call` is an injected function so the app is backend-agnostic and the analytics path never depends on it.

### 2.10 CLI

```python
# shuttle_coach/cli.py
import argparse, json
from shuttle_coach.io.loader import load_match, capabilities
from shuttle_coach.model.events import MatchModel
from shuttle_coach.metrics.run import run_metrics, save_metrics
from shuttle_coach.feedback.rules import derive_findings
from shuttle_coach.feedback.report import render_report

def main():
    ap = argparse.ArgumentParser(prog="shuttle-coach")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("data_dir")
    a.add_argument("--out", default="out")
    a.add_argument("--report", action="store_true")
    args = ap.parse_args()

    tables = load_match(args.data_dir)
    caps = capabilities(tables)
    match = MatchModel.from_tables(tables)
    results = run_metrics(match, caps)
    save_metrics(results, args.out)

    if args.report:
        by_id = {}
        for r in results:
            by_id.setdefault(r.metric_id, []).append(r)
        findings = derive_findings(by_id)
        report = render_report(findings)
        with open(f"{args.out}/report.md", "w") as f:
            f.write(report)
        print(report)

if __name__ == "__main__":
    main()
```

Usage:
```
pip install -e .
shuttle-coach analyze ./match_0042_parquet --out ./out --report
```

### 2.11 Metric catalogue (what to compute, and why it coaches)

| metric_id | family | needs | coaching meaning |
|-----------|--------|-------|------------------|
| `movement.recovery_time` | movement | positions(court) | late recovery → out of position |
| `movement.court_coverage` | movement | positions(court) | zone over/under-use, base discipline |
| `movement.distance_per_rally` | movement | positions(court) | workload / efficiency |
| `shots.mix` | shots | shots | style fingerprint, predictability |
| `shots.effectiveness` | shots | shots+rallies | which shots actually win points |
| `tactical.placement` | tactical | hits/shuttle(court) | depth & width control, targeting |
| `tactical.rally_construction` | tactical | shots(seq) | shot sequences that precede winners |
| `errors.location_reason` | errors | rallies | where/how points are lost |
| `technique.preparation_consistency` | technique | pose | grooved vs inconsistent preparation |
| `technique.contact_point` | technique | pose+hits | early/late contact tendencies |

### 2.12 Data-quality handling

- **Confidence weighting:** metrics carry the mean upstream confidence (e.g. `shot_conf`); rules require a minimum sample size before firing.
- **Missing coordinates:** rows without `court_x/court_y` are dropped per-metric, not globally.
- **Label-noise guard:** shot-type metrics can optionally exclude `shot_conf` below a configurable floor.
- **Loud failures:** schema/court-range violations raise on load (NFR3), rather than producing silently wrong advice.

### 2.13 Testing

- **Unit:** each metric against a small synthetic `MatchModel` with known answer.
- **Property:** metrics never emit a result with `sample_size == 0`; values within unit-valid ranges.
- **Groundedness:** automated check that every report finding's `evidence` ids exist in the metrics store; every LLM sentence carries a valid citation (NFR1).
- **Golden match:** one fully-annotated match fixture; snapshot the metrics + report.

### 2.14 Build order (milestones)

1. **M1** — Loader + capability detection + event model; round-trip a real parquet folder.
2. **M2** — Movement + shots + errors metrics; metrics store output.
3. **M3** — Rule-based findings + prioritized report.md (the trustworthy default).
4. **M4** — Tactical + technique (pose) metrics behind capability flags.
5. **M5** — RAG narration with citation enforcement, backend-agnostic LLM call.

### 2.15 Open questions to confirm

- Exact parquet column names/dtypes vs §2.1 (especially `end_reason` vocabulary and whether `court_x/court_y` are populated downstream of the homography step).
- Are court coordinates in metres or normalized units? Zone edges in `court_coverage` must match.
- Singles only? Player-id stability across a match (do `player_id`s persist correctly through the rally)?
- Is there a ground-truth/labelled match available for the golden-test fixture and for validating shot-effectiveness logic?
