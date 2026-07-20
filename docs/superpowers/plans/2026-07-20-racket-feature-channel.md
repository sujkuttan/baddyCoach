# Racket Feature Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real racket-detection feature channel (YOLOv8 on RacketDB) and feed racket geometry into the existing non-BST consumers — ownership scorer, rule-based stroke classifier, and hit-frame refinement — replacing/augmenting wrist proxies, with graceful fallback when racket detection is unavailable.

**Architecture:** A new `RacketTracker` (lazy YOLOv8 singleton, parallel to `get_yolov8`) runs once per video and emits `racket_detections` (per-frame, per-player bbox + head point). That artifact is threaded into `strokes.py` and `hits.py`, then consumed by `ownership_scorer` (racket motion + blended proximity), `stroke_features` (racket contact evidence), and `hits` (racket-nearest tiebreaker). BST is untouched. All consumers fall back to current wrist-proxy behavior when `get_racket()` returns `None`.

**Tech Stack:** Python 3.12, `ultralytics` YOLOv8 (already a dependency), NumPy, pandas, PyTorch (CPU inference). RacketDB YOLOv8 weights from HuggingFace `muhabdulhaq/racketdb`.

## Global Constraints

- No retraining of BST or any stroke model; BST tensor `in_dim` is frozen by its checkpoint — do NOT modify BST inputs.
- All new thresholds/constants go in `backend/app/config/settings.py` (pydantic `Settings`); no hardcoded magic numbers in stage modules.
- Use `PipelineLogger` from `shared/logging.py` with **keyword args** (never printf-style positional args).
- Follow existing patterns: lazy singleton in `shared/models.py` with graceful `None` return (mirror `get_mmaction2`).
- Log only via `logger.*`; avoid bare `print()` in model/pipeline code.
- Full backend suite (`python -m pytest -m "not gpu and not model and not integration"`) must pass except the 2 known pre-existing failures:
  - `test_colab_pipeline.py::test_colab_delegates_court_space_enrichment_to_backend_helper`
  - `test_colab_pipeline.py::test_colab_uses_continuity_aware_tracknet_candidate_selection`
- Colab parity: any new settings fields must be surfaced as Colab form fields / CLI args in `colab/pipeline.py`.
- Frequent commits; one task per commit.

---

### Task 1: Settings — racket configuration fields

**Files:**
- Modify: `backend/app/config/settings.py`
- Test: `backend/tests/test_settings.py` (create if absent)

**Interfaces:**
- Produces: new pydantic settings fields consumed by later tasks.
- Consumes: existing `Settings` class pattern.

- [ ] **Step 1: Write the failing test**

```python
def test_racket_settings_defaults():
    from app.config.settings import settings
    assert settings.racket_enabled is True
    assert settings.racket_min_conf == 0.4
    assert settings.racket_proximity_blend == 0.5
    assert settings.racket_head_margin == 0.1
    assert settings.racket_model_path is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_settings.py::test_racket_settings_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'racket_enabled'`

- [ ] **Step 3: Write minimal implementation**

Add to `backend/app/config/settings.py` inside the `Settings` class (near the other model path fields, e.g. after `yolov8_model_path`):

```python
    # ── Racket detection (RacketDB YOLOv8) ─────────────────────────────
    racket_enabled: bool = True
    racket_model_path: str = "ckpts/racketdb_yolov8.pt"
    racket_min_conf: float = 0.4
    racket_proximity_blend: float = 0.5
    racket_motion_weight: float = 0.6
    racket_dist_weight: float = 0.4
    racket_head_margin: float = 0.1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_settings.py::test_racket_settings_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/settings.py backend/tests/test_settings.py
git commit -m "feat(settings): add racket detection config fields"
```

---

### Task 2: `model_downloader` — RacketDB weight entry

**Files:**
- Modify: `backend/app/config/model_downloader.py`
- Test: `backend/tests/test_model_downloader.py` (create if absent)

**Interfaces:**
- Consumes: `ensure_model`, `MODEL_REGISTRY` from `app.pipeline.shared.models`.
- Produces: `racketdb` registered in `MODEL_REGISTRY` + `BACKEND_MODELS` so `python app/config/model_downloader.py` downloads it.

- [ ] **Step 1: Write the failing test**

```python
def test_racketdb_in_model_registry():
    from app.pipeline.shared.models import MODEL_REGISTRY
    assert "racketdb" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["racketdb"]
    assert entry["kind"] == "ultralytics"
    assert entry["url"].endswith(".pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_model_downloader.py::test_racketdb_in_model_registry -v`
Expected: FAIL (`"racketdb" not in MODEL_REGISTRY`)

- [ ] **Step 3: Write minimal implementation**

In `backend/app/pipeline/shared/models.py`, add `racketdb` to `MODEL_REGISTRY` (alongside `yolov8s`). Mirror the `yolov8s` entry format:

```python
    "racketdb": (
        Path("ckpts/racketdb_yolov8.pt"),
        "https://huggingface.co/muhabdulhaq/racketdb/resolve/main/racketdb_yolov8.pt",
        "ultralytics",
    ),
```

In `backend/app/config/model_downloader.py`, add `"racketdb"` to `BACKEND_MODELS`:

```python
BACKEND_MODELS = [
    "tracknet", "inpaintnet", "bst", "rtmpose", "hrnet", "court_kprcnn", "yolov8s", "racketdb",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_model_downloader.py::test_racketdb_in_model_registry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/shared/models.py backend/app/config/model_downloader.py backend/tests/test_model_downloader.py
git commit -m "feat(downloader): register RacketDB YOLOv8 weights"
```

---

### Task 3: `RacketTracker` model + lazy singleton

**Files:**
- Create: `backend/app/models/racket.py`
- Modify: `backend/app/pipeline/shared/models.py` (add `get_racket()`)
- Test: `backend/tests/test_racket.py`

**Interfaces:**
- Produces:
  - `class RacketTracker` with method `detect(frames_dir_or_frames, ...) -> list[dict]` returning `[{"frame": int, "player_side": str, "bbox": tuple, "conf": float, "head_point": tuple}]`.
  - `get_racket() -> Optional[RacketTracker]` lazy singleton returning `None` on missing weights / import error.
- Consumes: `settings.racket_enabled`, `settings.racket_model_path`, `settings.racket_min_conf`, `settings.racket_head_margin`; YOLOv8 via `ultralytics` (already used by `YOLOv8Tracker`).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

def test_racket_tracker_head_point_extraction():
    from app.models.racket import RacketTracker
    tr = RacketTracker.__new__(RacketTracker)
    # bbox (x1,y1,x2,y2) = (100,200,140,300); head = top-center + margin
    head = tr._head_point((100, 200, 140, 300), margin=0.1)
    # top-center x = (100+140)/2 = 120 ; y nudged up by margin*height = 0.1*100 =10 => 200-10=190
    assert abs(head[0] - 120.0) < 1e-6
    assert abs(head[1] - 190.0) < 1e-6

def test_get_racket_returns_none_when_disabled(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "racket_enabled", False)
    from app.pipeline.shared.models import get_racket
    assert get_racket() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_racket.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.models.racket'`)

- [ ] **Step 3: Write minimal implementation**

`backend/app/models/racket.py`:

```python
"""Racket detection tracker (YOLOv8 on RacketDB weights).

Produces per-frame, per-player racket detections with a derived
racket-head point. Returns None gracefully when weights are missing.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.config.settings import settings

logger = logging.getLogger("racket_tracker")


class RacketTracker:
    """Single-class YOLOv8 racket detector with player association."""

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.4,
                 device: str = "cpu"):
        from ultralytics import YOLO
        self.model_path = model_path or settings.racket_model_path
        self.conf = conf
        self.device = device
        self.model = YOLO(self.model_path)

    @staticmethod
    def _head_point(bbox: tuple, margin: float = 0.1) -> tuple:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        h = max(y2 - y1, 1.0)
        head_y = y1 - margin * h
        return (float(cx), float(head_y))

    def detect(self, frames: List[np.ndarray], player_bboxes: dict) -> List[dict]:
        """Detect rackets per frame and associate to nearer player.

        frames: list of BGR images (one per frame index 0..N-1)
        player_bboxes: {frame: {side: bbox_tuple}}
        Returns list of {"frame","player_side","bbox","conf","head_point"}.
        """
        results = self.model(frames, conf=self.conf, device=self.device, verbose=False)
        out: List[dict] = []
        for fi, res in enumerate(results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item()) if box.conf is not None else 1.0
                bbox = (x1, y1, x2, y2)
                head = self._head_point(bbox, margin=settings.racket_head_margin)
                side = self._associate(fi, (x1, y1, x2, y2), player_bboxes)
                out.append({
                    "frame": fi,
                    "player_side": side or "near",
                    "bbox": bbox,
                    "conf": conf,
                    "head_point": head,
                })
        return out

    @staticmethod
    def _associate(frame: int, rbbox: tuple, player_bboxes: dict) -> Optional[str]:
        cands = player_bboxes.get(frame, {})
        if not cands:
            return None
        rcx, rcy = (rbbox[0] + rbbox[2]) / 2.0, (rbbox[1] + rbbox[3]) / 2.0
        best_side, best_d = None, None
        for side, pb in cands.items():
            pcx, pcy = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
            d = (pcx - rcx) ** 2 + (pcy - rcy) ** 2
            if best_d is None or d < best_d:
                best_d, best_side = d, side
        return best_side
```

In `backend/app/pipeline/shared/models.py`, add after `get_yolov8`:

```python
def get_racket():
    """Lazy getter for RacketTracker. Returns None if disabled or weights missing."""
    from app.config.settings import settings as s
    if not s.racket_enabled:
        return None
    if "racket" not in _models:
        try:
            from app.models.racket import RacketTracker
            tr = RacketTracker(
                model_path=str(s.racket_model_path) if s.racket_model_path else None,
                conf=s.racket_min_conf,
                device=_get_device(),
            )
            _models["racket"] = tr
        except Exception as e:
            logger.warning("RacketTracker not available: %s", e)
            return None
    return _models.get("racket")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_racket.py -v`
Expected: PASS (the `get_racket` disabled test needs `ultralytics` importable; if ultralytics is unavailable in the test env, decorate with `@pytest.mark.skipif` on import — but backend already imports ultralytics for YOLOv8, so it should import.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/racket.py backend/app/pipeline/shared/models.py backend/tests/test_racket.py
git commit -m "feat(racket): add RacketTracker model + get_racket singleton"
```

---

### Task 4: Racket artifact pass-through in `strokes.py`

**Files:**
- Modify: `backend/app/pipeline/strokes.py` (`_build_clip` and `StrokeClassificationStage.run`)
- Test: `backend/tests/test_strokes_racket.py` (create)

**Interfaces:**
- Consumes: `racket_detections` list from `RacketTracker.detect` (Task 3), threaded in via `run()` kwargs.
- Produces: per-clip `racket_head` (seq_len, 2, 2) and `racket_present` (seq_len, 2) arrays stored in the clip dict, plus `provenance` racket distance fields, so `extract_clip_features` (Task 6) and `ownership_scorer` (Task 5) can read them.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

def test_build_clip_includes_racket_arrays():
    from app.pipeline.strokes import _build_clip
    frames = list(range(100))
    racket_det = [
        {"frame": 10, "player_side": "near", "bbox": (1, 2, 3, 4), "conf": 0.9, "head_point": (2.0, 1.0)},
        {"frame": 10, "player_side": "far", "bbox": (5, 6, 7, 8), "conf": 0.9, "head_point": (6.0, 5.0)},
    ]
    clip = _build_clip(
        frames, 50, 100, None, None, None, None, None, None, None,
        racket_detections=racket_det, vid_w=1920, vid_h=1080,
        original_len=100, player_sides={"near": "player_1", "far": "player_2"},
    )
    assert "racket_head" in clip
    assert clip["racket_head"].shape == (100, 2, 2)
    # near player, frame 10 head point present
    assert np.allclose(clip["racket_head"][10, 0], [2.0, 1.0])
    assert np.allclose(clip["racket_head"][10, 1], [6.0, 5.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_strokes_racket.py::test_build_clip_includes_racket_arrays -v`
Expected: FAIL (`TypeError: _build_clip() got an unexpected keyword argument 'racket_detections'`)

- [ ] **Step 3: Write minimal implementation**

In `backend/app/pipeline/strokes.py`, add `racket_detections: list | None = None` parameter to `_build_clip` (after `player_sides=`). Inside `_build_clip`, after the shuttle array is built (around line 305), add:

```python
    # ── Racket head points + presence (Scope A racket channel) ─────
    racket_head = np.zeros((seq_len, 2, 2), dtype=np.float32)
    racket_present = np.zeros((seq_len, 2), dtype=bool)
    racket_lookup = {}
    if racket_detections is not None:
        for rd in racket_detections:
            racket_lookup.setdefault(int(rd["frame"]), {})[rd["player_side"]] = rd
    for t, frame in enumerate(frames[:seq_len]):
        rframe = racket_lookup.get(int(frame))
        if not rframe:
            continue
        for p_idx, side in ((0, "far"), (1, "near")):
            rd = rframe.get(side)
            if rd is None:
                continue
            hp = rd.get("head_point")
            if hp is not None:
                racket_head[t, p_idx] = [float(hp[0]), float(hp[1])]
                racket_present[t, p_idx] = True
```

Add `"racket_head": racket_head, "racket_present": racket_present` to the returned clip dict.

In `StrokeClassificationStage.run`, accept `racket_detections` from `ctx`/`input_data` (mirror how `shuttle_df` is obtained) and pass it into every `_build_clip(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_strokes_racket.py::test_build_clip_includes_racket_arrays -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/strokes.py backend/tests/test_strokes_racket.py
git commit -m "feat(strokes): thread racket detections into clip building"
```

---

### Task 5: Ownership scorer — racket motion + blended proximity

**Files:**
- Modify: `backend/app/pipeline/shared/ownership_scorer.py` (`racket_motion_score`, `normalized_proximity_score`, and their call sites at lines ~802, ~812)
- Test: `backend/tests/test_ownership_scorer.py` (extend)

**Interfaces:**
- Consumes: racket detections threaded via `OwnershipScorer.score(...)` new kwarg `racket_detections: list | None = None`; per-player racket head sequences derived inside `score` (mirror `_kps_window`).
- Produces: updated `(near_score, far_score)` from rewritten `racket_motion_score` and blended `normalized_proximity_score`. Must preserve 0–1 output range and fallback to wrist behavior when `racket_detections is None`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

def test_racket_motion_score_uses_racket_when_present():
    from app.pipeline.shared.ownership_scorer import racket_motion_score
    # near has high racket-head speed around hit; far has none
    near_heads = [np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([0.0, 0.0])]
    far_heads = [np.array([0.0, 0.0])] * 3
    racket_seq = {"near": near_heads, "far": far_heads}
    n, f = racket_motion_score(racket_seq, hit_idx=1, motion_weight=0.6, dist_weight=0.4)
    assert n > f  # near moved its racket, far did not

def test_racket_motion_score_falls_back_when_none():
    from app.pipeline.shared.ownership_scorer import racket_motion_score
    n, f = racket_motion_score(None, hit_idx=1)
    # returns neutral (unknown_score=0.5) split, no error
    assert n == 0.5 and f == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_ownership_scorer.py -v`
Expected: FAIL (`racket_motion_score() got an unexpected keyword argument 'motion_weight'` or signature mismatch)

- [ ] **Step 3: Write minimal implementation**

Rewrite `racket_motion_score` signature to accept racket-head sequences instead of keypoint lists:

```python
def racket_motion_score(racket_head_seq: dict | None,
                        hit_idx: int,
                        motion_weight: float = 0.6,
                        dist_weight: float = 0.4,
                        shuttle_px_seq: dict | None = None,
                        unknown_score: float = 0.50,
                        vel_norm: float = 50.0) -> tuple[float, float]:
    """Racket-based motion score (Scope A).

    Uses racket-head speed around hit_idx and min racket-shuttle distance.
    racket_head_seq: {"near": [ (x,y), ... ], "far": [...]}.
    Falls back to (unknown_score, unknown_score) when no racket data.
    """
    def _motion(seq, sh_seq):
        if not seq or len(seq) < 3:
            return unknown_score
        if hit_idx < 1 or hit_idx >= len(seq) - 1:
            return unknown_score
        prev, nxt = np.array(seq[hit_idx - 1]), np.array(seq[hit_idx + 1])
        speed = float(np.linalg.norm(nxt - prev) / 2.0)
        speed_n = min(1.0, speed / vel_norm)
        dist_n = 0.0
        if sh_seq and len(sh_seq) > hit_idx:
            sh = np.array(sh_seq[hit_idx], dtype=float)
            if np.all(np.isfinite(sh)):
                d = float(np.linalg.norm(np.array(seq[hit_idx], dtype=float) - sh))
                dist_n = float(np.exp(-d / 100.0))
        raw = motion_weight * speed_n + dist_weight * dist_n
        return min(1.0, raw / (motion_weight + dist_weight))

    near = _motion(racket_head_seq.get("near") if racket_head_seq else None,
                   shuttle_px_seq.get("near") if shuttle_px_seq else None)
    far = _motion(racket_head_seq.get("far") if racket_head_seq else None,
                  shuttle_px_seq.get("far") if shuttle_px_seq else None)
    total = near + far
    if total > 0:
        near, far = near / total, far / total
    return float(near), float(far)
```

Update `normalized_proximity_score` to accept an optional `racket_head` point and blend with the wrist term using `racket_proximity_blend` (new param, default from settings at call site). Add `racket_head_near`/`racket_head_far` params; when present, compute an analogous `exp(-dist/sigma)` and blend: `score = (1-blend)*wrist_score + blend*racket_score`.

At the call site (lines ~802, ~812) inside `OwnershipScorer.score`, build per-player racket head windows (mirror `_kps_window`) from the new `racket_detections` kwarg, then pass them in. When `racket_detections is None`, pass `None` so fallback behavior is identical to today.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_ownership_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/shared/ownership_scorer.py backend/tests/test_ownership_scorer.py
git commit -m "feat(ownership): racket motion + blended proximity sub-scores"
```

---

### Task 6: Rule-based classifier — racket contact features

**Files:**
- Modify: `backend/app/pipeline/shared/stroke_features.py` (`extract_clip_features`, `classify_by_family`, `_build_evidence`)
- Test: `backend/tests/test_stroke_features.py` (extend)

**Interfaces:**
- Consumes: `clip["racket_head"]` (seq_len, 2, 2) and `clip["racket_present"]` (seq_len, 2) produced in Task 4; shuttle from `clip["shuttle"]`.
- Produces: new feature keys `racket_contact_distance`, `racket_present_frac`, `racket_peak_speed` in the returned feats dict; used in `classify_by_family` / `_build_evidence` as the genuine contact cue.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

def test_extract_clip_features_includes_racket():
    from app.pipeline.shared.stroke_features import extract_clip_features
    seq_len = 10
    clip = {
        "shuttle": np.tile(np.array([0.5, 0.5]), (seq_len, 1)).astype(float),
        "pos": np.zeros((seq_len, 2, 2)),
        "JnB": np.zeros((seq_len, 2, 72)),
        "video_len": seq_len,
        "racket_head": np.zeros((seq_len, 2, 2)),
        "racket_present": np.ones((seq_len, 2), dtype=bool),
    }
    # put a near-player racket head close to shuttle at frame 0
    clip["racket_head"][0, 0] = [0.5, 0.5]
    feats = extract_clip_features(clip)
    assert "racket_contact_distance" in feats
    assert "racket_present_frac" in feats
    assert "racket_peak_speed" in feats
    assert feats["racket_contact_distance"] < 0.1
    assert feats["racket_present_frac"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_stroke_features.py -v`
Expected: FAIL (`KeyError: 'racket_contact_distance'`)

- [ ] **Step 3: Write minimal implementation**

In `extract_clip_features`, after the shuttle contact block, add:

```python
    # ── Racket contact cues (Scope A) ──────────────────────────
    rh = clip.get("racket_head")
    rp = clip.get("racket_present")
    if rh is not None and rp is not None and len(rh) > 0:
        present_frac = float(np.mean(rp))
        feats["racket_present_frac"] = present_frac
        # contact distance = min over frames of near/far racket-head to shuttle
        min_d = None
        peak_speed = 0.0
        prev_head = None
        for t in range(min(len(rh), len(shuttle))):
            for p_idx in (0, 1):
                if t < len(rp) and rp[t, p_idx]:
                    head = rh[t, p_idx]
                    sh = shuttle[t]
                    if np.all(np.isfinite(head)) and np.all(np.isfinite(sh)):
                        d = float(np.linalg.norm(head - sh))
                        if min_d is None or d < min_d:
                            min_d = d
                    if prev_head is not None:
                        sp = float(np.linalg.norm(head - prev_head))
                        peak_speed = max(peak_speed, sp)
            prev_head = rh[t, 0] if (t < len(rp) and rp[t, 0]) else None
        feats["racket_contact_distance"] = min_d if min_d is not None else 1.0
        feats["racket_peak_speed"] = peak_speed
    else:
        feats["racket_present_frac"] = 0.0
        feats["racket_contact_distance"] = 1.0
        feats["racket_peak_speed"] = 0.0
```

In `classify_by_family` / `_build_evidence`, where wrist-shuttle distance currently informs contact height/zone, substitute `feats["racket_contact_distance"]` as the genuine contact cue. Keep `wrist_shuttle_distance` as a fallback only when `racket_present_frac == 0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_stroke_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/shared/stroke_features.py backend/tests/test_stroke_features.py
git commit -m "feat(stroke_features): racket contact cues in rule-based classifier"
```

---

### Task 7: Hit refinement — racket-nearest tiebreaker

**Files:**
- Modify: `backend/app/pipeline/hits.py` (`_find_nearest_wrist_frame`)
- Test: `backend/tests/test_hits.py` (extend)

**Interfaces:**
- Consumes: `racket_detections` threaded into `HitFrameLocalizationStage.run` (mirror `shuttle_df`); per-frame racket head points.
- Produces: refined best frame that, when racket data is present, uses racket-shuttle distance as the 30% tiebreaker instead of (or in addition to) wrist.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

def test_find_nearest_racket_frame_prefers_racket():
    from app.pipeline.hits import _find_nearest_racket_frame
    # candidate frames 0..8, hit at 4, window +-4
    shuttle_df = pd.DataFrame({
        "frame": list(range(9)),
        "x": [100.0] * 9,
        "y": [100.0] * 9,
        "confidence": [0.9] * 9,
    })
    # racket head closest to shuttle at frame 6 (not 4)
    racket_det = {6: {"near": (100.0, 100.0)}}
    best = _find_nearest_racket_frame(
        candidate_frame=4, shuttle_df=shuttle_df, racket_detections=racket_det,
        window=4, min_shuttle_conf=0.3,
    )
    assert best == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_hits.py -v`
Expected: FAIL (`ImportError: cannot import name '_find_nearest_racket_frame'`)

- [ ] **Step 3: Write minimal implementation**

In `backend/app/pipeline/hits.py`, add a new function mirroring `_find_nearest_wrist_frame` but using racket head points:

```python
def _find_nearest_racket_frame(candidate_frame: int, shuttle_df: pd.DataFrame,
                               racket_detections: dict, window: int = 4,
                               min_shuttle_conf: float = 0.3) -> int:
    """Refine hit frame using racket-head proximity to shuttle (Scope A).

    racket_detections: {frame: {side: (hx, hy)}}.
    Returns the frame in +-window minimizing racket-shuttle distance,
    or candidate_frame if no racket data available.
    """
    lo = candidate_frame - window
    hi = candidate_frame + window + 1
    best_frame = candidate_frame
    best_dist: float | None = None
    for f in range(lo, hi):
        srows = shuttle_df[(shuttle_df["frame"] == f)]
        if len(srows) == 0:
            continue
        sx, sy = float(srows.iloc[0]["x"]), float(srows.iloc[0]["y"])
        rframe = racket_detections.get(f, {})
        for head in rframe.values():
            if head is None:
                continue
            d = float(np.sqrt((head[0] - sx) ** 2 + (head[1] - sy) ** 2))
            if best_dist is None or d < best_dist:
                best_dist, best_frame = d, f
    return best_frame
```

In `HitFrameLocalizationStage.run` / the refinement loop, after computing the current best frame via the existing wrist + direction-reversal score, if `racket_detections` is present, call `_find_nearest_racket_frame` and use its result as the 30% tiebreaker (combine: `final = 0.7*wrist_based + 0.3*racket_based` when racket present, else keep wrist-based). When `racket_detections is None`, behavior is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_hits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/hits.py backend/tests/test_hits.py
git commit -m "feat(hits): racket-nearest tiebreaker in hit-frame refinement"
```

---

### Task 8: Colab parity + integration wiring

**Files:**
- Modify: `colab/pipeline.py` (form fields / CLI args for the 7 settings; pass `racket_detections` into the colab stroke stage)
- Test: `backend/tests/test_colab_pipeline.py` (ensure no new breakage beyond the 2 known failures)

**Interfaces:**
- Consumes: the settings fields from Task 1; `RacketTracker` from Task 3.
- Produces: colab run produces `racket_detections` and passes them to the same consumers as the backend.

- [ ] **Step 1: Write the failing test**

```python
def test_colab_exposes_racket_settings():
    import importlib.util
    spec = importlib.util.spec_from_file_location("colab_pipeline", "colab/pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parser = mod.build_arg_parser() if hasattr(mod, "build_arg_parser") else None
    # at minimum the module should reference the new settings
    assert "racket_enabled" in mod.__dict__ or hasattr(mod, "RACKET_ENABLED")
```

(If `colab/pipeline.py` has no `build_arg_parser`, instead assert the module imports and contains the racket flag string.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_colab_pipeline.py::test_colab_exposes_racket_settings -v`
Expected: FAIL (flag/attribute not present)

- [ ] **Step 3: Write minimal implementation**

In `colab/pipeline.py`, add form fields / CLI args: `--racket-enabled`, `--racket-min-conf`, `--racket-proximity-blend`, `--racket-head-margin`, `--racket-motion-weight`, `--racket-dist-weight` wired to `settings`. Run `RacketTracker` once per video (mirror how `get_yolov8` is used) to build `racket_detections`, and pass it into the colab stroke-classification call the same way as the backend (Tasks 4–7).

- [ ] **Step 4: Run full suite (excluding known failures)**

Run: `cd backend && python -m pytest -m "not gpu and not model and not integration" -q`
Expected: PASS except the 2 known pre-existing failures:
- `test_colab_pipeline.py::test_colab_delegates_court_space_enrichment_to_backend_helper`
- `test_colab_pipeline.py::test_colab_uses_continuity_aware_tracknet_candidate_selection`

- [ ] **Step 5: Commit**

```bash
git add colab/pipeline.py backend/tests/test_colab_pipeline.py
git commit -m "feat(colab): racket detection parity with backend"
```

---

### Task 9: Graceful fallback verification

**Files:**
- Test: `backend/tests/test_racket_fallback.py` (create)

**Interfaces:**
- Consumes: all Tasks 3–8 behaviors with `racket_enabled=False` or `get_racket()` returning `None`.
- Produces: proof that pipeline output is byte-for-byte equivalent to pre-racket behavior when racket unavailable.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

def test_ownership_fallback_without_racket(monkeypatch):
    from app.pipeline.shared import ownership_scorer as os_mod
    # Build minimal inputs; racket_detections=None must not raise and must
    # return neutral-ish scores like the pre-racket wrist path.
    near_kps = np.random.rand(17, 3)
    far_kps = np.random.rand(17, 3)
    ns, fs = os_mod.normalized_proximity_score(
        np.array([10.0, 10.0]), None, near_kps, far_kps, 100.0, 100.0, None,
    )
    assert 0.0 <= ns <= 1.0 and 0.0 <= fs <= 1.0

def test_racket_disabled_yields_none(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "racket_enabled", False)
    from app.pipeline.shared.models import get_racket
    assert get_racket() is None
```

- [ ] **Step 2: Run test to verify it passes (fallback already wired in Tasks 3–5)**

Run: `cd backend && python -m pytest tests/test_racket_fallback.py -v`
Expected: PASS

- [ ] **Step 3: Run full suite once more**

Run: `cd backend && python -m pytest -m "not gpu and not model and not integration" -q`
Expected: PASS except the 2 known pre-existing failures.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_racket_fallback.py
git commit -m "test(racket): verify graceful fallback when racket unavailable"
```
