# Racket Feature Channel — Design

**Date:** 2026-07-20
**Status:** Approved (brainstorming → writing-plans pending)
**Scope:** Scope A — pipeline-side racket signals, no stroke-model retrain (Approach 1: augment existing wrist-proxy signals)

## Context

The BaddyCoach stroke-classification pipeline relies on BST (frozen checkpoint
`BST_CG_JnB_bone_merged.pt`) plus downstream heuristic consumers: `ownership_scorer`,
`stroke_features` (rule-based classifier), and `hits` (hit-frame refinement). The BST
tensor `in_dim` is fixed by the checkpoint, so new input features cannot be injected
into BST without retraining — a dead end (same gap that blocks MMAction2 / TemPose /
ST-GCN, none of which ship downloadable badminton-trained weights).

The literature (TemPose CVPRW 2023; Pribylina 2026 thesis, 89.5% on 10 classes)
consistently concludes that *pose alone is insufficient* for fine-grained badminton
strokes and that **racket/shuttle context is the missing signal**. Today the pipeline
approximates "racket" purely from wrist/elbow/shoulder COCO joints — a proxy, not a
real racket object.

**Goal:** Add a real racket-detection feature channel and feed it into the existing
non-BST consumers, replacing/augmenting wrist proxies with genuine racket geometry.
No retraining, no BST change, graceful fallback when racket detection is unavailable.

## Why RacketDB (not a dead end)

`RacketDB` (HuggingFace `muhabdulhaq/racketdb`, 22,682 images, YOLOv8 annotation
format; paper mAP50≈0.78 for YOLOv8) provides a **downloadable, pretrained YOLOv8
racket detector**. The pipeline already runs YOLOv8 (`ultralytics`) for player
detection via `shared/models.py::get_yolov8`, so the detector infra is free. This is
a real artifact — unlike TemPose/ST-GCN which have only training code, no weights.

## Architecture & Components (Section 1)

### New: `RacketTracker` — `backend/app/models/racket.py`
- Lazy-loads a YOLOv8 model from `settings.racket_model_path`.
- Weight source: **RacketDB YOLOv8 checkpoint**, downloaded by
  `app/config/model_downloader.py` to `ckpts/racketdb_yolov8.pt`
  (sourced from HuggingFace `muhabdulhaq/racketdb`, YOLOv8 format).
- Runs per-frame single-class ("racket") detection at conf ≥ `racket_min_conf`.
- Lightweight association: each racket bbox is assigned to the **nearer** on-court
  player (nearest bbox-center to player bbox-center), producing
  `racket_detections: [{frame, player_side, bbox, conf, head_point}]`.
- `head_point` = bbox top-center (racket head), the most informative point for
  shuttle contact. `racket_head_margin` controls where on the bbox the point is taken.
- Returns `None` when weights absent / ultralytics unavailable (graceful, like
  `get_mmaction2`).

### New: lazy singleton `get_racket()` — `backend/app/pipeline/shared/models.py`
Parallel to `get_yolov8()`. Returns `None` on missing weights → consumers fall back.

### Integration points (all existing; BST untouched)
1. `ownership_scorer.py` — replace wrist-proxy in `racket_motion_score` / proximity
   with real racket signals.
2. `stroke_features.py` — add `racket_shuttle_distance` + `racket_present` to
   `extract_clip_features`; use in `classify_by_family` evidence.
3. `hits.py` — racket-nearest-to-shuttle as a hit-refinement tiebreaker alongside
   wrist (extends `_find_nearest_wrist_frame`).
4. `strokes.py` `_build_clip` — accept `racket_detections` and thread it into clip
   features (parallel to `player_detections`).

## Data Flow (Section 2)

### Per-video pass
- `RacketTracker` runs once over all frames (CPU, cheap single-class YOLOv8),
  producing `racket_detections`.
- Persisted as pipeline artifact `racket_detections.json` (parallel to `players.json`).
- Threaded into `StrokeClassificationStage` (`strokes.py`) and
  `HitFrameLocalizationStage` (`hits.py`) via `input_keys` / `run()` kwargs.

### Per-clip feature derivation (in `_build_clip` / `extract_clip_features`)
- `racket_head_point[frame]` interpolated per player using the same short-gap linear
  interpolation + `bst_max_bbox_interp_gap` semantics already used for player bboxes.
- `racket_shuttle_distance[frame]` = ‖racket_head − shuttle_px‖ normalized by player
  bbox diagonal (matches existing wrist-distance convention); court-space variant via
  homography when valid.
- `racket_motion[frame]` = central-difference speed of `racket_head_point` (px/frame);
  `racket_arc` = angular sweep over ±`hit_refine_window`.
- `racket_present[frame]` = bool (detection conf ≥ threshold).

### Ownership (`ownership_scorer.py`)
- `racket_motion_score` rewritten: uses `racket_motion` + `racket_shuttle_distance` at
  the hit frame instead of wrist/elbow/shoulder angular velocity. Same 0–1 output range
  and existing weight slot, so Viterbi calibration is unaffected.
- `normalized_proximity_score` gains an optional racket term (racket-head → shuttle
  distance) blended with the existing wrist-based term via new
  `racket_proximity_blend` (default 0.5).

### Rule-based classifier (`stroke_features.py`)
- `extract_clip_features` adds: `racket_contact_distance` (min racket-shuttle distance
  in clip), `racket_present_frac` (fraction of clip frames with a racket),
  `racket_peak_speed`.
- `classify_by_family` / `_build_evidence` use `racket_contact_distance` as the genuine
  contact cue (replacing the wrist-proxy `wrist_shuttle_distance` it currently leans on
  for contact height/zone).

### Hit refinement (`hits.py`)
- `_find_nearest_wrist_frame` extended to `_find_nearest_racket_frame`: among
  ±`hit_refine_window` frames, pick the one minimizing racket-shuttle distance; used as
  the 30% tiebreaker (currently wrist-only) in the combined direction-reversal +
  proximity score.

## Error Handling, Config, Testing (Section 3)

### Graceful degradation (racket detection is noisier than joints)
- `get_racket()` returns `None` (weights missing / ultralytics busy) → every consumer
  falls back to its **existing wrist-proxy behavior unchanged**. No break, no regression.
- Per-frame: no racket detection within `hit_refine_window` of a hit → hit refinement
  ignores the racket term, uses wrist/direction-reversal as today.
- Confidence gate `racket_min_conf` (default 0.4) suppresses low-quality boxes;
  missing frames treated as "racket absent" (not zero-distance).

### Settings (`config/settings.py`) — new tunable fields, no magic numbers
- `racket_enabled: bool = True`
- `racket_model_path` (absolute, like `yolov8_model_path`)
- `racket_min_conf: float = 0.4`
- `racket_proximity_blend: float = 0.5` (racket vs wrist in proximity score)
- `racket_motion_weight: float`, `racket_dist_weight: float` (within rewritten motion score)
- `racket_head_margin: float = 0.1` (where on bbox to take head point)
- Colab parity: same fields surfaced as form fields / CLI args in `colab/pipeline.py`.

### Tests (synthetic inputs, mocked models — mirror existing patterns)
- `test_racket.py`: tracker I/O shape, head-point extraction, graceful `None` fallback.
- `test_ownership_scorer.py`: updated — racket motion score vs wrist-proxy on a
  synthetic clip; blend weighting.
- `test_stroke_features.py`: racket contact feature present and consumed in evidence.
- `test_hits.py`: racket tiebreaker prefers racket-near frame when wrist ambiguous.
- `model_downloader.py`: add RacketDB download entry; verify path resolution.
- Full suite must stay green except the 2 known pre-existing failures
  (`test_colab_pipeline.py::test_colab_delegates_court_space_enrichment_to_backend_helper`,
  `test_colab_uses_continuity_aware_tracknet_candidate_selection`).

### Validation
Re-run Kaggle on phone footage; compare vs `labels_enriched.csv` baseline
(hit-frame median, within±8, stroke match, side). Expect:
- `owner_uncertain` rate to drop (racket gives a real contact signal).
- Racket-aware hit refinement to tighten the ~8-frame median label error.
- No regression when `racket_enabled=False` or weights absent.

## Out of Scope (explicit)
- Retraining BST / any stroke model with extended input dim.
- A separately-trained racket-trajectory classifier ensembled via `probs_matrix`
  (Scope B) — deferred.
- Shuttle detection changes (TrackNetV3 untouched).
