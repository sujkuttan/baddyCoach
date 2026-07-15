# BST Input Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve accepted-shot stroke accuracy by routing only evidence-supported clips into BST and preserving explicit, auditable abstentions for poor clips.

**Architecture:** A pure shared evaluator scores clip provenance before batching. `_build_clip()` records raw/repaired/interpolated/rejected shuttle state, pose coverage/confidence, and bounded bbox source gaps while preserving BST’s tensor shape. The stroke stage batches only eligible clips, creates `unknown` records for quality abstentions, and persists a complete per-shot audit trail; Colab inherits this backend stage and exports its new debug artifact.

**Tech Stack:** Python 3, NumPy, Pandas, Pydantic Settings, PyTorch BST inference, pytest.

---

## File map

- Create: `backend/app/pipeline/shared/bst_input_quality.py` — pure provenance evaluator and serializable quality result.
- Modify: `backend/app/config/settings.py` — all admission and normalization thresholds.
- Modify: `backend/app/pipeline/shared/bst_preproc.py` — confidence masking that also applies when a detection bbox is present.
- Modify: `backend/app/pipeline/strokes.py` — provenance collection, court-rejected shuttle zeroing, capped bbox interpolation, selective batching, routing, artifacts, and summary logging.
- Modify: `colab/pipeline.py` — copy `debug_bst_input_quality.parquet` with the existing backend-stage artifacts.
- Modify: `backend/scripts/evaluate_labels.py` — quality-stratified manual-label metrics.
- Create: `backend/tests/test_bst_input_quality.py` — evaluator unit tests.
- Modify: `backend/tests/test_bst.py` — detection-bbox confidence-masking regression.
- Modify: `backend/tests/test_strokes.py` — clip provenance and ineligible routing tests.
- Modify: `backend/tests/test_colab_pipeline.py` — Colab debug-artifact parity guard.
- Create: `backend/tests/test_evaluate_labels.py` — quality-stratified evaluation tests.

### Task 1: Add configurable, pure BST input-quality evaluation

**Files:**
- Create: `backend/app/pipeline/shared/bst_input_quality.py`
- Modify: `backend/app/config/settings.py:140-160`
- Create: `backend/tests/test_bst_input_quality.py`

- [ ] **Step 1: Write failing evaluator tests**

  Create `backend/tests/test_bst_input_quality.py` with these tests. The provenance sequences represent the unpadded part of a clip only.

  ```python
  import numpy as np

  from app.pipeline.shared.bst_input_quality import evaluate_bst_clip_quality


  def _provenance(**overrides):
      value = {
          "video_len": 20,
          "shuttle_observed": [True] * 12 + [False] * 7 + [True],
          "shuttle_repaired": [False] * 20,
          "shuttle_interpolated": [False] * 20,
          "shuttle_court_rejected": [False] * 20,
          "pose_present_far": [True] * 20,
          "pose_present_near": [True] * 20,
          "pose_keypoint_confidence_far": [0.9] * 20,
          "pose_keypoint_confidence_near": [0.9] * 20,
          "bbox_gap_far": [0] * 20,
          "bbox_gap_near": [0] * 20,
      }
      value.update(overrides)
      return value


  def test_quality_accepts_clip_with_sufficient_observed_shuttle_and_pose():
      result = evaluate_bst_clip_quality(_provenance())

      assert result["eligible"] is True
      assert result["score"] == 1.0
      assert result["reasons"] == []
      assert result["observed_shuttle_frames"] == 13
      assert result["max_shuttle_gap_frames"] == 7


  def test_quality_rejects_court_rejected_point_even_when_other_coverage_is_good():
      rejected = [False] * 20
      rejected[3] = True

      result = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))

      assert result["eligible"] is False
      assert result["score"] == 0.8
      assert result["reasons"] == ["court_rejected_shuttle"]


  def test_quality_accumulates_all_failed_hard_checks_and_clamps_score():
      result = evaluate_bst_clip_quality(_provenance(
          video_len=10,
          shuttle_observed=[False] * 10,
          shuttle_repaired=[False] * 10,
          shuttle_interpolated=[True] * 10,
          shuttle_court_rejected=[True] + [False] * 9,
          pose_present_far=[False] * 10,
          pose_present_near=[False] * 10,
          pose_keypoint_confidence_far=[0.1] * 10,
          pose_keypoint_confidence_near=[0.1] * 10,
          bbox_gap_far=[11] * 10,
          bbox_gap_near=[11] * 10,
      ))

      assert result["eligible"] is False
      assert result["score"] == 0.0
      assert result["reasons"] == [
          "clip_too_short",
          "low_observed_shuttle",
          "long_shuttle_gap",
          "court_rejected_shuttle",
          "low_pose_coverage",
          "low_keypoint_confidence",
          "long_bbox_gap",
          "low_quality_score",
      ]
  ```

- [ ] **Step 2: Run the evaluator tests to verify they fail**

  Run: `cd backend && python3 -m pytest tests/test_bst_input_quality.py -v`

  Expected: collection fails with `ModuleNotFoundError: No module named 'app.pipeline.shared.bst_input_quality'`.

- [ ] **Step 3: Add Settings fields and the evaluator implementation**

  Add these fields under the existing BST settings in `backend/app/config/settings.py`:

  ```python
  bst_input_quality_enabled: bool = True
  bst_min_clip_video_frames: int = 15
  bst_min_observed_shuttle_fraction: float = 0.35
  bst_max_raw_shuttle_gap_frames: int = 7
  bst_min_pose_coverage: float = 0.70
  bst_min_keypoint_confidence: float = 0.35
  bst_max_bbox_interp_gap: int = 10
  bst_quality_score_min: float = 0.70
  ```

  Create `backend/app/pipeline/shared/bst_input_quality.py` with the following implementation. Keep it independent of Pandas and model loading so synthetic provenance is easy to test.

  ```python
  import numpy as np

  from app.config.settings import settings


  def _longest_false_run(values: np.ndarray) -> int:
      longest = current = 0
      for value in values:
          if bool(value):
              current = 0
          else:
              current += 1
              longest = max(longest, current)
      return longest


  def _coverage(values: np.ndarray) -> float:
      return float(values.mean()) if len(values) else 0.0


  def _median_confidence(values: np.ndarray, present: np.ndarray) -> float:
      usable = values[present]
      return float(np.median(usable)) if len(usable) else 0.0


  def evaluate_bst_clip_quality(provenance: dict) -> dict:
      video_len = int(provenance["video_len"])

      def values(name, dtype):
          return np.asarray(provenance[name][:video_len], dtype=dtype)

      observed = values("shuttle_observed", bool)
      repaired = values("shuttle_repaired", bool)
      interpolated = values("shuttle_interpolated", bool)
      rejected = values("shuttle_court_rejected", bool)
      far_present = values("pose_present_far", bool)
      near_present = values("pose_present_near", bool)
      far_conf = values("pose_keypoint_confidence_far", float)
      near_conf = values("pose_keypoint_confidence_near", float)
      far_gaps = values("bbox_gap_far", float)
      near_gaps = values("bbox_gap_near", float)

      observed_fraction = _coverage(observed)
      max_shuttle_gap = _longest_false_run(observed)
      far_coverage = _coverage(far_present)
      near_coverage = _coverage(near_present)
      far_median_conf = _median_confidence(far_conf, far_present)
      near_median_conf = _median_confidence(near_conf, near_present)
      max_bbox_gap = int(max(np.max(far_gaps, initial=0), np.max(near_gaps, initial=0)))

      reasons = []
      score = 1.0
      if video_len < settings.bst_min_clip_video_frames:
          reasons.append("clip_too_short")
      if observed_fraction < settings.bst_min_observed_shuttle_fraction:
          reasons.append("low_observed_shuttle")
          score -= 0.35
      if max_shuttle_gap > settings.bst_max_raw_shuttle_gap_frames:
          reasons.append("long_shuttle_gap")
          score -= 0.25
      if rejected.any():
          reasons.append("court_rejected_shuttle")
          score -= 0.20
      if min(far_coverage, near_coverage) < settings.bst_min_pose_coverage:
          reasons.append("low_pose_coverage")
          score -= 0.20
      if min(far_median_conf, near_median_conf) < settings.bst_min_keypoint_confidence:
          reasons.append("low_keypoint_confidence")
          score -= 0.15
      if max_bbox_gap > settings.bst_max_bbox_interp_gap:
          reasons.append("long_bbox_gap")
          score -= 0.15

      score = float(np.clip(score, 0.0, 1.0))
      hard_failures = bool(reasons)
      if score < settings.bst_quality_score_min:
          reasons.append("low_quality_score")
      eligible = not hard_failures and score >= settings.bst_quality_score_min
      return {
          "eligible": eligible,
          "score": score,
          "reasons": reasons,
          "observed_shuttle_frames": int(observed.sum()),
          "repaired_shuttle_frames": int(repaired.sum()),
          "interpolated_shuttle_frames": int(interpolated.sum()),
          "court_rejected_shuttle_frames": int(rejected.sum()),
          "observed_shuttle_fraction": observed_fraction,
          "max_shuttle_gap_frames": max_shuttle_gap,
          "far_pose_coverage": far_coverage,
          "near_pose_coverage": near_coverage,
          "far_pose_median_confidence": far_median_conf,
          "near_pose_median_confidence": near_median_conf,
          "max_bbox_gap_frames": max_bbox_gap,
      }
  ```

- [ ] **Step 4: Run evaluator tests to verify they pass**

  Run: `cd backend && python3 -m pytest tests/test_bst_input_quality.py -v`

  Expected: all three evaluator tests pass.

- [ ] **Step 5: Commit the evaluator and thresholds**

  ```bash
  git add backend/app/config/settings.py backend/app/pipeline/shared/bst_input_quality.py backend/tests/test_bst_input_quality.py
  git commit -m "feat: score BST input quality"
  ```

### Task 2: Preserve provenance and prevent bad coordinates from entering a BST clip

**Files:**
- Modify: `backend/app/pipeline/shared/bst_preproc.py:19-79`
- Modify: `backend/app/pipeline/strokes.py:90-299`
- Modify: `backend/tests/test_bst.py:332-380`
- Modify: `backend/tests/test_strokes.py`

- [ ] **Step 1: Write failing preprocessing and clip-provenance tests**

  Add this test to `backend/tests/test_bst.py`:

  ```python
  def test_normalize_joints_masks_low_confidence_joint_with_detection_bbox():
      from app.pipeline.shared.bst_preproc import normalize_joints

      coords = np.full((17, 2), [50.0, 50.0], dtype=np.float32)
      coords[10] = [999.0, 999.0]
      confidence = np.ones(17, dtype=np.float32)
      confidence[10] = 0.1

      normalized = normalize_joints(
          coords, det_bbox=(0.0, 0.0, 100.0, 100.0), conf=confidence,
          min_confidence=0.35,
      )

      np.testing.assert_array_equal(normalized[10], [0.0, 0.0])
      assert np.any(normalized[9] != 0.0)
  ```

  Add this test to `backend/tests/test_strokes.py`:

  ```python
  def test_build_clip_zeros_court_rejected_shuttle_and_records_provenance():
      from app.pipeline.strokes import _build_clip

      frames = [0, 1, 2]
      shuttle = pd.DataFrame({
          "frame": frames,
          "x": [100.0, 200.0, 300.0],
          "y": [100.0, 200.0, 300.0],
          "confidence": [0.9, 0.9, 0.9],
          "was_interpolated": [False, True, False],
          "court_rejected": [False, True, False],
      })
      shuttle_raw = pd.DataFrame({
          "frame": frames,
          "x": [100.0, np.nan, 300.0],
          "y": [100.0, np.nan, 300.0],
          "confidence": [0.9, 0.0, 0.9],
          "was_repaired": [False, True, False],
      })
      keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
      pose = pd.DataFrame([
          {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}
          for frame in frames for player in ("player_1", "player_2")
      ])
      players = [
          {"id": "player_1", "side": "near", "detections": [
              {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames
          ]},
          {"id": "player_2", "side": "far", "detections": [
              {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames
          ]},
      ]

      clip = _build_clip(
          frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,
          player_detections=players, player_ids=["player_1", "player_2"],
          shuttle_raw=shuttle_raw,
      )

      np.testing.assert_array_equal(clip["shuttle"][1], [0.0, 0.0])
      assert clip["_bst_provenance"]["shuttle_observed"] == [True, False, True]
      assert clip["_bst_provenance"]["shuttle_repaired"] == [False, True, False]
      assert clip["_bst_provenance"]["shuttle_interpolated"] == [False, True, False]
      assert clip["_bst_provenance"]["shuttle_court_rejected"] == [False, True, False]
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  Run: `cd backend && python3 -m pytest tests/test_bst.py -k detection_bbox tests/test_strokes.py -k provenance -v`

  Expected: the normalization test fails because low-confidence coordinates are only masked when no detection bbox is supplied; the clip test fails because `_build_clip()` does not accept `shuttle_raw` or emit `_bst_provenance`.

- [ ] **Step 3: Apply confidence masking to both normalization paths**

  Change the `normalize_joints()` signature and mask application in `backend/app/pipeline/shared/bst_preproc.py`:

  ```python
  def normalize_joints(
      coords: np.ndarray,
      det_bbox: tuple | None = None,
      bbox_margin: float = 0.0,
      conf: np.ndarray | None = None,
      min_confidence: float = 0.35,
  ) -> np.ndarray:
      coords = np.asarray(coords, dtype=np.float64)
      invalid_mask = np.all(coords == 0.0, axis=1)
      if conf is not None:
          invalid_mask |= np.asarray(conf) < min_confidence
      # Keep the existing bbox selection and normalized-coordinate calculation.
      # When det_bbox is absent, use only ~invalid_mask to derive bbox_min/max.
      # After normalization in either path, force invalid joints to exactly zero.
      normalized[invalid_mask] = 0.0
      return normalized.astype(np.float32)
  ```

  Pass `min_confidence=settings.bst_min_keypoint_confidence` at the existing `_build_clip()` normalization call.

- [ ] **Step 4: Record bounded bbox gaps and clip provenance**

  Extend `_build_clip()` in `backend/app/pipeline/strokes.py` with the optional parameter below and update every call in `StrokeClassificationStage.run()` to pass the `shuttle_raw` parquet artifact:

  ```python
  def _build_clip(
      frames: list[int],
      shuttle_df: pd.DataFrame | None,
      pose_df: pd.DataFrame | None,
      vid_w: float,
      vid_h: float,
      court_length: float,
      court_width: float,
      seq_len: int,
      player_sides: dict | None = None,
      player_detections: dict | None = None,
      homography: np.ndarray | None = None,
      original_len: int | None = None,
      player_ids: list | None = None,
      shuttle_raw: pd.DataFrame | None = None,
  ) -> dict:
  ```

  Replace the local `_interpolate_bboxes()` return value with `(bbox, source_gap)` pairs. It must use this decision tree for each target frame:

  ```python
  if frame in lookup:
      return lookup[frame], 0
  if before and after and after_frame - before_frame <= settings.bst_max_bbox_interp_gap:
      ratio = (frame - before_frame) / (after_frame - before_frame)
      return interpolated_bbox, max(frame - before_frame, after_frame - frame)
  if before and frame - before_frame <= settings.bst_max_bbox_interp_gap:
      return lookup[before_frame], frame - before_frame
  if after and after_frame - frame <= settings.bst_max_bbox_interp_gap:
      return lookup[after_frame], after_frame - frame
  return None, settings.bst_max_bbox_interp_gap + 1
  ```

  Create a `_bst_provenance` dict containing the exact keys consumed by `evaluate_bst_clip_quality()`. For each unpadded frame, derive `shuttle_observed` from a finite, confidence-qualified `shuttle_raw` point with `was_repaired=False`; derive `shuttle_repaired` from `shuttle_raw.was_repaired`; derive `shuttle_interpolated` from cleaned `was_interpolated`; and derive `shuttle_court_rejected` from cleaned `court_rejected`. If the cleaned point is court-rejected, leave `shuttle[t]` at `[0.0, 0.0]` and do not project its pixel coordinates.

  Set `pose_present_*` only when at least one keypoint passes `bst_min_keypoint_confidence`; append each frame’s median passing keypoint confidence, or `0.0` for missing pose. Append the bounded bbox source gap for each far/near slot. Return the dict with:

  ```python
  "_bst_provenance": provenance,
  ```

- [ ] **Step 5: Run focused preprocessing and clip tests**

  Run: `cd backend && python3 -m pytest tests/test_bst.py -k 'normalize_joints' tests/test_strokes.py -k 'provenance or temporal_resample' -v`

  Expected: all selected tests pass.

- [ ] **Step 6: Commit provenance-safe clip construction**

  ```bash
  git add backend/app/pipeline/shared/bst_preproc.py backend/app/pipeline/strokes.py backend/tests/test_bst.py backend/tests/test_strokes.py
  git commit -m "fix: preserve BST clip input provenance"
  ```

### Task 3: Gate batching, preserve abstentions, and persist audit fields

**Files:**
- Modify: `backend/app/pipeline/strokes.py:470-710`
- Modify: `backend/tests/test_strokes.py`
- Modify: `backend/app/config/settings.py:140-160`

- [ ] **Step 1: Write failing stage-routing tests with a fake classifier**

  Add these helpers and test to `backend/tests/test_strokes.py`:

  ```python
  class _QualityGateClassifier:
      seq_len = 20

      def __init__(self):
          self.received = []

      def predict_from_clips(self, clips, **kwargs):
          self.received = clips
          results = [("smash", 0.9, 3, 0.5, 0.0, 0.0) for _ in clips]
          probs = np.zeros((len(clips), 25), dtype=np.float32)
          if len(clips):
              probs[:, 3] = 1.0
          return results, probs


  def test_stroke_stage_skips_ineligible_clip_and_persists_quality(monkeypatch, tmp_job_dir):
      from app.pipeline.shared import models

      classifier = _QualityGateClassifier()
      monkeypatch.setattr(models, "get_bst", lambda: classifier)
      monkeypatch.setattr("app.pipeline.strokes.settings.fusion_enabled", False)
      monkeypatch.setattr("app.pipeline.strokes.settings.hierarchical_enabled", False)
      monkeypatch.setattr("app.pipeline.strokes.settings.confusion_pair_enabled", False)
      monkeypatch.setattr("app.pipeline.strokes.settings.physics_gate_enabled", False)

      store = ArtifactStore(tmp_job_dir)
      store.set_parquet("hits", pd.DataFrame({"frame": [0, 30], "confidence": [0.9, 0.9]}))
      store.set_parquet("shuttle", pd.DataFrame({
          "frame": list(range(50)), "x": [100.0] * 50, "y": [100.0] * 50,
          "confidence": [0.9] * 50, "was_interpolated": [False] * 50,
          "court_rejected": [False] * 30 + [True] + [False] * 19,
      }))
      store.set_parquet("shuttle_raw", pd.DataFrame({
          "frame": list(range(50)), "x": [100.0] * 50, "y": [100.0] * 50,
          "confidence": [0.9] * 50, "was_repaired": [False] * 50,
      }))
      keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])
      store.set_parquet("pose", pd.DataFrame([
          {"frame": f, "player_id": p, "keypoints": keypoints.tolist()}
          for f in range(50) for p in ("player_1", "player_2")
      ]))
      store.set("court", {"court_length": 13.4, "court_width": 6.1})
      store.set("players", {"players": [
          {"id": "player_1", "side": "near", "detections": [
              {"frame": f, "bbox": [0, 0, 100, 100]} for f in range(50)
          ]},
          {"id": "player_2", "side": "far", "detections": [
              {"frame": f, "bbox": [200, 0, 300, 100]} for f in range(50)
          ]},
      ]})

      StrokeClassificationStage().run(store, StageConfig(debug_level=1))
      shots = store.get_parquet("shots").sort_values("frame").reset_index(drop=True)

      assert len(classifier.received) == 1
      assert shots.loc[0, "bst_input_route"] == "bst"
      assert shots.loc[1, "bst_input_route"] == "quality_abstain"
      assert shots.loc[1, "stroke_type"] == "unknown"
      assert shots.loc[1, "is_bst_fallback"] is True
      assert "court_rejected_shuttle" in shots.loc[1, "bst_input_quality_reasons"]
      assert store.get_parquet("debug_bst_input_quality") is not None
  ```

- [ ] **Step 2: Run the routing test to verify it fails**

  Run: `cd backend && python3 -m pytest tests/test_strokes.py -k quality_gate -v`

  Expected: FAIL because all clips are passed into `predict_from_clips()` and no `bst_input_route` or quality artifact exists.

- [ ] **Step 3: Evaluate every clip before batch inference and batch only eligible indices**

  Import `evaluate_bst_clip_quality` in `backend/app/pipeline/strokes.py`. After Phase 1 clip construction, evaluate every `clip["_bst_provenance"]` and keep the results in the same order as `all_clips`:

  ```python
  quality_records = [evaluate_bst_clip_quality(clip["_bst_provenance"]) for clip in all_clips]
  eligible_indices = [
      index for index, quality in enumerate(quality_records)
      if not settings.bst_input_quality_enabled or quality["eligible"]
  ]
  eligible_clips = [all_clips[index] for index in eligible_indices]
  ```

  Call `classifier.predict_from_clips()` only with `eligible_clips`. Rebuild full-length results with this default for every ineligible index:

  ```python
  abstained_result = ("unknown", 0.0, 0, 0.5, 0.0, 0.0)
  all_results = [abstained_result for _ in all_clips]
  probs_matrix = np.zeros((len(all_clips), 25), dtype=np.float32)
  ```

  Insert eligible results and probability rows back at their original indices. Keep a parallel map from original clip index to debug output so a skipped clip never reads another clip’s logits or rule evidence.

- [ ] **Step 4: Add audit fields, debug parquet, and safe downstream routing**

  When creating each `shot`, merge its `quality_records[i]` using these exact fields:

  ```python
  shot.update({
      "bst_input_eligible": bool(quality["eligible"]),
      "bst_input_quality_score": float(quality["score"]),
      "bst_input_quality_reasons": quality["reasons"],
      "bst_input_route": "bst" if i in eligible_indices else "quality_abstain",
      "bst_input_observed_shuttle_frames": quality["observed_shuttle_frames"],
      "bst_input_repaired_shuttle_frames": quality["repaired_shuttle_frames"],
      "bst_input_interpolated_shuttle_frames": quality["interpolated_shuttle_frames"],
      "bst_input_court_rejected_shuttle_frames": quality["court_rejected_shuttle_frames"],
      "bst_input_max_shuttle_gap_frames": quality["max_shuttle_gap_frames"],
      "bst_input_far_pose_coverage": quality["far_pose_coverage"],
      "bst_input_near_pose_coverage": quality["near_pose_coverage"],
      "bst_input_far_pose_median_confidence": quality["far_pose_median_confidence"],
      "bst_input_near_pose_median_confidence": quality["near_pose_median_confidence"],
      "bst_input_max_bbox_gap_frames": quality["max_bbox_gap_frames"],
  })
  ```

  For `debug_level >= 1`, persist `pd.DataFrame(quality_records)` as `debug_bst_input_quality`. Run MMAction2, context fusion, hierarchical refinement, and confusion-pair refinement only over `eligible_indices`; merge their updated probability rows back into `probs_matrix`. Continue to pass every shot through the existing physics ensemble. After it returns, set `bst_input_route="downstream_override"` only when a quality-abstained shot changed from `unknown` to a determinate stroke.

  Emit one keyword-argument `PipelineLogger.info()` summary:

  ```python
  logger.info(
      "BST input quality routing",
      total_clips=len(all_clips),
      bst_eligible=len(eligible_indices),
      quality_abstained=len(all_clips) - len(eligible_indices),
      abstention_reasons=str(Counter(
          reason for quality in quality_records for reason in quality["reasons"]
      )),
  )
  ```

- [ ] **Step 5: Run the stage tests and inspect the generated artifacts**

  Run: `cd backend && python3 -m pytest tests/test_strokes.py -v`

  Expected: all stroke-stage tests pass; the quality-gate test proves exactly one clip reached the fake classifier and the second shot persisted a quality abstention.

- [ ] **Step 6: Commit quality routing and audit output**

  ```bash
  git add backend/app/pipeline/strokes.py backend/tests/test_strokes.py
  git commit -m "feat: gate BST inference on clip quality"
  ```

### Task 4: Export quality evidence in Colab and evaluate it against labels

**Files:**
- Modify: `colab/pipeline.py:1215-1219`
- Modify: `backend/scripts/evaluate_labels.py:259-291`
- Modify: `backend/tests/test_colab_pipeline.py`
- Create: `backend/tests/test_evaluate_labels.py`

- [ ] **Step 1: Write failing Colab and evaluation tests**

  Add this source-level parity test to `backend/tests/test_colab_pipeline.py`:

  ```python
  def test_colab_exports_bst_input_quality_debug_artifact():
      source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

      assert '"debug_bst_input_quality"' in source
  ```

  Create `backend/tests/test_evaluate_labels.py`:

  ```python
  import pandas as pd

  from scripts.evaluate_labels import summarize_bst_input_quality


  def test_quality_summary_reports_accepted_accuracy_and_coverage():
      shots = pd.DataFrame({
          "stroke_type": ["smash", "drop", "lift"],
          "true_stroke": ["smash", "smash", "lift"],
          "bst_input_eligible": [True, True, False],
          "bst_input_quality_reasons": [[], [], ["long_shuttle_gap"]],
      })

      result = summarize_bst_input_quality(shots)

      assert result["total_labeled"] == 3
      assert result["eligible_labeled"] == 2
      assert result["coverage"] == 2 / 3
      assert result["accepted_accuracy"] == 0.5
      assert result["overall_accuracy"] == 2 / 3
      assert result["reason_counts"] == {"long_shuttle_gap": 1}
  ```

- [ ] **Step 2: Run the tests to verify they fail**

  Run: `cd backend && python3 -m pytest tests/test_colab_pipeline.py -k quality tests/test_evaluate_labels.py -v`

  Expected: the Colab source assertion fails because the new artifact is not copied; test collection fails because `summarize_bst_input_quality` does not exist.

- [ ] **Step 3: Copy the artifact in Colab and add quality-stratified evaluation**

  In the Colab debug-copy loop, replace the current list with:

  ```python
  for debug_key in ["debug_bst_outputs", "debug_bst_input_quality", "debug_hit_scores"]:
  ```

  Add this helper above `main()` in `backend/scripts/evaluate_labels.py`:

  ```python
  def summarize_bst_input_quality(shots: pd.DataFrame) -> dict:
      labeled = shots.dropna(subset=["true_stroke"]).copy()
      eligible = labeled[labeled["bst_input_eligible"].fillna(False)]
      correct = labeled["stroke_type"] == labeled["true_stroke"]
      accepted_correct = eligible["stroke_type"] == eligible["true_stroke"]
      reason_counts = {}
      for reasons in labeled.get("bst_input_quality_reasons", pd.Series(dtype=object)):
          for reason in reasons if isinstance(reasons, list) else []:
              reason_counts[reason] = reason_counts.get(reason, 0) + 1
      return {
          "total_labeled": len(labeled),
          "eligible_labeled": len(eligible),
          "coverage": len(eligible) / max(1, len(labeled)),
          "accepted_accuracy": float(accepted_correct.mean()) if len(eligible) else 0.0,
          "overall_accuracy": float(correct.mean()) if len(labeled) else 0.0,
          "reason_counts": reason_counts,
      }
  ```

  In `evaluate_enriched_csv()`, merge the matched label data and shots by match index, call `summarize_bst_input_quality()`, and return it under `metrics["bst_input_quality"]`. In `main()`, print `coverage`, `accepted_accuracy`, `overall_accuracy`, and the reason counts whenever that metrics block exists.

- [ ] **Step 4: Run the parity and evaluation tests**

  Run: `cd backend && python3 -m pytest tests/test_colab_pipeline.py tests/test_evaluate_labels.py -v`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit Colab export and evaluation reporting**

  ```bash
  git add colab/pipeline.py backend/scripts/evaluate_labels.py backend/tests/test_colab_pipeline.py backend/tests/test_evaluate_labels.py
  git commit -m "feat: report BST input quality against labels"
  ```

### Task 5: Verify the complete gate and establish the manual-label baseline

**Files:** no production code changes expected.

- [ ] **Step 1: Run the complete affected unit suite**

  Run: `cd backend && python3 -m pytest tests/test_bst_input_quality.py tests/test_bst.py tests/test_strokes.py tests/test_colab_pipeline.py tests/test_evaluate_labels.py -m 'not gpu and not model' -v`

  Expected: zero failures.

- [ ] **Step 2: Run static checks and inspect the committed diff**

  Run: `git diff --check master..HEAD && git status --short && git log --oneline master..HEAD`

  Expected: no whitespace errors, a clean worktree, and four focused feature commits.

- [ ] **Step 3: Run the baseline-versus-gate label report on a real completed job**

  Run: `cd backend && python3 scripts/evaluate_labels.py ../labels_enriched.csv`

  Expected: the report prints accepted-shot accuracy, overall accuracy, BST coverage, and per-reason abstention counts. Record this output beside the job artifacts before changing any threshold.

- [ ] **Step 4: Commit only if Step 3 changes a tracked evaluation fixture**

  Do not commit videos, job directories, parquet outputs, or generated label exports. If a deliberately tracked fixture is changed as part of reproducible evaluation, commit only that fixture with:

  ```bash
  git add <tracked-fixture-path>
  git commit -m "test: record BST input quality baseline"
  ```
