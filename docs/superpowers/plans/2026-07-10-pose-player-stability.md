# Pose and Player Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep each player’s pose crop geometrically stable through short YOLO detection gaps and ensure live player-tracking logging cannot abort the pipeline.

**Architecture:** Pose estimation will resolve a missing bbox only from the same persistent player. When detections exist on both sides of a frame, it will linearly interpolate each bbox; at a video edge it will retain the nearest same-player bbox. Player tracking will pass structured fields to `PipelineLogger` rather than unsupported printf-style positional arguments. The Colab crop resolver will use the same same-side temporal interpolation rule without changing its streamed GPU batches.

**Tech Stack:** Python, NumPy, FastAPI pipeline artifacts, pytest.

---

## File map

- `backend/app/pipeline/pose.py`: deterministic same-player temporal bbox resolution before RTMPose inference.
- `backend/app/pipeline/players.py`: PipelineLogger-compatible ByteTrack diagnostics.
- `colab/pipeline.py`: streamed pose crop fallback with the same interpolation and same-side identity rule.
- `backend/tests/test_pose.py`: unit coverage for interpolation and one-sided fallback.
- `backend/tests/test_players.py`: regression coverage for live-YOLO diagnostics.
- `backend/tests/test_colab_pipeline.py`: source-level guard for Colab’s shared fallback contract.

### Task 1: Specify temporal bbox resolution in pose tests

**Files:**
- Modify: `backend/tests/test_pose.py`
- Modify: `backend/app/pipeline/pose.py:225-234`

- [ ] **Step 1: Write failing interpolation and edge-fallback tests**

  Add these tests to `backend/tests/test_pose.py`:

  ```python
  def test_fallback_bbox_interpolates_between_same_player_detections():
      lookup = {
          "player_1": {
              0: [0.0, 0.0, 10.0, 10.0],
              2: [10.0, 10.0, 20.0, 20.0],
          },
      }

      bbox = PoseEstimationStage._find_fallback_bbox(1, "player_1", lookup, range_limit=2)

      assert bbox == [5.0, 5.0, 15.0, 15.0]


  def test_fallback_bbox_uses_nearest_same_player_box_at_video_edge():
      lookup = {"player_1": {3: [10.0, 20.0, 30.0, 40.0]}}

      bbox = PoseEstimationStage._find_fallback_bbox(0, "player_1", lookup, range_limit=3)

      assert bbox == [10.0, 20.0, 30.0, 40.0]
  ```

- [ ] **Step 2: Run the focused tests to verify the interpolation test fails**

  Run: `cd backend && python -m pytest tests/test_pose.py -k fallback_bbox -v`

  Expected: `test_fallback_bbox_interpolates_between_same_player_detections` fails because the current lookup returns frame 2’s full bbox instead of the midpoint; the edge-fallback test passes or remains valid.

- [ ] **Step 3: Implement bracketed linear interpolation with nearest one-sided fallback**

  Replace `_find_fallback_bbox` in `backend/app/pipeline/pose.py` with the following implementation. It searches only `det_lookup[player_id]`, so a missing far-player crop can never be filled from the near player.

  ```python
  @staticmethod
  def _find_fallback_bbox(
      frame_idx: int,
      player_id: str,
      det_lookup: dict,
      range_limit: int = 10,
  ) -> list[float] | None:
      """Resolve a short same-player detection gap without crossing identities."""
      my_dets = det_lookup.get(player_id, {})
      if not my_dets:
          return None

      before = next(
          (frame for frame in range(frame_idx - 1, frame_idx - range_limit - 1, -1)
           if frame in my_dets),
          None,
      )
      after = next(
          (frame for frame in range(frame_idx + 1, frame_idx + range_limit + 1)
           if frame in my_dets),
          None,
      )

      if before is not None and after is not None:
          start = np.asarray(my_dets[before], dtype=np.float64)
          end = np.asarray(my_dets[after], dtype=np.float64)
          fraction = (frame_idx - before) / (after - before)
          return (start + fraction * (end - start)).tolist()
      if before is not None:
          return list(my_dets[before])
      if after is not None:
          return list(my_dets[after])
      return None
  ```

- [ ] **Step 4: Run the pose tests to verify the contract is green**

  Run: `cd backend && python -m pytest tests/test_pose.py -v`

  Expected: all pose tests pass.

- [ ] **Step 5: Commit the pose fallback change**

  ```bash
  git add backend/app/pipeline/pose.py backend/tests/test_pose.py
  git commit -m "fix: interpolate missing pose crop boxes"
  ```

### Task 2: Fix live YOLO logging without changing tracking output

**Files:**
- Modify: `backend/tests/test_players.py`
- Modify: `backend/app/pipeline/players.py:168-185`

- [ ] **Step 1: Write a regression test for `_run_yolov8` diagnostics**

  Add these local fake types and test to `backend/tests/test_players.py`:

  ```python
  class _FakeDetection:
      def __init__(self, bbox, confidence, track_id):
          self.bbox = bbox
          self.confidence = confidence
          self.track_id = track_id


  class _FakeTracker:
      def track_frames(self, frames):
          return {
              "frames": {
                  0: [_FakeDetection([1, 2, 11, 22], 0.9, 7)],
                  1: [_FakeDetection([2, 3, 12, 23], 0.8, 7)],
              },
          }


  def test_live_yolo_logging_uses_pipeline_logger_contract(monkeypatch):
      monkeypatch.setattr(
          "app.pipeline.shared.models.get_yolov8", lambda: _FakeTracker()
      )

      detections = PlayerTrackingStage()._run_yolov8([
          np.zeros((8, 8, 3), dtype=np.uint8),
          np.zeros((8, 8, 3), dtype=np.uint8),
      ])

      assert detections == [
          {"frame": 0, "bbox": [1, 2, 11, 22], "confidence": 0.9, "track_id": 7},
          {"frame": 1, "bbox": [2, 3, 12, 23], "confidence": 0.8, "track_id": 7},
      ]
  ```

- [ ] **Step 2: Run the regression test to verify it fails with `TypeError`**

  Run: `cd backend && python -m pytest tests/test_players.py -k live_yolo_logging -v`

  Expected: FAIL because `PipelineLogger.info()` accepts one positional `message` argument but `_run_yolov8` supplies printf substitution arguments.

- [ ] **Step 3: Replace positional logging arguments with structured fields**

  Replace the two logging calls in `PlayerTrackingStage._run_yolov8` with the following calls:

  ```python
  logger.info(
      "ByteTrack raw tracking",
      unique_ids=len(id_counts),
      frames_with_detections=n_frames_with_dets,
      detections=sum(id_counts.values()),
  )
  ```

  ```python
  logger.info(
      "ByteTrack fragmentation",
      short_lived_id_count=len(small_ids),
      id_detection_counts=dict(sorted(small_ids.items(), key=lambda item: item[1])),
  )
  ```

  Keep the existing `if id_counts` and `if small_ids` guards; do not alter the returned detection schema or ByteTrack behavior.

- [ ] **Step 4: Run all player-tracking tests**

  Run: `cd backend && python -m pytest tests/test_players.py -v`

  Expected: all player tests pass, including the live-YOLO logging regression.

- [ ] **Step 5: Commit the logging blocker fix**

  ```bash
  git add backend/app/pipeline/players.py backend/tests/test_players.py
  git commit -m "fix: keep player tracking diagnostics logger-safe"
  ```

### Task 3: Mirror the same-player fallback rule in Colab

**Files:**
- Modify: `backend/tests/test_colab_pipeline.py`
- Modify: `colab/pipeline.py:1404-1444`

- [ ] **Step 1: Add a source-level parity test**

  Add this test to `backend/tests/test_colab_pipeline.py`:

  ```python
  def test_colab_pose_fallback_preserves_player_identity():
      source = (Path(__file__).resolve().parents[2] / "colab/pipeline.py").read_text()

      assert "def _interpolate_pose_bbox" in source
      assert "before_bbox" in source
      assert "after_bbox" in source
      assert "(1.0 - fraction) * before_bbox + fraction * after_bbox" in source
  ```

- [ ] **Step 2: Run the parity test to verify it fails**

  Run: `cd backend && python -m pytest tests/test_colab_pipeline.py -k pose_fallback -v`

  Expected: FAIL because Colab only selects the nearest detection and can fall back to an arbitrary side when the frame has no detections.

- [ ] **Step 3: Add an identity-preserving streamed bbox helper and use it for missing crops**

  In `colab/pipeline.py`, place this helper immediately before `_process_batch`:

  ```python
  def _interpolate_pose_bbox(
      frame_idx: int,
      player_side: str,
      frame_indices: list[int],
      detections_by_frame: dict[int, list[dict]],
      range_limit: int = 10,
  ) -> list[float] | None:
      """Resolve a pose bbox from detections of the requested side only."""
      before_bbox = None
      after_bbox = None
      before_frame = None
      after_frame = None

      for candidate in reversed(frame_indices):
          if candidate >= frame_idx or frame_idx - candidate > range_limit:
              continue
          match = next(
              (d for d in detections_by_frame.get(candidate, []) if d.get("side") == player_side),
              None,
          )
          if match is not None:
              before_frame, before_bbox = candidate, np.asarray(match["bbox"], dtype=np.float64)
              break

      for candidate in frame_indices:
          if candidate <= frame_idx or candidate - frame_idx > range_limit:
              continue
          match = next(
              (d for d in detections_by_frame.get(candidate, []) if d.get("side") == player_side),
              None,
          )
          if match is not None:
              after_frame, after_bbox = candidate, np.asarray(match["bbox"], dtype=np.float64)
              break

      if before_bbox is not None and after_bbox is not None:
          fraction = (frame_idx - before_frame) / (after_frame - before_frame)
          return ((1.0 - fraction) * before_bbox + fraction * after_bbox).tolist()
      if before_bbox is not None:
          return before_bbox.tolist()
      if after_bbox is not None:
          return after_bbox.tolist()
      return None
  ```

  In `_process_batch`, replace the branch that searches arbitrary `best_det` values when `not dets_for_frame`. For each `("player_1", "near")` and `("player_2", "far")` pair, call `_interpolate_pose_bbox(global_idx, side, global_indices, all_det)` and append a crop only when the result is not `None`. When a frame has one detected side, use the helper only for the missing side; retain its detected bbox unchanged.

- [ ] **Step 4: Run Colab parity and focused backend regression tests**

  Run: `cd backend && python -m pytest tests/test_colab_pipeline.py tests/test_pose.py tests/test_players.py -m 'not gpu and not model' -v`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit the Colab parity change**

  ```bash
  git add colab/pipeline.py backend/tests/test_colab_pipeline.py
  git commit -m "fix: preserve player identity in Colab pose fallback"
  ```

### Task 4: Run the complete non-model regression suite and inspect the diff

**Files:** no code changes expected.

- [ ] **Step 1: Run formatting and the relevant non-hardware regression suite**

  Run: `git diff --check && cd backend && python -m pytest tests/test_pose.py tests/test_players.py tests/test_colab_pipeline.py -m 'not gpu and not model' -q`

  Expected: `git diff --check` produces no output and pytest reports zero failures.

- [ ] **Step 2: Inspect the final working tree before handoff**

  Run: `git status --short && git log --oneline -3`

  Expected: only the intended pose/player commits are new; preserve any pre-existing unrelated worktree changes.

