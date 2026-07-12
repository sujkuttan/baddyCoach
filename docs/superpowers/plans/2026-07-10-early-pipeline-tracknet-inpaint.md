# Early Pipeline TrackNet and InpaintNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the shipped TrackNet and InpaintNet checkpoints with their trained semantics while retaining observed points and rejecting invalid shuttle coordinates.

**Architecture:** TrackNet produces raw observations from a background-plus-eight-frame window. InpaintNet fills only original gaps. Shuttle cleanup stores explicit court rejection provenance without mutating pixel observations. The Colab wrapper mirrors these semantics.

**Tech Stack:** Python, PyTorch, NumPy, Pandas, OpenCV, pytest.

---

## File map

- `backend/app/models/tracknet.py`: model loading, TrackNet windows/decoding, masked InpaintNet repair.
- `backend/app/pipeline/shuttle.py`: court-space rejection and provenance.
- `backend/app/pipeline/pose.py`, `backend/app/pipeline/players.py`: bbox interpolation and logger fix.
- `colab/pipeline.py`: equivalent tracking and repair path.
- `backend/tests/test_tracknet.py`, `test_shuttle.py`, `test_pose.py`, `test_players.py`, `test_colab_pipeline.py`: regression tests.

### Task 1: Make InpaintNet checkpoint-compatible

**Files:** modify `backend/tests/test_tracknet.py`, `backend/app/models/tracknet.py`.

- [ ] Write failing tests asserting every key/shape in `ckpts/InpaintNet_best.pt["model"]` matches `InpaintNet().state_dict()`, and that `_rectify_trajectory([(10.,20.,.9), None, (30.,40.,.8)], 100, 100)` preserves both observed `(x,y)` pairs.
- [ ] Run `cd backend && python -m pytest tests/test_tracknet.py -k 'inpaintnet_checkpoint or rectification_preserves' -v`; confirm RED because current keys are `conv1…out` and every point is smoothed.
- [ ] Replace `InpaintNet` with the checkpoint architecture (`down_1`, `down_2`, `down_3`, `buttelneck`, `up_1`, `up_2`, `up_3`, `predictor`) and `forward(coords, mask)`. Normalize/strip checkpoint keys, require full key-and-shape compatibility, and disable the model on mismatch. Fill gaps only for network input; blend model output only where the original observation is missing.
- [ ] Run `cd backend && python -m pytest tests/test_tracknet.py -k inpaintnet -v`; confirm GREEN.
- [ ] Commit with `git add backend/app/models/tracknet.py backend/tests/test_tracknet.py && git commit -m "fix: load and mask InpaintNet trajectory repair"`.

### Task 2: Match TrackNet checkpoint inference semantics

**Files:** modify `backend/tests/test_tracknet.py`, `backend/app/models/tracknet.py`.

- [ ] Write failing tests that `_build_input(eight_frames, background)` returns 27 channels with the background occupying channels 0–2, and that component extraction chooses a 3×3 component over an isolated higher-valued pixel.
- [ ] Run `cd backend && python -m pytest tests/test_tracknet.py -k 'background_then_eight or component' -v`; confirm RED because the current API builds nine-frame windows and uses argmax.
- [ ] Read checkpoint `param_dict`, validate `seq_len == 8` and `bg_mode == "concat"`, build inputs `[background,f0…f7]`, apply triangular overlap weighting to per-frame sigmoid heatmaps, threshold each aggregate, and return the largest component centre scaled to original resolution. Preserve chunked inference.
- [ ] Run `cd backend && python -m pytest tests/test_tracknet.py -v`; confirm GREEN.
- [ ] Commit with `git add backend/app/models/tracknet.py backend/tests/test_tracknet.py && git commit -m "fix: use TrackNet checkpoint inference contract"`.

### Task 3: Reject invalid court-space points

**Files:** modify `backend/tests/test_shuttle.py`, `backend/app/pipeline/shuttle.py`, `backend/app/config/settings.py`.

- [ ] Write failing tests where an identity homography projects a point outside a configurable court margin and a consecutive point exceeds a configurable metres-per-second limit. Assert both get `court_rejected=True` and NaN court fields rather than clamped values.
- [ ] Run `cd backend && python -m pytest tests/test_shuttle.py -k court_enrichment -v`; confirm RED because the code clamps and has no rejection provenance.
- [ ] Add `shuttle_oob_margin_meters` and `shuttle_max_speed_mps`. Project without clamp; reject OOB points and later points whose consecutive valid court-space speed exceeds the limit; add `court_rejected` and leave raw `x/y` untouched.
- [ ] Run `cd backend && python -m pytest tests/test_shuttle.py -v`; confirm GREEN.
- [ ] Commit with `git add backend/app/config/settings.py backend/app/pipeline/shuttle.py backend/tests/test_shuttle.py && git commit -m "fix: reject invalid court-space shuttle points"`.

### Task 4: Correct pose fallback and player logging

**Files:** modify `backend/tests/test_pose.py`, `backend/tests/test_players.py`, `backend/app/pipeline/pose.py`, `backend/app/pipeline/players.py`.

- [ ] Write a failing pose test with player_1 bboxes at frames 0 and 2, asking for frame 1 and expecting `[5,5,15,15]`; write a player test that exercises `_run_yolov8` with a minimal fake tracker and no logger `TypeError`.
- [ ] Run `cd backend && python -m pytest tests/test_pose.py tests/test_players.py -k 'fallback_bbox_interpolates or live_yolo_logging' -v`; confirm RED.
- [ ] Interpolate between bracketing same-player boxes within `range_limit`, retaining nearest one-sided fallback. Convert both `logger.info` calls in `_run_yolov8` to a single positional message plus keyword fields.
- [ ] Run `cd backend && python -m pytest tests/test_pose.py tests/test_players.py -v`; confirm GREEN.
- [ ] Commit with `git add backend/app/pipeline/pose.py backend/app/pipeline/players.py backend/tests/test_pose.py backend/tests/test_players.py && git commit -m "fix: stabilize pose fallback and player logging"`.

### Task 5: Port the behavior to Colab and verify

**Files:** modify `colab/pipeline.py`, `backend/tests/test_colab_pipeline.py`.

- [ ] Write a failing source-level parity test asserting Colab has `seq_len=8`, background input construction, and masked repair semantics.
- [ ] Run `cd backend && python -m pytest tests/test_colab_pipeline.py -k background_contract -v`; confirm RED.
- [ ] Update Colab's TrackNet/InpaintNet path to match backend semantics while retaining its GPU batch sizing and progress messages.
- [ ] Run `cd backend && python -m pytest tests/test_tracknet.py tests/test_shuttle.py tests/test_pose.py tests/test_players.py tests/test_colab_pipeline.py -m 'not gpu and not model' -q`; confirm zero failures.
- [ ] Commit with `git add colab/pipeline.py backend/tests/test_colab_pipeline.py && git commit -m "fix: keep Colab shuttle inference in backend parity"`.
