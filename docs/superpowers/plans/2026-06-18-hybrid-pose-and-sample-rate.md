# Hybrid Pose Estimator + Sample Rate CLI Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a hybrid mode that uses MMPose for stroke classification (training-consistent) and RTMPose for hit detection/coaching analytics (better confidence), plus a `--sample-rate` CLI arg for frame sampling control.

**Architecture:** Add `--sample-rate` CLI arg (default 3 = 10fps). Add `--pose-model hybrid` mode that runs both pose estimators: MMPose strokes + RTMPose hit confidence + RTMPose rally segmentation. Store both pose outputs in separate accumulators and merge at analytics stages.

**Tech Stack:** Python, onnxruntime, numpy, pandas, argparse

---

## Files to Modify

| File | Changes |
|------|---------|
| `colab/pipeline.py` | Add `--sample-rate` arg, add `hybrid` mode logic, dual pose accumulators, merge results |
| `colab/BMCA_Colab.ipynb` | Update cell 3 to pass `--sample-rate` and `--pose-model hybrid` |

---

### Task 1: Add --sample-rate CLI argument

**Files:**
- Modify: `colab/pipeline.py:1785-1797`

- [ ] **Step 1: Add --sample-rate argument**

```python
# At line 1789, after --pose-model, add:
parser.add_argument("--sample-rate", "-s", type=int, default=0,
                    help="Frame sampling interval (0=auto for 10fps, 1=every frame, 2=every 2nd, etc.)")
```

- [ ] **Step 2: Pass sample_rate to run_pipeline**

At line 1797, change:
```python
run_pipeline(args.video, args.output, args.device, pose_model=args.pose_model)
```
to:
```python
run_pipeline(args.video, args.output, args.device, pose_model=args.pose_model, sample_rate=args.sample_rate)
```

- [ ] **Step 3: Update run_pipeline signature and sampling logic**

At line 1437, change signature:
```python
def run_pipeline(video_path, output_path="report.json", device="cuda", pose_model="rtmpose", sample_rate=0):
```

At line 1451, change sampling calculation:
```python
    if sample_rate > 0:
        sample_interval = sample_rate
    else:
        sample_interval = max(1, int(video_fps / 10))
    num_samples = total_frames // sample_interval
    target_fps = video_fps / sample_interval
    print(f"  Sampling: every {sample_interval} frames -> ~{num_samples} frames ({target_fps:.0f}fps)")
```

- [ ] **Step 4: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: add --sample-rate CLI arg for frame sampling control"
```

---

### Task 2: Add hybrid mode — dual pose estimators

**Files:**
- Modify: `colab/pipeline.py:1467-1483` (model loading)
- Modify: `colab/pipeline.py:1485-1535` (batch processing loop)

- [ ] **Step 1: Update model loading for hybrid mode**

Replace lines 1467-1483 with:
```python
    # Pose model selection
    pose_estimator = None
    pose_estimator_secondary = None

    if pose_model == "hybrid":
        # Hybrid: MMPose for strokes, RTMPose for hit confidence
        if HRNET_PATH.exists():
            print(f"  Using HYBRID mode: MMPose (strokes) + RTMPose (hits)")
            pose_estimator = RTMPoseEstimator(str(HRNET_PATH), device=device)
            rtmpose_path = str(RTMOPOSE_PATH_ALT if RTMOPOSE_PATH_ALT.exists() else RTMOPOSE_PATH)
            if not Path(rtmpose_path).exists():
                rtmpose_dir = CKPT_DIR / "rtmpose"
                onnx_files = list(rtmpose_dir.rglob("*.onnx"))
                if onnx_files:
                    rtmpose_path = str(onnx_files[0])
            pose_estimator_secondary = RTMPoseEstimator(rtmpose_path, device=device)
        else:
            print(f"  WARNING: HRNet not found, falling back to RTMPose only")
            pose_model = "rtmpose"

    if pose_model == "mmpose" and HRNET_PATH.exists():
        pose_path = str(HRNET_PATH)
        print(f"  Using MMPose HRNet-W32 (accurate)")
        pose_estimator = RTMPoseEstimator(pose_path, device=device)
    elif pose_model != "hybrid":
        pose_path = str(RTMOPOSE_PATH_ALT if RTMOPOSE_PATH_ALT.exists() else RTMOPOSE_PATH)
        if not Path(pose_path).exists():
            rtmpose_dir = CKPT_DIR / "rtmpose"
            onnx_files = list(rtmpose_dir.rglob("*.onnx"))
            if onnx_files:
                pose_path = str(onnx_files[0])
                print(f"  Found RTMPose at: {onnx_files[0]}")
            else:
                print(f"  WARNING: No RTMPose .onnx found in {rtmpose_dir}")
        print(f"  Using RTMPose (fast)")
        pose_estimator = RTMPoseEstimator(pose_path, device=device)

    print("  Models loaded")
```

- [ ] **Step 2: Add secondary pose accumulator**

At line 1488, after `all_pose = []`, add:
```python
    all_pose_secondary = []  # RTMPose pose data (for hybrid hit confidence)
```

- [ ] **Step 3: Update _process_batch call for hybrid mode**

At line 1515, change the _process_batch call to pass secondary estimator:
```python
                _process_batch(batch_frames, batch_global_indices, sample_idx - len(batch_frames),
                               tracker, tracknet, pose_estimator, device,
                               all_shuttle, all_det, all_pose, all_player_detections,
                               batch_count, total_batches,
                               pose_estimator_secondary=pose_estimator_secondary,
                               all_pose_secondary=all_pose_secondary)
```

Same change at line 1529 for the remaining frames batch.

- [ ] **Step 4: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: hybrid mode with dual pose estimators"
```

---

### Task 3: Update _process_batch to handle secondary pose estimator

**Files:**
- Modify: `colab/pipeline.py:_process_batch` function (around line 1740)

- [ ] **Step 1: Add secondary pose parameters to _process_batch**

Find the `_process_batch` function definition and update its signature:
```python
def _process_batch(batch_frames, batch_global_indices, batch_start_idx,
                   tracker, tracknet, pose_estimator, device,
                   all_shuttle, all_det, all_pose, all_player_detections,
                   batch_count, total_batches,
                   pose_estimator_secondary=None, all_pose_secondary=None):
```

- [ ] **Step 2: Add secondary pose estimation inside _process_batch**

After the primary pose estimation block (after `kps_batch = pose_estimator.estimate_batch(crops)` and the loop that appends to `all_pose`), add:
```python
        # Secondary pose estimation (for hybrid mode)
        if pose_estimator_secondary is not None and all_pose_secondary is not None:
            kps_secondary = pose_estimator_secondary.estimate_batch(crops)
            for j, (bbox, frame_idx) in enumerate(zip(player_bboxes, batch_global_indices)):
                all_pose_secondary.append({
                    "frame": frame_idx,
                    "player_id": f"player_{j+1}",
                    "keypoints": kps_secondary[j].tolist() if kps_secondary[j] is not None else np.zeros((17, 3)).tolist(),
                })
```

- [ ] **Step 3: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: secondary pose estimation in batch processing"
```

---

### Task 4: Merge hybrid results at analytics stages

**Files:**
- Modify: `colab/pipeline.py:1589-1640` (analytics stages)

- [ ] **Step 1: Use RTMPose hits for hybrid mode**

After line 1591 (`hits = stage_hits(all_shuttle)`), the hits are already from RTMPose shuttle tracking (both modes use the same shuttle data). No change needed here — RTMPose shuttle tracking provides the hit frames.

- [ ] **Step 2: Use MMPose strokes for hybrid mode**

At line 1597, the strokes use `all_pose` which is MMPose in hybrid mode. This is already correct — BST was trained on MMPose data. No change needed.

- [ ] **Step 3: Use RTMPose pose for fitness/coaching analytics**

At line 1622-1628, change footwork and fitness to use secondary (RTMPose) pose when available:

```python
    print("\n[10/14] Footwork analytics...")
    # Use RTMPose for footwork/fitness (better movement tracking)
    footwork_pose = all_pose_secondary if all_pose_secondary else all_pose
    footwork = stage_footwork(footwork_pose, shots)
    print("  Done")

    print("\n[11/14] Fitness analytics...")
    fitness = stage_fitness(footwork, rallies, shots)
    print("  Done")
```

- [ ] **Step 4: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: hybrid mode merges MMPose strokes + RTMPose fitness analytics"
```

---

### Task 5: Update CLI help and Colab notebook

**Files:**
- Modify: `colab/pipeline.py:1789-1790` (help text)
- Modify: `colab/BMCA_Colab.ipynb` (cell 3)

- [ ] **Step 1: Update --pose-model choices and help**

At line 1789-1790, change:
```python
    parser.add_argument("--pose-model", default="rtmpose", choices=["rtmpose", "mmpose", "hybrid"],
                        help="Pose model: rtmpose (fast), mmpose/hrnet (accurate), or hybrid (MMPose strokes + RTMPose hits)")
```

- [ ] **Step 2: Update Colab notebook cell 3**

Find the cell that runs the pipeline and update the command to include hybrid mode and sample rate options. The exact cell content depends on the current notebook state — update the pipeline invocation to show the new args.

- [ ] **Step 3: Commit**

```bash
git add colab/pipeline.py colab/BMCA_Colab.ipynb
git commit -m "docs: update CLI help text and Colab notebook for hybrid mode"
```

---

### Task 6: Test the implementation

**Files:**
- Test: `colab/pipeline.py` (manual test)

- [ ] **Step 1: Test --sample-rate arg**

```bash
cd /home/sujith/baddyCoach
PYTHONPATH=backend .venv/bin/python colab/pipeline.py --help
```

Verify `--sample-rate` appears in help output.

- [ ] **Step 2: Test hybrid mode with test video**

```bash
PYTHONPATH=backend .venv/bin/python colab/pipeline.py videos/test_match.mp4 --output results/hybrid_test.json --device cuda --pose-model hybrid --sample-rate 2
```

Verify:
- Both pose estimators load
- Output shows "HYBRID mode" message
- Report generated successfully

- [ ] **Step 3: Verify results contain expected data**

```bash
python -c "
import json
with open('results/hybrid_test.json') as f:
    r = json.load(f)
print(f'Strokes: {len(r.get(\"shots\", []))}')
print(f'Rallies: {len(r.get(\"rallies\", []))}')
print(f'Coach weaknesses: {len(r.get(\"coach\", {}).get(\"weaknesses\", []))}')
"
```

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: hybrid pose estimator + sample-rate CLI complete"
```
