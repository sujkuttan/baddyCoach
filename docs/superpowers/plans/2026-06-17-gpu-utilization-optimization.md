# GPU Utilization Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase GPU utilization from ~3% (0.4/15 GB) to 60-80% for faster video processing on T4/A100 GPUs.

**Architecture:** Batch all per-frame model inferences (YOLO, TrackNet, RTMPose, BST) to process N frames in a single GPU call. Add FP16 mixed precision. Pipeline independent stages across batches using CUDA streams where possible.

**Tech Stack:** PyTorch, ONNX Runtime, Ultralytics YOLO, tqdm

---

## Current Bottleneck Analysis

| Model | Current Approach | GPU Util | Fix |
|-------|-----------------|----------|-----|
| YOLOv8 | Per-frame `model.track()` call | ~5% | Batch via `model.track(batch_tensor)` |
| TrackNet | Per-frame forward pass (9-frame window) | ~3% | Batch windows into single forward pass |
| RTMPose | Per-crop ONNX `model.run()` | ~2% | Batch crops into single ONNX run |
| BST | Per-clip inference (already batched) | ~15% | Add FP16 |
| All models | FP32 precision | 1x speed | FP16 for 1.5-2x speedup |

**File:** `colab/pipeline.py` (all changes in one file)

---

### Task 1: Batched YOLOv8 Tracking

**File:** `colab/pipeline.py:225-253`

Currently `track_batch` calls `self.model.track(frame, ...)` per-frame in a Python loop. YOLOv8 supports batched inference natively.

- [ ] **Step 1: Replace per-frame YOLO with batched inference**

Replace the `track_batch` method in `YOLOv8Tracker`:

```python
def track_batch(self, frames, global_frame_offsets):
    """Run YOLO tracking on all frames in batch for GPU efficiency."""
    all_det = {}
    if not frames:
        return all_det

    h, w = frames[0].shape[:2]
    
    # YOLOv8 supports batched input — pass all frames at once
    # Process in GPU-friendly chunks to avoid OOM
    CHUNK = 64
    for chunk_start in range(0, len(frames), CHUNK):
        chunk = frames[chunk_start:chunk_start + CHUNK]
        results = self.model.track(
            source=chunk,
            classes=[0],
            conf=self.conf,
            verbose=False,
            persist=True,
            device=self.device,
        )
        for local_offset, r in enumerate(results):
            local_idx = chunk_start + local_offset
            global_idx = global_frame_offsets + local_idx
            dets = []
            if r.boxes is not None and r.boxes.id is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    bbox_area = bw * bh
                    frame_area = w * h
                    if bbox_area < frame_area * 0.001 or bbox_area > frame_area * 0.5:
                        continue
                    dets.append({"frame": global_idx, "bbox": [x1, y1, x2, y2],
                               "confidence": box.conf[0].item(), "track_id": int(box.id[0].item())})
            dets.sort(key=lambda d: d["confidence"], reverse=True)
            dets = dets[:2]
            all_det[global_idx] = dets
    return all_det
```

- [ ] **Step 2: Verify by running pipeline on sample video**

Run: `python colab/pipeline.py videos/sample_5min_h264.mp4 --output results/gpu_test.json --device cuda`

Expected: YOLO tracking should be 3-5x faster than before.

---

### Task 2: Batched TrackNet Shuttle Tracking

**File:** `colab/pipeline.py:196-222`

Currently `predict_batch` iterates per-frame, building a 9-frame window and running a single forward pass each time. This is the biggest GPU waste — N frames = N separate GPU calls.

- [ ] **Step 1: Replace per-frame TrackNet with batched multi-window inference**

```python
def predict_batch(self, frames, original_size=None):
    """Run TrackNet on all frames using batched GPU inference."""
    import torch
    
    if self.model is None or len(frames) < 3:
        return [{"x": 0, "y": 0, "confidence": 0}] * len(frames)

    ow = original_size[0] if original_size else frames[0].shape[1]
    oh = original_size[1] if original_size else frames[0].shape[0]
    results = []

    # Build all windows at once (numpy), then batch on GPU
    all_windows = []
    for i in range(len(frames)):
        window = frames[max(0, i - 8):i + 1]
        while len(window) < 9:
            window.insert(0, window[0])
        processed = []
        for f in window[-9:]:
            r = cv2.resize(f, (self.input_width, self.input_height))
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            processed.append(r)
        all_windows.append(np.stack(processed).reshape(self.input_height, self.input_width, 27))

    # Batch process in GPU-friendly chunks
    CHUNK = 256
    for chunk_start in range(0, len(all_windows), CHUNK):
        chunk = all_windows[chunk_start:chunk_start + CHUNK]
        batch = np.stack(chunk).transpose(0, 3, 1, 2)  # (B, 27, H, W)
        tensor = torch.from_numpy(batch).float().to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        heatmaps = 1 / (1 + np.exp(-out.cpu().numpy()[:, 0]))
        for j in range(len(chunk)):
            hm = heatmaps[j]
            y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
            results.append({
                "x": float(x_idx * ow / self.input_width),
                "y": float(y_idx * oh / self.input_height),
                "confidence": float(hm.max()),
            })

    return results
```

- [ ] **Step 2: Run pipeline to verify shuttle tracking accuracy unchanged**

Run: `python colab/pipeline.py videos/sample_5min_h264.mp4 --output results/gpu_test.json --device cuda`

Verify: shuttle positions in `results/debug/shuttle.parquet` match previous run within floating point tolerance.

---

### Task 3: Batched RTMPose Pose Estimation

**File:** `colab/pipeline.py:256-308` and `colab/pipeline.py:1571-1602`

Currently `RTMPoseEstimator.estimate()` processes one crop at a time via ONNX Runtime. The `_process_batch` function calls it in a per-frame, per-player loop.

- [ ] **Step 1: Add batch inference method to RTMPoseEstimator**

```python
def estimate_batch(self, crops):
    """Run RTMPose on multiple crops in a single ONNX call.
    
    Args:
        crops: list of (bbox_tuple, frame) pairs
    Returns:
        list of (17, 3) keypoint arrays
    """
    if self.model is None:
        return [np.random.rand(17, 3).astype(np.float32) for _ in crops]

    batch_tensors = []
    valid_indices = []
    crop_infos = []  # (x1, y1, crop_w, crop_h) for each valid crop

    for i, (bbox, frame) in enumerate(crops):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        r = cv2.resize(crop, (self.w, self.h))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = (r - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        batch_tensors.append(r.transpose(2, 0, 1).astype(np.float32))
        valid_indices.append(i)
        crop_infos.append((x1, y1, x2 - x1, y2 - y1))

    if not batch_tensors:
        return [np.zeros((17, 3), dtype=np.float32) for _ in crops]

    batch_np = np.stack(batch_tensors)  # (N, 3, H, W)
    outputs = self.model.run(None, {"input": batch_np})

    kps_all = []
    for j in range(len(batch_np)):
        x1, y1, crop_w, crop_h = crop_infos[j]
        if len(outputs) == 2:
            simcc_x = outputs[0][j]
            simcc_y = outputs[1][j]
            x_coords = np.argmax(simcc_x, axis=1) / 2.0
            y_coords = np.argmax(simcc_y, axis=1) / 2.0
            x_conf = np.max(simcc_x, axis=1)
            y_conf = np.max(simcc_y, axis=1)
            conf = (x_conf + y_conf) / 2.0
            kps = np.zeros((17, 3), dtype=np.float32)
            kps[:, 0] = x1 + x_coords * (crop_w / self.w)
            kps[:, 1] = y1 + y_coords * (crop_h / self.h)
            kps[:, 2] = 1.0 / (1.0 + np.exp(-conf))
        else:
            out = outputs[0][j]
            kps = out.reshape(17, 3) if out.ndim == 3 else out[0]
            kps[:, 0] = x1 + kps[:, 0] * crop_w
            kps[:, 1] = y1 + kps[:, 1] * crop_h
        kps_all.append(kps)

    # Fill in results for all crops (including empty ones)
    results = [np.zeros((17, 3), dtype=np.float32) for _ in crops]
    for j, idx in enumerate(valid_indices):
        results[idx] = kps_all[j]
    return results
```

- [ ] **Step 2: Refactor `_process_batch` to collect all crops first, then batch-estimate**

In `_process_batch`, replace the per-frame RTMPose loop:

```python
    # 3. Pose estimation (RTMPose) — batch all crops
    crop_list = []  # list of (global_idx, player_id, bbox, frame)
    for local_idx, global_idx in enumerate(global_indices):
        frame = frames[local_idx]
        dets_for_frame = all_det.get(global_idx, [])
        if not dets_for_frame:
            # Closest-in-time fallback
            for pid in ["player_1", "player_2"]:
                best_det = None
                best_dist = float('inf')
                for other_idx in range(max(0, local_idx - 10), min(len(global_indices), local_idx + 10)):
                    other_global = global_indices[other_idx]
                    for d in all_det.get(other_global, []):
                        dist = abs(other_idx - local_idx)
                        if dist < best_dist:
                            best_dist = dist
                            best_det = d
                if best_det:
                    crop_list.append((global_idx, pid, best_det["bbox"], frame))
            continue
        tid_to_pid = {}
        for d in dets_for_frame[:2]:
            tid = d.get("track_id", 0)
            if tid not in tid_to_pid:
                tid_to_pid[tid] = f"player_{len(tid_to_pid)+1}"
        for d in dets_for_frame[:2]:
            pid = tid_to_pid.get(d.get("track_id", 0), "player_1")
            crop_list.append((global_idx, pid, d["bbox"], frame))

    tqdm.write(f"{tag} | RTMPose batch pose estimation ({len(crop_list)} crops)...")
    BATCH_CHUNK = 128
    for crop_chunk_start in range(0, len(crop_list), BATCH_CHUNK):
        chunk = crop_list[crop_chunk_start:crop_chunk_start + BATCH_CHUNK]
        crops = [(c[2], c[3]) for c in chunk]
        kps_batch = pose_estimator.estimate_batch(crops)
        for j, (global_idx, pid, _, _) in enumerate(chunk):
            all_pose.append({"frame": global_idx, "player_id": pid, "keypoints": kps_batch[j].tolist()})
```

- [ ] **Step 3: Run pipeline to verify pose output matches**

Verify: `results/debug/pose.parquet` row count and keypoint values match previous run.

---

### Task 4: FP16 Mixed Precision for BST Classification

**File:** `colab/pipeline.py:474-560` (BST loading and inference in `stage_strokes`)

BST runs in FP32. FP16 halves memory and ~1.5-2x speeds up inference on T4 (which has tensor cores).

- [ ] **Step 1: Add FP16 inference to BST model loading and prediction**

In the BST loading section (~line 480), after `model.to(device).eval()`, add half-precision:

```python
            model.load_state_dict(state_dict)
            model.to(device).eval()
            if device == "cuda":
                model = model.half()
                print(f"  BST_CG loaded (FP16): in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
            else:
                print(f"  BST_CG loaded (FP32): in_dim={in_dim}, seq_len={seq_len}, n_classes={n_classes}")
```

In the BST inference section (~line 530), ensure inputs are also FP16:

```python
                JnB_t = torch.from_numpy(clip_data['JnB']).float().unsqueeze(0).to(device)
                shuttle_t = torch.from_numpy(clip_data['shuttle']).float().unsqueeze(0).to(device)
                pos_t = torch.from_numpy(clip_data['pos']).float().unsqueeze(0).to(device)
                video_len_t = torch.tensor([clip_data['video_len']], dtype=torch.long).to(device)
                
                if device == "cuda":
                    JnB_t = JnB_t.half()
                    shuttle_t = shuttle_t.half()
                    pos_t = pos_t.half()
```

- [ ] **Step 2: Run pipeline and verify BST predictions unchanged**

Verify: stroke classifications in report.json match previous run.

---

### Task 5: FP16 for TrackNet

**File:** `colab/pipeline.py:133-222`

- [ ] **Step 1: Add FP16 support to TrackNetV3 model loading**

After `self.model.to(device).eval()`, add:

```python
            self.model.to(device).eval()
            if device == "cuda":
                self.model = self.model.half()
```

In `predict_batch`, cast the input tensor to match:

```python
        tensor = torch.from_numpy(batch).float().to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
```

- [ ] **Step 2: Verify shuttle tracking accuracy unchanged**

---

### Task 6: Increase Batch Size for T4

**File:** `colab/pipeline.py:1273`

Currently `BATCH_SIZE = 2000`. With batching and FP16, the actual GPU memory per batch drops significantly. Increase to fill more of the 15GB T4.

- [ ] **Step 1: Increase BATCH_SIZE and add chunk sizes**

```python
BATCH_SIZE = 4000  # Total frames per batch (was 2000)
YOLO_CHUNK = 64    # YOLO processes 64 frames at once
TRACKNET_CHUNK = 256  # TrackNet processes 256 windows at once
RTMPOSE_CHUNK = 128   # RTMPose processes 128 crops at once
```

- [ ] **Step 2: Add GPU memory monitoring**

Add after each batch in `_process_batch`:

```python
    if device == "cuda":
        import torch
        used_mb = torch.cuda.memory_allocated() / 1024 / 1024
        tqdm.write(f"  GPU memory: {used_mb:.0f} MB allocated")
```

- [ ] **Step 3: Run pipeline on full sample video and measure total time**

Expected: 2-4x speedup over current baseline. GPU utilization should jump to 60-80%.

---

### Task 7: Pipeline Parallelism (Optional — Higher Complexity)

Run YOLO on batch N while TrackNet processes batch N-1 using CUDA streams.

- [ ] **Step 1: Implement double-buffering in main loop**

This is a more advanced optimization. Implement only if Tasks 1-6 don't reach target utilization. The approach:
1. Read batch N+1 from disk while batch N runs on GPU
2. Run YOLO on batch N while TrackNet finishes batch N-1
3. Use `torch.cuda.Stream()` for concurrent execution

Estimated benefit: 15-25% additional speedup on top of batching.

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| GPU RAM usage | 0.4/15 GB (3%) | 8-12/15 GB (53-80%) |
| YOLO tracking | ~518 fps (per-frame) | ~2000+ fps (batched) |
| TrackNet | ~30 fps (per-frame) | ~1500+ fps (batched) |
| RTMPose | ~50 crops/sec (per-crop) | ~800+ crops/sec (batched) |
| Total 5-min video | ~5 min | ~1.5-2 min |
| BST inference | FP32 | FP16 (1.5-2x faster) |

## Risk Factors

1. **YOLO batch tracking**: `model.track()` with batch input may reset track IDs between chunks. Verify persistent tracking works across chunks.
2. **RTMPose batching**: ONNX Runtime batch inference may have a max batch size. Start with chunks of 128 and tune.
3. **FP16 numerical issues**: BST transformer attention can be sensitive to FP16. If accuracy drops, fall back to `torch.cuda.amp.autocast` (dynamic FP16) instead of full FP16.
4. **Memory OOM**: If T4 OOMs at BATCH_SIZE=4000, reduce to 3000 and increase YOLO_CHUNK to 32.
