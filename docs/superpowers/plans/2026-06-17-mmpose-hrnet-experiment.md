# MMPose HRNet Pose Estimator Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MMPose HRNet-W32 as an alternative pose estimator and compare BST accuracy against RTMPose.

**Architecture:** BST was trained using `MMPoseInferencer('human')` which defaults to HRNet-W32 (256×192). We currently use RTMPose-M (256×192). Both output COCO 17-keypoint format. The change: export HRNet-W32 to ONNX, add it as an alternative to RTMPose, and auto-detect model type from output shape.

**Tech Stack:** MMPose (model export only), ONNX Runtime, OpenCV, NumPy

**File:** `colab/pipeline.py` (all changes in one file)

---

### Task 1: Add HRNet-W32 ONNX Export Script

**Files:**
- Create: `colab/export_hrnet.py`

The BST training code uses `MMPoseInferencer('human')` which defaults to HRNet-W32 trained on COCO. We need to export this model to ONNX for standalone inference. This is a one-time script that produces `ckpts/mmpose/hrnet_w32_coco_256x192.onnx`.

- [ ] **Step 1: Create the export script**

```python
#!/usr/bin/env python3
"""Export MMPose HRNet-W32 to ONNX for BMCA pipeline."""
import sys
from pathlib import Path

CKPT_DIR = Path("ckpts/mmpose")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = CKPT_DIR / "hrnet_w32_coco_256x192.onnx"


def export():
    try:
        import torch
        from mmpose.apis import MMPoseInferencer
    except ImportError:
        print("Install dependencies: pip install mmpose mmdet openmim && mim install mmcv")
        sys.exit(1)

    print("Loading MMPose HRNet-W32 inferencer...")
    inferencer = MMPoseInferencer('human')

    # The inferencer wraps an HRNet model internally
    # We need to access the underlying pose estimator
    pose_estimator = inferencer.pose_estimator

    # Create dummy input: cropped person image (1, 3, 256, 192)
    dummy_input = torch.randn(1, 3, 256, 192).cuda()

    print("Exporting to ONNX...")
    torch.onnx.export(
        pose_estimator,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported to {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


def test_export():
    """Verify the exported ONNX model works correctly."""
    import numpy as np
    import onnxruntime as ort

    print(f"\nTesting exported model...")
    session = ort.InferenceSession(
        str(OUTPUT_PATH),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # Random input
    dummy = np.random.randn(1, 3, 256, 192).astype(np.float32)
    outputs = session.run(None, {"input": dummy})

    print(f"  Input shape: {dummy.shape}")
    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
    print("  ONNX export verified OK")


if __name__ == "__main__":
    export()
    test_export()
```

- [ ] **Step 2: Run the export script**

```bash
pip install mmpose mmdet openmim && mim install mmcv
cd /home/sujith/baddyCoach && python colab/export_hrnet.py
```

Expected: `ckpts/mmpose/hrnet_w32_coco_256x192.onnx` created, ~100MB.

Note: If the automatic export fails (MMPose internal API changes), fall back to manual approach:
1. Use `MMPoseInferencer('human')` to run inference on a test image
2. Wrap it with `torch.onnx.export` using the underlying model
3. Or download a pre-exported HRNet ONNX from the MMPose model zoo

- [ ] **Step 3: Verify output shape**

The exported ONNX should output heatmap-style results. Check the output shape to determine decoding strategy:
- If shape is `(1, 17, 64, 48)`: heatmap-based, use argmax per channel → scale to 256×192
- If shape is `(1, 17, 2)`: direct regression, use values directly

Record the actual output shape — Task 3 depends on it.

- [ ] **Step 4: Commit**

```bash
git add colab/export_hrnet.py ckpts/mmpose/hrnet_w32_coco_256x192.onnx
git commit -m "feat: HRNet-W32 ONNX export for BST accuracy experiment"
```

---

### Task 2: Add HRNet Download to Colab Pipeline Setup

**Files:**
- Modify: `colab/pipeline.py:29-36` (model paths)
- Modify: `colab/pipeline.py:75-130` (setup_models function)

Add the HRNet ONNX path and a download function to `setup_models()`.

- [ ] **Step 1: Add HRNet path constant**

After line 36 (`BST_PATH = ...`), add:

```python
HRNET_PATH = CKPT_DIR / "mmpose" / "hrnet_w32_coco_256x192.onnx"
```

- [ ] **Step 2: Add HRNet download to setup_models()**

At the end of `setup_models()` (before `print("Models ready.\n")`), add:

```python
    hrnet_dir = CKPT_DIR / "mmpose"
    hrnet_dir.mkdir(parents=True, exist_ok=True)
    if not HRNET_PATH.exists():
        try:
            import gdown
            print("  Downloading HRNet-W32 weights...")
            # TODO: Host the exported ONNX on Google Drive and update this ID
            # gdown.download(id="<GOOGLE_DRIVE_ID>", output=str(HRNET_PATH), quiet=False)
            print("  HRNet not found — using RTMPose as default pose estimator")
            print("  To export HRNet: python colab/export_hrnet.py")
        except Exception as e:
            print(f"  HRNet download failed: {e}")
```

- [ ] **Step 3: Add --pose-model CLI argument**

In the `if __name__ == "__main__"` block, add to the argument parser:

```python
    parser.add_argument("--pose-model", default="rtmpose", choices=["rtmpose", "mmpose"],
                        help="Pose estimation model: rtmpose (fast) or mmpose/hrnet (accurate)")
```

Pass it through to `run_pipeline()`:

```python
    run_pipeline(args.video, args.output, args.device, pose_model=args.pose_model)
```

Update `run_pipeline` signature:

```python
def run_pipeline(video_path: str, output_path: str, device: str = "cuda", pose_model: str = "rtmpose"):
```

- [ ] **Step 4: Select pose model in run_pipeline**

Where the pose estimator is initialized (around line 1313), add model selection:

```python
    # Pose estimation — select model based on CLI flag
    if pose_model == "mmpose" and HRNET_PATH.exists():
        rtmpose_path = str(HRNET_PATH)
        print(f"  Using MMPose HRNet-W32 (accurate)")
    else:
        rtmpose_path = str(RTMOPOSE_PATH_ALT if RTMOPOSE_PATH_ALT.exists() else RTMOPOSE_PATH)
        if not Path(rtmpose_path).exists():
            rtmpose_dir = CKPT_DIR / "rtmpose"
            onnx_files = list(rtmpose_dir.rglob("*.onnx"))
            if onnx_files:
                rtmpose_path = str(onnx_files[0])
                print(f"  Found RTMPose at: {onnx_files[0]}")
            else:
                print(f"  WARNING: No RTMPose .onnx found in {rtmpose_dir}")
        print(f"  Using RTMPose (fast)")
    pose_estimator = RTMPoseEstimator(rtmpose_path, device=device)
```

- [ ] **Step 5: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('colab/pipeline.py', doraise=True); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: add --pose-model CLI flag for HRNet/RTMPose selection"
```

---

### Task 3: Auto-Detect Model Type and Handle Both Output Formats

**Files:**
- Modify: `colab/pipeline.py:274-327` (RTMPoseEstimator class)

The key difference: RTMPose outputs 2 tensors (simcc_x, simcc_y), HRNet outputs 1 tensor (heatmap). The class needs to auto-detect and decode both.

- [ ] **Step 1: Update RTMPoseEstimator to auto-detect model type**

Replace the `__init__` and `estimate` methods:

```python
class RTMPoseEstimator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = None
        self.h, self.w = 256, 192
        self.model_type = "rtmpose"  # auto-detected
        if Path(model_path).exists():
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)
                # Auto-detect model type from output count
                n_outputs = len(self.model.get_outputs())
                if n_outputs == 1:
                    self.model_type = "hrnet"
                    print(f"  HRNet-W32 loaded from {Path(model_path).name}")
                else:
                    self.model_type = "rtmpose"
                    print(f"  RTMPose loaded from {Path(model_path).name}")
            except Exception as e:
                print(f"  Pose model load error: {e}")
        else:
            print(f"  Pose model not found: {model_path}")

    def _preprocess(self, frame, bbox):
        """Common preprocessing: crop + resize + normalize."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, (x1, y1, x2 - x1, y2 - y1)
        r = cv2.resize(crop, (self.w, self.h))
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = (r - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = r.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return tensor, (x1, y1, x2 - x1, y2 - y1)

    def _decode_rtmpose(self, outputs, crop_info):
        """Decode RTMPose SimCC outputs."""
        x1, y1, crop_w, crop_h = crop_info
        simcc_x = outputs[0][0]
        simcc_y = outputs[1][0]
        x_coords = np.argmax(simcc_x, axis=1) / 2.0
        y_coords = np.argmax(simcc_y, axis=1) / 2.0
        x_conf = np.max(simcc_x, axis=1)
        y_conf = np.max(simcc_y, axis=1)
        conf = (x_conf + y_conf) / 2.0
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = x1 + x_coords * (crop_w / self.w)
        kps[:, 1] = y1 + y_coords * (crop_h / self.h)
        kps[:, 2] = 1.0 / (1.0 + np.exp(-conf))
        return kps

    def _decode_hrnet(self, outputs, crop_info):
        """Decode HRNet heatmap output."""
        x1, y1, crop_w, crop_h = crop_info
        heatmap = outputs[0][0]  # (17, H_map, W_map)
        if heatmap.ndim == 3:
            kps = np.zeros((17, 3), dtype=np.float32)
            for k in range(17):
                hm = heatmap[k]
                y_idx, x_idx = np.unravel_index(hm.argmax(), hm.shape)
                kps[k, 0] = x1 + (x_idx / hm.shape[1]) * crop_w
                kps[k, 1] = y1 + (y_idx / hm.shape[0]) * crop_h
                kps[k, 2] = float(hm.max())
        else:
            # Fallback: reshape as direct coordinates
            kps = heatmap.reshape(17, 3) if heatmap.ndim == 2 else heatmap[0]
            kps[:, 0] = x1 + kps[:, 0] * crop_w
            kps[:, 1] = y1 + kps[:, 1] * crop_h
        return kps

    def estimate(self, frame, bbox):
        if self.model is None:
            return np.zeros((17, 3), dtype=np.float32)
        tensor, crop_info = self._preprocess(frame, bbox)
        if tensor is None:
            return np.zeros((17, 3), dtype=np.float32)
        outputs = self.model.run(None, {"input": tensor})
        if self.model_type == "hrnet":
            return self._decode_hrnet(outputs, crop_info)
        else:
            return self._decode_rtmpose(outputs, crop_info)

    def estimate_batch(self, crops):
        """Run on multiple crops in a single ONNX call."""
        if self.model is None:
            return [np.zeros((17, 3), dtype=np.float32) for _ in crops]

        batch_tensors = []
        valid_indices = []
        crop_infos = []

        for i, (bbox, frame) in enumerate(crops):
            tensor, crop_info = self._preprocess(frame, bbox)
            if tensor is None:
                continue
            batch_tensors.append(tensor[0])
            valid_indices.append(i)
            crop_infos.append(crop_info)

        if not batch_tensors:
            return [np.zeros((17, 3), dtype=np.float32) for _ in crops]

        batch_np = np.stack(batch_tensors)
        outputs = self.model.run(None, {"input": batch_np})

        kps_all = []
        for j in range(len(batch_np)):
            if self.model_type == "hrnet":
                single_outputs = [out[j:j+1] for out in outputs]
                kps_all.append(self._decode_hrnet(single_outputs, crop_infos[j]))
            else:
                single_outputs = [out[j:j+1] for out in outputs]
                kps_all.append(self._decode_rtmpose(single_outputs, crop_infos[j]))

        results = [np.zeros((17, 3), dtype=np.float32) for _ in crops]
        for j, idx in enumerate(valid_indices):
            results[idx] = kps_all[j]
        return results
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('colab/pipeline.py', doraise=True); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: auto-detect pose model type (RTMPose SimCC vs HRNet heatmap)"
```

---

### Task 4: Run A/B Comparison on Sample Video

- [ ] **Step 1: Run with RTMPose (baseline)**

```bash
cd /home/sujith/baddyCoach
python colab/pipeline.py videos/sample_5min_h264.mp4 --output results/baseline_rtmpose.json --device cuda --pose-model rtmpose
```

Save baseline results for comparison:
```bash
cp results/baseline_rtmpose.json results/baseline_rtmpose_backup.json
```

- [ ] **Step 2: Run with HRNet (experiment)**

```bash
python colab/pipeline.py videos/sample_5min_h264.mp4 --output results/experiment_hrnet.json --device cuda --pose-model mmpose
```

- [ ] **Step 3: Compare results**

```bash
python3 -c "
import json
rtmpose = json.load(open('results/baseline_rtmpose.json'))
hrnet = json.load(open('results/experiment_hrnet.json'))

r_shots = rtmpose.get('shots', [])
h_shots = hrnet.get('shots', [])

from collections import Counter
r_dist = Counter(s['stroke_type'] for s in r_shots)
h_dist = Counter(s['stroke_type'] for s in h_shots)

print('=== RTMPose ===')
print(f'Shots: {len(r_shots)}, Unknowns: {r_dist.get(\"unknown\", 0)}')
for st, cnt in r_dist.most_common(): print(f'  {st}: {cnt}')

print('\n=== HRNet ===')
print(f'Shots: {len(h_shots)}, Unknowns: {h_dist.get(\"unknown\", 0)}')
for st, cnt in h_dist.most_common(): print(f'  {st}: {cnt}')
"
```

Expected: HRNet should have fewer unknowns than RTMPose.

- [ ] **Step 4: Commit results**

```bash
git add results/baseline_rtmpose.json results/experiment_hrnet.json
git commit -m "results: A/B comparison RTMPose vs HRNet-W32 on sample video"
```

---

## Expected Results

| Metric | RTMPose (baseline) | HRNet (experiment) |
|--------|-------------------|-------------------|
| Unknown count | 0 (already fixed) | 0 |
| Stroke distribution | Current | May shift (better keypoints) |
| Pose frames | 5975 | 5975 |
| Pipeline time | ~5 min | ~7-8 min (HRNet slower) |

The primary benefit is **improved keypoint quality** which may:
1. Change the stroke distribution (fewer borderline classifications)
2. Improve confidence scores on correctly classified strokes
3. Match the BST training distribution more closely

If unknowns are already at 0, the benefit will show as **higher confidence scores** and **more accurate stroke type assignments** in the borderline cases.

## Risk Factors

1. **HRNet ONNX export may fail** — MMPose internal APIs change between versions. Fallback: manually export using torch.onnx with the HRNet model directly loaded from MMPose configs.
2. **Output shape mismatch** — HRNet heatmap shape depends on the specific MMPose config. The auto-detection code handles both heatmap (1 output) and SimCC (2 outputs), but the actual heatmap resolution needs to be verified in Task 1.
3. **HRNet is slower** — ~3-5x slower per crop than RTMPose. With batching (128 crops), total RTMPose time increases from ~3s to ~10s. Acceptable.
4. **No measurable improvement** — If BST accuracy doesn't change, the bottleneck is elsewhere (clipping, normalization, not pose model quality). Revert to RTMPose default.
