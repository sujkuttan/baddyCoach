# MMPose HRNet for BST Accuracy Experiment

## Goal

Replace RTMPose with MMPose HRNet for pose estimation to match the BST training pipeline, reducing unknown stroke classifications by improving keypoint quality.

## Context

BST was trained using MMPose (HRNet) for human pose estimation. We currently use RTMPose (a lightweight ONNX model from the same ecosystem). Both output COCO 17-keypoint format, but HRNet produces higher-quality keypoints that better match BST's training distribution. This mismatch is a likely contributor to the remaining ~10% regression from BST's theoretical accuracy ceiling.

## Architecture

### Current Pipeline (RTMPose)
```
Frame → YOLO bbox crop → RTMPose ONNX (256×192) → 17 keypoints (17, 3)
```

### Proposed Pipeline (MMPose HRNet)
```
Frame → YOLO bbox crop → HRNet ONNX (256×192) → 17 keypoints (17, 3)
```

The change is **minimal** — same input format (cropped bbox), same output format (COCO 17 keypoints), different ONNX model.

### Key Differences

| Aspect | RTMPose-M | HRNet-W32 |
|--------|-----------|-----------|
| Input size | 256×192 | 256×192 |
| Output | SimCC (2 outputs) | Direct heatmap (1 output) |
| Model size | 4MB | ~100MB |
| Keypoint accuracy | Good (optimized for speed) | Better (trained for accuracy) |
| Inference time (ONNX) | ~5ms/crop | ~15-25ms/crop |
| Dependencies | Standalone ONNX | Same ONNX runtime |

### Output Format Handling

RTMPose uses SimCC (similarity-based coordinate decoding) returning 2 outputs (simcc_x, simcc_y). HRNet uses heatmap-based decoding returning a single output (17, H, W). The `estimate_batch` method needs to handle both formats:

- RTMPose: `outputs[0]` = simcc_x (17, W*2), `outputs[1]` = simcc_y (17, H*2) → argmax per keypoint
- HRNet: `outputs[0]` = heatmap (1, 17, H/4, W/4) → argmax per heatmap channel → scale to input coords

## Implementation

### Changes to `colab/pipeline.py`

1. **Download HRNet ONNX** — Add HRNet-W32 ONNX download in `setup_models()` using MMPose's model export
   - Source: MMPose model zoo or pre-exported from HuggingFace
   - Path: `ckpts/mmpose/hrnet_w32_coco_256x192.onnx`

2. **Update `RTMPoseEstimator`** → rename to `PoseEstimator` (supports both models)
   - Auto-detect model type from output shape: 2 outputs = RTMPose SimCC, 1 output = HRNet heatmap
   - Add `_decode_heatmap()` method for HRNet output
   - Keep existing `_decode_simcc()` method for RTMPose

3. **Model selection** — CLI flag `--pose-model rtmpose|mmpose`
   - Default: `mmpose` (new default for accuracy)
   - `rtmpose` available for speed-critical use

4. **Batch inference** — `estimate_batch()` works for both models (preprocessing identical)

### Changes to `backend/app/models/rtmpose.py`

- Same auto-detection logic
- Model path config in `settings.py`

## Testing

1. Run pipeline with `--pose-model mmpose` on `videos/sample_5min_h264.mp4`
2. Compare pose.parquet output: row count should match, keypoint values will differ (expected)
3. Compare report.json: unknown count should decrease, stroke distribution should shift
4. Run pipeline with `--pose-model rtmpose` to verify baseline is unchanged
5. All 68 backend tests must pass

## Risk Factors

1. **HRNet ONNX export** — Need to export HRNet from MMPose to ONNX. MMPose has built-in export tools, but may require mmcv installation. Mitigation: use pre-exported ONNX from HuggingFace if available, or export in Colab during setup.
2. **Inference speed** — HRNet is 3-5x slower per crop. With batching, total RTMPose time increases from ~3s to ~10s on 3000 frames. Acceptable since total pipeline is ~5 min.
3. **Memory** — HRNet is ~100MB vs 4MB for RTMPose. Negligible on T4 (15GB).
4. **No improvement** — If unknowns don't decrease, the bottleneck is elsewhere (clipping strategy, normalization). Revert to RTMPose default.

## Success Criteria

- Unknown stroke count decreases by ≥3 compared to current baseline (20 → ≤17 on v1.0 data)
- All 68 backend tests pass
- Pipeline runs without errors on sample video
