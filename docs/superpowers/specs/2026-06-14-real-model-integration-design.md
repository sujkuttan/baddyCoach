# Real ML Model Integration — Design Spec

**Version:** 1.0
**Date:** 2026-06-14
**Status:** Design Approved — Ready for Implementation

---

## 1. Overview

Integrate actual ML models into the BMCA pipeline to replace mock data with real inference. This will enable accurate shuttle tracking, player detection, pose estimation, and stroke classification.

**Goal:** Process real match videos through the full pipeline with actual ML model inference.

---

## 2. Models to Integrate

### 2.1 TrackNetV3 — Shuttle Tracking
- **Source:** `ckpts/TrackNet_best.pt` (136MB)
- **Input:** 3 consecutive frames (BGR, 1280x720 or resized)
- **Output:** Heatmap → shuttle position (x, y) + confidence
- **Usage:** Sliding window of 3 frames, predict shuttle position per frame

### 2.2 YOLOv8 — Player Detection
- **Source:** Ultralytics pretrained `yolov8n.pt` (auto-downloaded)
- **Input:** Video frame (any resolution)
- **Output:** Person bounding boxes (class 0 = person)
- **Usage:** Detect 2 players per frame, track across frames

### 2.3 RTMPose — Pose Estimation
- **Source:** MMPose ONNX `rtmpose-m_8xb64-270e_coco-256x192.onnx`
- **Input:** Cropped player region (256x192)
- **Output:** 17 COCO keypoints (x, y, confidence)
- **Usage:** Estimate pose for each detected player per frame

### 2.4 BST-CG-AP — Stroke Classification
- **Source:** Google Drive (ShuttleSet, 25 classes, seq_len=100)
- **Input:** Normalized joints + shuttle trajectory + player position
- **Output:** Stroke type classification (25 classes)
- **Usage:** Classify each rally segment into stroke types

---

## 3. Pipeline Changes

### 3.1 Video Frame Extraction
- Add frame extraction utility to read video into numpy arrays
- Extract frames at configured FPS (default: 30fps)
- Store frames in memory for model inference

### 3.2 Updated Pipeline Stages

| Stage | Current | New |
|-------|---------|-----|
| Court Detection | Manual corners | Manual corners (unchanged) |
| Player Detection | Mock detections | YOLOv8 inference |
| Shuttle Tracking | Mock data | TrackNetV3 inference |
| Pose Estimation | Mock data | RTMPose inference |
| Hit Frame Localization | Multi-signal fusion | Multi-signal fusion (unchanged) |
| Stroke Classification | Random classification | BST-CG-AP inference |
| Player Attribution | Shuttle position | Shuttle position (unchanged) |
| Rally Segmentation | Frame gaps | Frame gaps (unchanged) |

### 3.3 Model Loading
- Load models once at startup, not per-request
- Store in a global registry or dependency injection
- Support CPU and GPU inference
- Handle model download for missing weights

---

## 4. Implementation Plan

### Task 1: Download and Setup Models
- Download BST weights from Google Drive
- Download RTMPose ONNX model from MMPose
- Verify all model files exist

### Task 2: Update TrackNetV3 Wrapper
- Fix inference to use sliding window of 3 frames
- Add proper preprocessing (resize, normalize)
- Add postprocessing (heatmap → coordinates)

### Task 3: Update YOLOv8 Wrapper
- Already functional with ultralytics
- Add person tracking across frames (simple IOU-based)

### Task 4: Update RTMPose Wrapper
- Fix ONNX inference with proper input shape (256x192)
- Add preprocessing (crop, resize, normalize)
- Add postprocessing (coordinates → keypoints)

### Task 5: Update BST Wrapper
- Implement proper input normalization
- Add sequence padding/truncation to seq_len=100
- Load real weights

### Task 6: Update Pipeline Stages
- `ShuttleTrackingStage`: Run TrackNetV3 on video frames
- `PlayerTrackingStage`: Run YOLOv8 + tracking
- `PoseEstimationStage`: Run RTMPose on player crops
- `StrokeClassificationStage`: Run BST on normalized inputs

### Task 7: Update API Pipeline Runner
- Extract frames from uploaded video
- Pass frames through pipeline stages
- Handle model errors gracefully

### Task 8: Testing
- Test with sample video
- Verify end-to-end flow
- Check output quality

---

## 5. Normalization Requirements (BST)

### 5.1 Shuttlecock Trajectory
```python
# Normalize by video resolution
x_normalized = x / video_width
y_normalized = y / video_height
# Range: [0, 1]
```

### 5.2 Joint Keypoints
```python
# Normalize by bounding box diagonal distance
bbox_diagonal = sqrt((x2-x1)^2 + (y2-y1)^2)
x_normalized = (keypoint_x - bbox_x1) / bbox_diagonal
y_normalized = (keypoint_y - bbox_y1) / bbox_diagonal
# Optional: center_align=True (center of bbox as origin)
```

### 5.3 Player Position
```python
# Feet position in court coordinates
# Normalized by court boundary
x_normalized = (x - court_left) / (court_right - court_left)
y_normalized = (y - court_top) / (court_bottom - court_top)
# Range: [0, 1]
```

---

## 6. Error Handling

- If a model fails on a frame, use interpolation from neighboring frames
- If player detection fails (< 2 players), mark frame as invalid
- If pose estimation fails, use zero keypoints
- Log all model errors for debugging

---

## 7. Performance Considerations

- **TrackNetV3:** ~30ms/frame on CPU (batch of 3 frames)
- **YOLOv8n:** ~50ms/frame on CPU
- **RTMPose:** ~30ms/frame on CPU
- **BST:** ~10ms per rally segment
- **Total:** ~110ms/frame → ~9 FPS processing speed
- **1-hour video (30fps):** ~10 minutes processing time

---

## 8. Dependencies

- `ultralytics>=8.0.0` (YOLOv8) — already installed
- `onnxruntime>=1.16.0` (RTMPose) — already installed
- `torch>=2.1.0` (TrackNetV3, BST) — already installed
- `mmpose` (optional, for model download) — not required

---

## 9. File Changes

| File | Change |
|------|--------|
| `backend/app/models/tracknet.py` | Fix inference, add preprocessing |
| `backend/app/models/yolov8.py` | Add tracking, improve detection |
| `backend/app/models/rtmpose.py` | Fix ONNX inference |
| `backend/app/models/bst.py` | Add normalization, load weights |
| `backend/app/pipeline/shuttle.py` | Run TrackNetV3 on frames |
| `backend/app/pipeline/players.py` | Run YOLOv8 + tracking |
| `backend/app/pipeline/pose.py` | Run RTMPose on crops |
| `backend/app/pipeline/strokes.py` | Run BST on normalized data |
| `backend/app/api/routes.py` | Extract frames, run pipeline |
| `backend/config/settings.py` | Add model paths |

---

## 10. Success Criteria

1. ✅ Shuttle tracking produces real trajectory (not random)
2. ✅ Player detection identifies 2 players per frame
3. ✅ Pose estimation produces valid 17-keypoint skeletons
4. ✅ Stroke classification predicts real stroke types
5. ✅ Rally segmentation finds natural breaks between rallies
6. ✅ End-to-end pipeline completes without errors
