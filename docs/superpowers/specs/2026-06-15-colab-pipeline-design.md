# Colab/Kaggle Pipeline Bundle — Design Spec

**Date:** 2026-06-15
**Status:** Approved

---

## 1. Overview

Create a GPU-accelerated pipeline that runs on Colab/Kaggle, outputs a JSON report, and can be loaded into the local UI for analysis. Solves the CPU-only performance problem by moving inference to cloud GPUs.

**Deliverables:**
1. `colab/pipeline.py` — Self-contained pipeline script (~300 lines)
2. `colab/BMCA_Colab.ipynb` — Colab notebook
3. Local UI "Load Report" button

---

## 2. Pipeline Script (`colab/pipeline.py`)

Fully self-contained Python script. No imports from `backend/app/`.

### Capabilities
- Downloads all model weights on first run
- Runs all 13 pipeline stages with GPU inference
- Outputs `report.json` with full analytics
- Prints inline summary (shot distribution, fatigue trend, top improvements)

### CLI Interface
```bash
python pipeline.py video.mp4 --output report.json --device cuda
```

### Model Downloads
| Model | Source | Size |
|-------|--------|------|
| TrackNetV3 | `ckpts/TrackNet_best.pt` (bundled or HuggingFace) | 136MB |
| YOLOv8s | ultralytics auto-download | 22MB |
| RTMPose | MMPose model zoo ONNX | 52MB |
| BST-CG-AP | Google Drive (gdown) | 7MB |

### Pipeline Stages (all GPU)
1. Court detection (hardcoded corners or homography)
2. Player tracking (YOLOv8s)
3. Shuttle tracking (TrackNetV3)
4. Pose estimation (RTMPose)
5. Hit frame localization
6. Stroke classification (BST)
7. Player attribution
8. Rally segmentation
9. Court position analytics
10. Footwork analytics
11. Fitness analytics
12. Tactical analytics
13. Technical analytics
14. Coach recommendations

### Output Format
Same JSON structure as the local backend `/api/jobs/{id}/report` endpoint. Compatible with the existing ReportView.

---

## 3. Colab Notebook (`colab/BMCA_Colab.ipynb`)

### Cell 1: Setup
```python
!pip install torch torchvision ultralytics onnxruntime gdown
!git clone https://github.com/user/baddyCoach.git || true
# Download model weights
```

### Cell 2: Upload
```python
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
```

### Cell 3: Run Pipeline
```python
!python baddyCoach/colab/pipeline.py {video_path} --output report.json
```

### Cell 4: Results
```python
import json
from IPython.display import display, JSON
report = json.load(open('report.json'))
# Display summary stats
# Download report
from google.colab import files
files.download('report.json')
```

---

## 4. Local UI "Load Report"

### Upload Screen Changes
- Add "Or load an existing report" link below the upload button
- Clicking opens a file picker for `.json` files
- Loads the JSON directly into ReportView (no job creation, no processing)

### ReportView Changes
- Accept `reportData` prop (direct JSON) as alternative to `jobId`
- When loaded from file, hide video player (no video available)
- Show "Imported Report" badge in header

### File Structure
- `frontend/src/views/UploadView.tsx` — Add file picker + loadReport callback
- `frontend/src/App.tsx` — Add `loadedReport` state for direct report loading
- `frontend/src/views/ReportView.tsx` — Support both jobId and direct data modes

---

## 5. Files to Create/Modify

| File | Action |
|------|--------|
| `colab/pipeline.py` | Create |
| `colab/BMCA_Colab.ipynb` | Create |
| `frontend/src/App.tsx` | Modify — add loadedReport state |
| `frontend/src/views/UploadView.tsx` | Modify — add Load Report button |
| `frontend/src/views/ReportView.tsx` | Modify — support direct data mode |

---

## 6. Non-Goals

- Kaggle kernel script (focus on Colab first)
- Batch processing multiple videos
- Video upload to cloud storage
- Model fine-tuning
