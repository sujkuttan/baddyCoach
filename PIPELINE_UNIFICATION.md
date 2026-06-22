# Pipeline Unification Plan

## Overview
This document provides a comprehensive plan for unifying the Colab pipeline (`colab/pipeline.py`) and Backend pipeline (`backend/app/pipeline/`) to eliminate code duplication and ensure consistency.

## Current State
- **Colab Pipeline:** `colab/pipeline.py` - 3,395 lines, standalone implementation
- **Backend Pipeline:** `backend/app/pipeline/` - Modular implementation with 13 stages

## Key Differences
1. **Code Structure:** Colab is a single monolithic file, Backend is modular
2. **Model Loading:** Different model loading mechanisms
3. **Configuration:** Different configuration approaches
4. **Error Handling:** Different error handling strategies
5. **Logging:** Different logging approaches

## Unification Strategy

### Phase 1: Analysis and Preparation (Week 1)

#### 1.1 Code Analysis
**Objective:** Analyze colab/pipeline.py to identify reusable components.

**Key Shared Components Found:**

1. **Core Infrastructure:**
   - `_get_gpu_batch_config()` - GPU batch configuration
   - `setup_models()` - Model loading
   - `get_video_info()` - Video info extraction
   - `frame_generator()` - Frame extraction
   - `detect_court_from_frame()` - Court detection

2. **Geometric Processing:**
   - `_correct_court_points()` - Court point correction
   - `_validate_court_geometry()` - Court geometry validation
   - `compute_homography()` - Homography computation
   - `image_to_court()` - Image to court transformation
   - `make_undistorter()` - Undistorter creation
   - `foot_midpoint_from_pose()` - Foot midpoint calculation
   - `foot_point_from_bbox()` - Foot point from bbox
   - `compute_court_homography()` - Court homography computation

3. **Pipeline Stages:**
   - `stage_court_detection()` - Court detection
   - `stage_hits()` - Hit detection
   - `stage_strokes()` - Stroke classification
   - `stage_attribution()` - Player attribution
   - `stage_rallies()` - Rally segmentation
   - `stage_court_position()` - Court position analytics
   - `stage_footwork()` - Footwork analytics
   - `stage_fitness()` - Fitness analytics
   - `stage_tactical()` - Tactical analytics
   - `stage_technical()` - Technical analytics
   - `stage_rally_stats()` - Rally statistics

4. **Helper Functions:**
   - `_rule_based_shuttle_predict()` - Rule-based shuttle prediction
   - `_infer_end_reason()` - Infer rally end reason
   - `_is_rally_ending_shot()` - Check if shot ends rally
   - `_evaluate_shot()` - Evaluate shot

### Phase 2: Create Shared Modules (Week 2)

#### 2.1 Create Shared Directory Structure
```bash
mkdir -p backend/app/pipeline/shared
mkdir -p backend/app/pipeline/shared/core
mkdir -p backend/app/pipeline/shared/utils
mkdir -p backend/app/pipeline/shared/models
mkdir -p backend/app/pipeline/shared/logging
```

#### 2.2 Implement Core Shared Modules

**Create `backend/app/pipeline/shared/core.py`:**
```python
"""
Core shared functionality for both colab and backend pipelines.
"""

# Import all shared modules
from .utils import *
from .models import *
from .logging import *

# Common constants
COURT_LENGTH = 13.4
COURT_WIDTH = 5.18
NET_HEIGHT = 1.55

# Court model definition
COURT_MODEL = {
    "outer_tl": (0.0, 0.0),
    "outer_tr": (0.0, COURT_WIDTH),
    "outer_bl": (COURT_LENGTH, 0.0),
    "outer_br": (COURT_LENGTH, COURT_WIDTH),
}

# Stroke classes
STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]

# Rules definition (simplified)
RULES = []

# GPU batch configuration
def _get_gpu_batch_config(device: str) -> dict:
    """Detect GPU VRAM and return optimal batch sizes per pipeline stage."""
    tiers = [
        (12, {"yolo_chunk": 1000, "yolo_batch": 64, "tracknet_chunk": 128, "rtmpose_chunk": 256, "bst_batch": 128}),
        (6,  {"yolo_chunk": 500,  "yolo_batch": 32, "tracknet_chunk": 64,  "rtmpose_chunk": 128, "bst_batch": 64}),
        (2,  {"yolo_chunk": 200,  "yolo_batch": 16, "tracknet_chunk": 16,  "rtmpose_chunk": 64,  "bst_batch": 32}),
        (0,  {"yolo_chunk": 100,  "yolo_batch": 8,  "tracknet_chunk": 8,   "rtmpose_chunk": 32,  "bst_batch": 16}),
    ]
    cpu_cfg = {"yolo_chunk": 100, "yolo_batch": 8, "tracknet_chunk": 8, "rtmpose_chunk": 32, "bst_batch": 16}
    if "cuda" not in device.lower():
        return dict(cpu_cfg)
    try:
        import torch
        if not torch.cuda.is_available():
            return dict(cpu_cfg)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        for min_gb, cfg in tiers:
            if vram_gb >= min_gb:
                return dict(cfg)
    except Exception:
        pass
    return dict(cpu_cfg)
```

**Create `backend/app/pipeline/shared/utils.py`:**
```python
"""
Utility functions shared by both colab and backend pipelines.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """Get video information (width, height, fps)."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def frame_generator(video_path: str, sample_interval: int = 3, target_fps: int = 10) -> List[np.ndarray]:
    """Generate frames from video with specified sampling interval."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def detect_court_from_frame(frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Detect court corners from a single frame."""
    # Implementation would be in shared court module
    pass

def compute_homography(image_corners: List[List[int]], min_points: int = 4) -> Optional[np.ndarray]:
    """Compute homography matrix from image corners."""
    # Implementation would be in shared court module
    pass

def image_to_court(H: np.ndarray, uv: Tuple[float, float]) -> Tuple[float, float]:
    """Project image point to court coordinates."""
    # Implementation would be in shared court module
    pass
```

**Create `backend/app/pipeline/shared/models.py`:**
```python
"""
Model loading and management shared by both colab and backend pipelines.
"""

from pathlib import Path
from typing import Optional
import torch
import gdown
import zipfile
import os

def setup_models(device: str, pose_model: str = "rtmpose") -> dict:
    """
    Set up all models for the pipeline.
    
    Args:
        device: Device to use ('cuda' or 'cpu')
        pose_model: Pose model to use ('rtmpose' or 'mmpose')
    
    Returns:
        Dictionary of loaded models
    """
    # Implementation would be in shared models module
    pass

def _download_model_from_gdown(url: str, output_path: Path) -> bool:
    """Download model from Google Drive."""
    try:
        gdown.download(id=url, output=str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def _extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False
```

**Create `backend/app/pipeline/shared/logging.py`:**
```python
"""
Structured logging shared by both colab and backend pipelines.
"""

import logging
from typing import Dict, Any

class PipelineLogger:
    """Logger for pipeline operations."""
    
    def __init__(self, name: str = "pipeline"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

# Global logger instance
logger = PipelineLogger()
```

### Phase 3: Update Backend Pipeline (Week 3)

#### 3.1 Update Backend Pipeline Structure
**Update `backend/app/pipeline/__init__.py`:**
```python
"""
Backend pipeline package.
"""

from .shared.core import *
from .shared.utils import *
from .shared.models import *
from .shared.logging import *

# Export key components
__all__ = [
    # Core components
    'COURT_LENGTH',
    'COURT_WIDTH',
    'NET_HEIGHT',
    'COURT_MODEL',
    'STROKE_CLASSES',
    'RULES',
    '_get_gpu_batch_config',
    
    # Utility functions
    'get_video_info',
    'frame_generator',
    'detect_court_from_frame',
    'compute_homography',
    'image_to_court',
    'make_undistorter',
    'foot_midpoint_from_pose',
    'foot_point_from_bbox',
    'compute_court_homography',
    
    # Model loading
    'setup_models',
    '_download_model_from_gdown',
    '_extract_zip',
    
    # Logging
    'logger',
]
```

**Refactor existing pipeline stages to use shared modules:**
- Update `court.py` to use shared court module
- Update `players.py` to use shared tracking module
- Update `shuttle.py` to use shared preprocessing module
- Update `pose.py` to use shared validation module
- Update `hits.py` to use shared utils module
- Update `rallies.py` to use shared utils module
- Update `attribution.py` to use shared validation module
- Update `strokes.py` to use shared utils module

### Phase 4: Update Colab Pipeline (Week 4)

#### 4.1 Update Colab Pipeline Structure
**Update `colab/pipeline.py`:**
- Import shared modules
- Refactor components to use shared modules
- Remove duplicate code
- Maintain standalone functionality

#### 4.2 Refactor Components
**Update colab components to use shared modules:**
- Replace custom model loading with shared model module
- Replace custom utility functions with shared utils module
- Replace custom configuration with shared config module
- Replace custom error handling with shared logging module

### Phase 5: Testing and Validation (Week 5)

#### 5.1 Create Golden Regression Tests
**Create tests to ensure both pipelines produce identical results:**
- Define test scenarios that should produce identical results
- Create test suite for shared modules
- Create integration tests for both pipelines
- Create performance tests

#### 5.2 Validate Results
**Validate that unification is successful:**
- Compare outputs of both pipelines
- Ensure no regressions
- Document any differences
- Measure performance impact

## Implementation Benefits

### Technical Benefits:
1. **Reduced Code Duplication:** Eliminate ~2,000+ lines of duplicate code
2. **Improved Maintainability:** Single source of truth for pipeline logic
3. **Enhanced Consistency:** Both pipelines behave identically
4. **Better Testing:** Comprehensive test coverage for shared components
5. **Easier Debugging:** Common code makes debugging easier

### Operational Benefits:
1. **Faster Development:** New features only need to be implemented once
2. **Better Documentation:** Shared modules have clear documentation
3. **Improved Reliability:** Reduced risk of inconsistencies
4. **Easier Onboarding:** New developers only need to learn one codebase

### Business Benefits:
1. **Cost Reduction:** Reduced development and maintenance costs
2. **Time Savings:** Faster development cycles
3. **Quality Improvement:** Better code quality and fewer bugs
4. **Scalability:** Easier to scale both implementations

## Risks and Mitigation

### High-Risk Changes:
1. **Integration Complexity:** High risk of breaking existing functionality
   - **Mitigation:** Comprehensive testing and gradual rollout
2. **Performance Impact:** Potential performance degradation
   - **Mitigation:** Performance testing and optimization
3. **Testing Coverage:** Incomplete test coverage
   - **Mitigation:** Comprehensive test strategy

### Testing Approach:
1. **Unit Tests:** Test individual components
2. **Integration Tests:** Test pipeline end-to-end
3. **Regression Tests:** Ensure no regressions
4. **Performance Tests:** Validate performance
5. **Compatibility Tests:** Ensure both pipelines work correctly

## Conclusion

The unification of the Colab and Backend pipelines will:

1. **Eliminate Code Duplication:** Reduce ~2,000+ lines of duplicate code
2. **Improve Maintainability:** Single source of truth for pipeline logic
3. **Enhance Consistency:** Ensure both pipelines behave identically
4. **Reduce Costs:** Lower development and maintenance costs
5. **Improve Quality:** Better code quality and fewer bugs

This unification is a significant undertaking but will provide long-term benefits for the project. The implementation should be done carefully with comprehensive testing to ensure no regressions.

## Next Steps

1. **Start Analysis:** Begin analyzing colab/pipeline.py to identify reusable components
2. **Create Shared Modules:** Create the initial shared modules
3. **Update Backend:** Update backend pipeline to use shared modules
4. **Update Colab:** Update colab pipeline to use shared modules
5. **Test:** Run comprehensive tests to validate changes

This plan provides a comprehensive roadmap for unifying the two pipelines while minimizing risks and ensuring successful implementation.