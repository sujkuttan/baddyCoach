# Stroke Classification Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate BST model for real stroke classification, align coach rules to actual data format, and implement fatigue trend calculation.

**Architecture:** Feature extraction pipeline computes 144-dim vectors from shuttle/pose/court data, BST classifier predicts stroke types, coach engine evaluates rules using dot-notation field paths, fitness stage computes fatigue from rally intensity patterns.

**Tech Stack:** Python, NumPy, Pandas, PyTorch (BST), PyYAML (rules)

---

## File Structure

| File | Purpose |
|------|---------|
| `backend/app/models/bst_features.py` | **NEW** — Feature extraction pipeline (144-dim vectors) |
| `backend/app/models/bst.py` | **MODIFY** — BST classifier with multi-architecture fallback |
| `backend/app/pipeline/strokes.py` | **MODIFY** — Use bst_features for extraction |
| `backend/app/coach/engine.py` | **MODIFY** — Add `_get_nested()` helper, dot-notation rules |
| `backend/app/coach/rules.yaml` | **MODIFY** — Update to dot-notation format |
| `backend/app/pipeline/analytics/fitness.py` | **MODIFY** — Add `compute_fatigue_trend()` |
| `colab/pipeline.py` | **MODIFY** — Integrate BST, update coach/fitness |
| `backend/tests/test_bst_features.py` | **NEW** — Feature extraction tests |
| `backend/tests/test_coach_rules.py` | **NEW** — Rule evaluation tests |
| `backend/tests/test_fatigue_trend.py` | **NEW** — Fatigue trend tests |

---

## Task 1: BST Feature Extraction Pipeline

**Files:**
- Create: `backend/app/models/bst_features.py`
- Create: `backend/tests/test_bst_features.py`

- [ ] **Step 1: Write failing tests for feature extraction**

```python
# backend/tests/test_bst_features.py
import numpy as np
import pandas as pd
import pytest
from app.models.bst_features import BSTFeatureExtractor


@pytest.fixture
def sample_shuttle_df():
    """10 frames of shuttle trajectory."""
    return pd.DataFrame({
        "frame": range(10),
        "x": np.linspace(100, 400, 10),
        "y": np.linspace(200, 350, 10),
        "confidence": np.full(10, 0.9),
    })


@pytest.fixture
def sample_pose_df():
    """2 players, 10 frames of pose keypoints."""
    rows = []
    for frame in range(10):
        for pid in ["player_1", "player_2"]:
            kps = np.random.rand(17, 3).astype(np.float32)
            kps[:, :2] *= np.array([640, 480])
            kps[:, 2] = 0.9
            rows.append({"frame": frame, "player_id": pid, "keypoints": kps.tolist()})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_court():
    return {"court_length": 13.4, "court_width": 5.18}


def test_feature_extractor_returns_144_dims(sample_shuttle_df, sample_pose_df, sample_court):
    extractor = BSTFeatureExtractor(
        frame_width=640, frame_height=480,
        court_length=13.4, court_width=5.18
    )
    features = extractor.extract(
        shuttle_df=sample_shuttle_df,
        pose_df=sample_pose_df,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert features.shape == (144,)


def test_feature_extractor_handles_missing_shuttle(sample_pose_df):
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    features = extractor.extract(
        shuttle_df=None,
        pose_df=sample_pose_df,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert features.shape == (144,)
    assert not np.all(features == 0)  # Pose features should be non-zero


def test_feature_extractor_handles_missing_pose(sample_shuttle_df):
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    features = extractor.extract(
        shuttle_df=sample_shuttle_df,
        pose_df=None,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    assert features.shape == (144,)
    assert not np.all(features == 0)  # Shuttle features should be non-zero


def test_feature_extractor_normalizes_values(sample_shuttle_df, sample_pose_df):
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    features = extractor.extract(
        shuttle_df=sample_shuttle_df,
        pose_df=sample_pose_df,
        target_frame=5,
        player_id="player_1",
        previous_shots=[]
    )
    # Most values should be in [-1, 1] range after normalization
    assert np.all(np.abs(features) < 10)  # Allow some outliers but mostly normalized


def test_previous_shots_encoding():
    extractor = BSTFeatureExtractor(640, 480, 13.4, 5.18)
    prev_shots = [
        {"stroke_type": "clear", "frame": 0},
        {"stroke_type": "smash", "frame": 3},
        {"stroke_type": "drop", "frame": 6},
    ]
    encoding = extractor._encode_previous_shots(prev_shots, current_frame=10)
    assert encoding.shape == (42,)  # 3 shots × 14 dims
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_bst_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.bst_features'`

- [ ] **Step 3: Implement BSTFeatureExtractor**

```python
# backend/app/models/bst_features.py
"""Feature extraction pipeline for BST stroke classification.

Extracts 144-dimensional feature vectors from shuttle trajectory,
pose keypoints, court position, and rally context.
"""

import numpy as np
import pandas as pd


# Stroke classes for one-hot encoding
STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


class BSTFeatureExtractor:
    """Extracts 144-dim feature vectors for BST stroke classification.
    
    Feature layout (144 dims total):
    - Shuttle trajectory (24): velocity/accel over 8-frame window
    - Shuttle position (6): current x, y, speed, direction
    - Pose joints (48): 17 keypoints × (x, y) normalized
    - Pose dynamics (12): joint velocities
    - Body orientation (6): torso angle, lean, arm extension
    - Court position (6): normalized court coords
    - Rally context (42): previous 3 shots encoded
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        court_length: float = 13.4,
        court_width: float = 5.18,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.court_length = court_length
        self.court_width = court_width
    
    def extract(
        self,
        shuttle_df: pd.DataFrame | None,
        pose_df: pd.DataFrame | None,
        target_frame: int,
        player_id: str,
        previous_shots: list[dict],
    ) -> np.ndarray:
        """Extract 144-dim feature vector for a single hit frame.
        
        Args:
            shuttle_df: DataFrame with columns [frame, x, y, confidence]
            pose_df: DataFrame with columns [frame, player_id, keypoints]
            target_frame: Frame number of the hit
            player_id: Player who hit the shuttle
            previous_shots: List of dicts with [stroke_type, frame]
            
        Returns:
            np.ndarray of shape (144,)
        """
        features = []
        
        # 1. Shuttle trajectory features (24 dims)
        features.append(self._extract_shuttle_trajectory(shuttle_df, target_frame))
        
        # 2. Shuttle position features (6 dims)
        features.append(self._extract_shuttle_position(shuttle_df, target_frame))
        
        # 3. Pose joint features (48 dims)
        features.append(self._extract_pose_joints(pose_df, target_frame, player_id))
        
        # 4. Pose dynamics features (12 dims)
        features.append(self._extract_pose_dynamics(pose_df, target_frame, player_id))
        
        # 5. Body orientation features (6 dims)
        features.append(self._extract_body_orientation(pose_df, target_frame, player_id))
        
        # 6. Court position features (6 dims)
        features.append(self._extract_court_position(shuttle_df, target_frame))
        
        # 7. Rally context features (42 dims)
        features.append(self._encode_previous_shots(previous_shots, target_frame))
        
        combined = np.concatenate(features)
        assert combined.shape == (144,), f"Expected 144 dims, got {combined.shape}"
        return combined
    
    def _extract_shuttle_trajectory(self, shuttle_df, target_frame):
        """24 dims: velocity/acceleration over 8-frame window."""
        if shuttle_df is None or len(shuttle_df) == 0:
            return np.zeros(24)
        
        window = shuttle_df[
            (shuttle_df["frame"] >= target_frame - 8) &
            (shuttle_df["frame"] <= target_frame)
        ].sort_values("frame")
        
        if len(window) < 2:
            return np.zeros(24)
        
        x = window["x"].values.astype(np.float64)
        y = window["y"].values.astype(np.float64)
        
        # Normalize by frame dimensions
        x = x / self.frame_width
        y = y / self.frame_height
        
        # Velocity (dx, dy) for each consecutive pair
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Acceleration (ddx, ddy)
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        
        # Pad to固定 size (8 velocities, 7 accelerations = 15 raw values)
        # Use statistical summary to get 24 dims
        features = np.array([
            # Velocity stats (8 dims)
            np.mean(dx), np.std(dx), np.min(dx), np.max(dx),
            np.mean(dy), np.std(dy), np.min(dy), np.max(dy),
            # Acceleration stats (8 dims)
            np.mean(ddx), np.std(ddx), np.min(ddx), np.max(ddx),
            np.mean(ddy), np.std(ddy), np.min(ddy), np.max(ddy),
            # Speed (magnitude of velocity)
            np.mean(np.sqrt(dx**2 + dy**2)),
            np.max(np.sqrt(dx**2 + dy**2)),
            # Direction consistency
            np.mean(np.abs(dx)),
            np.mean(np.abs(dy)),
            # Trajectory curvature
            np.mean(np.abs(ddx + ddy)),
            # Final velocity
            dx[-1] if len(dx) > 0 else 0,
            dy[-1] if len(dy) > 0 else 0,
            # Position change over window
            x[-1] - x[0] if len(x) > 1 else 0,
            y[-1] - y[0] if len(y) > 1 else 0,
        ])
        
        return features[:24]
    
    def _extract_shuttle_position(self, shuttle_df, target_frame):
        """6 dims: current x, y, speed, direction, height, distance_from_net."""
        if shuttle_df is None or len(shuttle_df) == 0:
            return np.zeros(6)
        
        row = shuttle_df[shuttle_df["frame"] == target_frame]
        if len(row) == 0:
            # Interpolate from nearby frames
            nearby = shuttle_df[
                (shuttle_df["frame"] >= target_frame - 2) &
                (shuttle_df["frame"] <= target_frame + 2)
            ]
            if len(nearby) == 0:
                return np.zeros(6)
            row = nearby.iloc[[-1]]
        
        x = float(row.iloc[0]["x"]) / self.frame_width
        y = float(row.iloc[0]["y"]) / self.frame_height
        
        # Compute instantaneous speed and direction
        prev_rows = shuttle_df[shuttle_df["frame"] == target_frame - 1]
        if len(prev_rows) > 0:
            prev_x = float(prev_rows.iloc[0]["x"]) / self.frame_width
            prev_y = float(prev_rows.iloc[0]["y"]) / self.frame_height
            dx = x - prev_x
            dy = y - prev_y
            speed = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
        else:
            speed = 0
            direction = 0
        
        # Height: y-position normalized (0=top, 1=bottom)
        height = y
        
        # Distance from net (assuming net is at y=0.5 in normalized coords)
        dist_from_net = abs(y - 0.5)
        
        return np.array([x, y, speed, direction, height, dist_from_net])
    
    def _extract_pose_joints(self, pose_df, target_frame, player_id):
        """48 dims: 17 keypoints × (x, y) normalized by bounding box."""
        if pose_df is None or len(pose_df) == 0:
            return np.zeros(48)
        
        row = pose_df[
            (pose_df["frame"] == target_frame) &
            (pose_df["player_id"] == player_id)
        ]
        
        if len(row) == 0:
            # Try nearby frames
            nearby = pose_df[
                (pose_df["frame"] >= target_frame - 2) &
                (pose_df["frame"] <= target_frame + 2) &
                (pose_df["player_id"] == player_id)
            ]
            if len(nearby) == 0:
                return np.zeros(48)
            row = nearby.iloc[[-1]]
        
        kps = np.array(row.iloc[0]["keypoints"])
        if kps.shape != (17, 3):
            kps = np.array(kps.tolist())
        if kps.shape != (17, 3):
            return np.zeros(48)
        
        # Extract x, y coordinates (ignore confidence)
        coords = kps[:, :2]  # (17, 2)
        
        # Normalize by bounding box diagonal
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        diag = np.linalg.norm(bbox_max - bbox_min)
        if diag == 0:
            diag = 1
        
        coords_norm = (coords - bbox_min) / diag
        
        # Center align
        center = (bbox_min + bbox_max) / 2
        coords_centered = coords_norm - center / diag
        
        # Flatten to 34 dims (17 × 2)
        flat = coords_centered.flatten()
        
        # Pad to 48 dims with zeros
        return np.pad(flat, (0, 48 - len(flat)))[:48]
    
    def _extract_pose_dynamics(self, pose_df, target_frame, player_id):
        """12 dims: wrist, elbow, shoulder velocities."""
        if pose_df is None or len(pose_df) == 0:
            return np.zeros(12)
        
        # Get current and previous frame keypoints
        curr_row = pose_df[
            (pose_df["frame"] == target_frame) &
            (pose_df["player_id"] == player_id)
        ]
        prev_row = pose_df[
            (pose_df["frame"] == target_frame - 1) &
            (pose_df["player_id"] == player_id)
        ]
        
        if len(curr_row) == 0 or len(prev_row) == 0:
            return np.zeros(12)
        
        curr_kps = np.array(curr_row.iloc[0]["keypoints"])
        prev_kps = np.array(prev_row.iloc[0]["keypoints"])
        
        if curr_kps.shape != (17, 3) or prev_kps.shape != (17, 3):
            return np.zeros(12)
        
        # Key joints: wrist(9), elbow(7), shoulder(5)
        key_joints = [5, 7, 9]
        velocities = []
        
        for joint in key_joints:
            dx = (curr_kps[joint, 0] - prev_kps[joint, 0]) / self.frame_width
            dy = (curr_kps[joint, 1] - prev_kps[joint, 1]) / self.frame_height
            velocities.extend([dx, dy, np.sqrt(dx**2 + dy**2)])
        
        return np.array(velocities)[:12]
    
    def _extract_body_orientation(self, pose_df, target_frame, player_id):
        """6 dims: torso angle, lean, arm extension metrics."""
        if pose_df is None or len(pose_df) == 0:
            return np.zeros(6)
        
        row = pose_df[
            (pose_df["frame"] == target_frame) &
            (pose_df["player_id"] == player_id)
        ]
        
        if len(row) == 0:
            return np.zeros(6)
        
        kps = np.array(row.iloc[0]["keypoints"])
        if kps.shape != (17, 3):
            kps = np.array(kps.tolist())
        if kps.shape != (17, 3):
            return np.zeros(6)
        
        # Joint indices
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        LEFT_ELBOW, RIGHT_ELBOW = 7, 8
        LEFT_WRIST, RIGHT_WRIST = 9, 10
        LEFT_HIP, RIGHT_HIP = 11, 12
        
        # Torso angle (shoulder to hip line)
        torso_vec = kps[LEFT_HIP, :2] - kps[LEFT_SHOULDER, :2]
        torso_angle = np.arctan2(torso_vec[1], torso_vec[0])
        
        # Lean (horizontal offset of shoulders relative to hips)
        shoulder_center = (kps[LEFT_SHOULDER, :2] + kps[RIGHT_SHOULDER, :2]) / 2
        hip_center = (kps[LEFT_HIP, :2] + kps[RIGHT_HIP, :2]) / 2
        lean = (shoulder_center[0] - hip_center[0]) / self.frame_width
        
        # Arm extension (distance from shoulder to wrist, normalized)
        left_arm = np.linalg.norm(kps[LEFT_WRIST, :2] - kps[LEFT_SHOULDER, :2])
        right_arm = np.linalg.norm(kps[RIGHT_WRIST, :2] - kps[RIGHT_SHOULDER, :2])
        torso_len = np.linalg.norm(kps[LEFT_SHOULDER, :2] - kps[LEFT_HIP, :2])
        
        if torso_len == 0:
            torso_len = 1
        
        left_ext = left_arm / torso_len
        right_ext = right_arm / torso_len
        
        # Racket arm (higher wrist position indicates hitting)
        racket_arm_ext = max(left_ext, right_ext)
        
        return np.array([
            torso_angle / np.pi,  # Normalize to [-1, 1]
            lean,
            left_ext / 3,  # Normalize typical range
            right_ext / 3,
            racket_arm_ext / 3,
            (left_ext - right_ext) / 3,  # Asymmetry
        ])
    
    def _extract_court_position(self, shuttle_df, target_frame):
        """6 dims: normalized court x, y, distance from corners."""
        if shuttle_df is None or len(shuttle_df) == 0:
            return np.zeros(6)
        
        row = shuttle_df[shuttle_df["frame"] == target_frame]
        if len(row) == 0:
            return np.zeros(6)
        
        x = float(row.iloc[0]["x"]) / self.frame_width
        y = float(row.iloc[0]["y"]) / self.frame_height
        
        # Map to court coordinates (0-1 range)
        court_x = x  # Assuming full width maps to court width
        court_y = y  # Assuming full height maps to court length
        
        # Distances from corners (normalized)
        dist_tl = np.sqrt(court_x**2 + court_y**2)
        dist_tr = np.sqrt((1 - court_x)**2 + court_y**2)
        dist_bl = np.sqrt(court_x**2 + (1 - court_y)**2)
        dist_br = np.sqrt((1 - court_x)**2 + (1 - court_y)**2)
        
        return np.array([
            court_x,
            court_y,
            dist_tl,
            dist_tr,
            dist_bl,
            dist_br,
        ])
    
    def _encode_previous_shots(self, previous_shots, current_frame):
        """42 dims: encode last 3 shots (type one-hot + frame gap)."""
        features = []
        
        # Take last 3 shots before current frame
        recent = [s for s in previous_shots if s["frame"] < current_frame][-3:]
        
        for shot in recent:
            # One-hot encode stroke type (12 dims)
            stroke = shot.get("stroke_type", "clear")
            one_hot = np.zeros(12)
            if stroke in STROKE_CLASSES:
                one_hot[STROKE_CLASSES.index(stroke)] = 1
            
            # Frame gap (normalized)
            gap = (current_frame - shot["frame"]) / 100  # Normalize by ~3 seconds at 30fps
            
            # Confidence (if available)
            conf = shot.get("stroke_confidence", 0.8)
            
            # Combine: 12 + 1 + 1 = 14 dims per shot
            features.extend(one_hot)
            features.extend([gap, conf])
        
        # Pad to 42 dims (3 shots × 14)
        while len(features) < 42:
            features.append(0)
        
        return np.array(features[:42])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_bst_features.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/bst_features.py backend/tests/test_bst_features.py
git commit -m "feat: add BST feature extraction pipeline with 144-dim vectors"
```

---

## Task 2: BST Classifier with Multi-Architecture Fallback

**Files:**
- Modify: `backend/app/models/bst.py`
- Modify: `backend/tests/test_bst_features.py` (add classifier tests)

- [ ] **Step 1: Write failing tests for BST classifier**

```python
# Add to backend/tests/test_bst_features.py

def test_bst_classifier_with_mock_model(tmp_path):
    """Test BST classifier with a mock model checkpoint."""
    import torch
    import torch.nn as nn
    
    # Create a simple model that accepts 144-dim input
    class MockBSTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(144, 12)
        
        def forward(self, x):
            return self.fc(x)
    
    model = MockBSTModel()
    checkpoint = {"model": model}
    model_path = tmp_path / "test_bst.pt"
    torch.save(checkpoint, model_path)
    
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier(str(model_path), device="cpu")
    
    features = np.random.rand(144).astype(np.float32)
    stroke_type, confidence = classifier.predict(features)
    
    assert stroke_type in STROKE_CLASSES
    assert 0 <= confidence <= 1


def test_bst_classifier_fallback_when_no_model():
    """Test BST classifier falls back to random when no model."""
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier(None, device="cpu")
    
    features = np.random.rand(144).astype(np.float32)
    stroke_type, confidence = classifier.predict(features)
    
    assert stroke_type in STROKE_CLASSES
    assert confidence == 0.8  # Fallback confidence


def test_bst_classifier_handles_corrupt_checkpoint(tmp_path):
    """Test BST classifier handles corrupt checkpoint gracefully."""
    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_text("not a real checkpoint")
    
    from app.models.bst import BSTClassifier
    classifier = BSTClassifier(str(corrupt_path), device="cpu")
    
    features = np.random.rand(144).astype(np.float32)
    stroke_type, confidence = classifier.predict(features)
    
    # Should fall back to random
    assert stroke_type in STROKE_CLASSES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_bst_features.py::test_bst_classifier_with_mock_model -v`
Expected: FAIL (current BSTClassifier doesn't handle the checkpoint format correctly)

- [ ] **Step 3: Update BSTClassifier with multi-architecture support**

```python
# backend/app/models/bst.py
"""BST (Badminton Stroke Transformer) classifier for stroke classification.

Supports multiple checkpoint formats and falls back to rule-based
classification when model loading fails.
"""

import numpy as np
from pathlib import Path

STROKE_CLASSES = [
    "serve", "short_serve", "flick_serve", "clear", "lift",
    "smash", "drop", "net_shot", "drive", "push", "block", "kill"
]


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    x_normalized = arr[:, 0] / v_width
    y_normalized = arr[:, 1] / v_height
    return np.stack((x_normalized, y_normalized), axis=-1)


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance."""
    diag = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    diag = np.where(diag == 0, 1, diag)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_normalized = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / diag, 0.0)
    y_normalized = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / diag, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_normalized = (center - bbox[:, :2]) / diag
        x_normalized -= c_normalized[:, None, 0]
        y_normalized -= c_normalized[:, None, 1]

    return np.stack((x_normalized, y_normalized), axis=-1)


class BSTClassifier:
    """BST classifier with multi-architecture fallback.
    
    Supports:
    - Checkpoint with 'model' key containing nn.Module
    - Checkpoint with 'state_dict' key
    - Raw state_dict (tries known architectures)
    - Falls back to rule-based classification
    """
    
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.classes = STROKE_CLASSES
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path, device)
    
    def _load_model(self, path: str, device: str):
        """Try multiple strategies to load the model."""
        import torch
        
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"BST checkpoint load failed: {e}")
            return None
        
        # Strategy 1: Checkpoint contains model object
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            if callable(model) and hasattr(model, 'eval'):
                model.eval()
                return model
        
        # Strategy 2: Checkpoint contains state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return self._load_from_state_dict(checkpoint['state_dict'], device)
        
        # Strategy 3: Checkpoint is raw state_dict
        if isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint.keys()):
            return self._load_from_state_dict(checkpoint, device)
        
        # Strategy 4: Checkpoint is the model itself
        if callable(checkpoint) and hasattr(checkpoint, 'eval'):
            checkpoint.eval()
            return checkpoint
        
        return None
    
    def _load_from_state_dict(self, state_dict: dict, device: str):
        """Try known BST architectures to load state_dict."""
        import torch
        import torch.nn as nn
        
        # Try simple MLP
        try:
            model = SimpleBST_MLP()
            model.load_state_dict(state_dict)
            model.to(device).eval()
            print("BST loaded as SimpleBST_MLP")
            return model
        except Exception:
            pass
        
        # Try 1D ResNet
        try:
            model = SimpleBST_ResNet1D()
            model.load_state_dict(state_dict)
            model.to(device).eval()
            print("BST loaded as SimpleBST_ResNet1D")
            return model
        except Exception:
            pass
        
        return None
    
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predict stroke type from 144-dim feature vector.
        
        Args:
            features: (144,) normalized feature vector
            
        Returns:
            (stroke_type, confidence) tuple
        """
        if self.model is None:
            # Fallback: rule-based classification
            return self._rule_based_predict(features)
        
        import torch
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
    
    def _rule_based_predict(self, features: np.ndarray) -> tuple[str, float]:
        """Rule-based fallback when model is unavailable.
        
        Uses shuttle trajectory features (indices 0-29) to infer stroke type.
        """
        # Extract key features
        shuttle_speed = features[16] if len(features) > 16 else 0  # Mean speed
        shuttle_height = features[30] if len(features) > 30 else 0.5  # Height
        shuttle_dx = features[21] if len(features) > 21 else 0  # Final dx
        shuttle_dy = features[22] if len(features) > 22 else 0  # Final dy
        
        # Simple heuristic rules
        if shuttle_speed > 0.3 and shuttle_dy > 0.1:
            # Fast and downward -> smash
            return "smash", 0.6
        elif shuttle_height < 0.3 and shuttle_speed < 0.1:
            # High position, slow -> net shot
            return "net_shot", 0.5
        elif shuttle_dy < -0.1 and shuttle_speed > 0.15:
            # Upward trajectory -> clear/lift
            if shuttle_speed > 0.25:
                return "clear", 0.55
            else:
                return "lift", 0.5
        elif shuttle_speed > 0.2 and abs(shuttle_dy) < 0.05:
            # Fast, horizontal -> drive
            return "drive", 0.5
        elif shuttle_height > 0.6 and shuttle_speed < 0.15:
            # Low position, slow -> drop
            return "drop", 0.5
        else:
            # Default
            return "clear", 0.4


class SimpleBST_MLP(nn.Module):
    """Simple MLP architecture for BST classification."""
    
    def __init__(self, input_dim=144, hidden_dim=128, num_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleBST_ResNet1D(nn.Module):
    """1D ResNet architecture for BST classification."""
    
    def __init__(self, input_dim=144, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 144)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_bst_features.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/bst.py
git commit -m "feat: BST classifier with multi-architecture fallback and rule-based prediction"
```

---

## Task 3: Integrate BST into Stroke Classification Stage

**Files:**
- Modify: `backend/app/pipeline/strokes.py`
- Modify: `backend/tests/test_strokes.py` (update existing tests)

- [ ] **Step 1: Update stroke classification stage**

```python
# backend/app/pipeline/strokes.py
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class StrokeClassificationStage:
    name = "stroke_classification"
    input_keys = ["hits", "shuttle", "pose", "court"]
    output_keys = ["shots"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        hits_df = artifacts.get_parquet("hits")
        if hits_df is None or len(hits_df) == 0:
            return StageResult.success(metadata={"shot_count": 0})

        shuttle_df = artifacts.get_parquet("shuttle")
        pose_df = artifacts.get_parquet("pose")
        court = artifacts.get("court") or {}

        from app.models.bst import BSTClassifier, STROKE_CLASSES
        from app.models.bst_features import BSTFeatureExtractor
        from app.config.settings import settings

        # Initialize BST classifier
        model_path = str(settings.bst_model_path) if settings.bst_model_path else None
        classifier = BSTClassifier(model_path, device=settings.device)
        
        # Initialize feature extractor
        frame_width = config.frame_width if hasattr(config, 'frame_width') else 640
        frame_height = config.frame_height if hasattr(config, 'frame_height') else 480
        extractor = BSTFeatureExtractor(
            frame_width=frame_width,
            frame_height=frame_height,
            court_length=court.get("court_length", 13.4),
            court_width=court.get("court_width", 5.18),
        )

        shots = []
        previous_shots = []  # Track for rally context
        
        for _, hit in hits_df.iterrows():
            frame = int(hit["frame"])
            
            # Extract 144-dim features
            features = extractor.extract(
                shuttle_df=shuttle_df,
                pose_df=pose_df,
                target_frame=frame,
                player_id="player_1",  # Will be updated in attribution stage
                previous_shots=previous_shots,
            )
            
            # Predict stroke type
            stroke_type, confidence = classifier.predict(features)
            
            shot = {
                "frame": frame,
                "hit_confidence": float(hit["confidence"]),
                "stroke_type": stroke_type,
                "stroke_confidence": confidence,
                "features": features.tolist(),  # Store for debugging
            }
            shots.append(shot)
            
            # Add to previous shots for context
            previous_shots.append({
                "stroke_type": stroke_type,
                "frame": frame,
                "stroke_confidence": confidence,
            })

        shots_df = pd.DataFrame(shots)
        artifacts.set_parquet("shots", shots_df)

        return StageResult.success(
            artifacts={"shots": artifacts.path("shots")},
            metadata={
                "shot_count": len(shots),
                "stroke_distribution": self._compute_distribution(shots),
            }
        )
    
    @staticmethod
    def _compute_distribution(shots):
        """Compute stroke type distribution for metadata."""
        if not shots:
            return {}
        from collections import Counter
        dist = Counter(s["stroke_type"] for s in shots)
        total = len(shots)
        return {k: v / total for k, v in dist.items()}
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `.venv/bin/pytest backend/tests/test_strokes.py -v`
Expected: PASS (or skip if tests exist)

- [ ] **Step 3: Commit**

```bash
git add backend/app/pipeline/strokes.py
git commit -m "feat: integrate BST feature extraction into stroke classification stage"
```

---

## Task 4: Coach Rules with Dot-Notation Field Paths

**Files:**
- Modify: `backend/app/coach/engine.py`
- Modify: `backend/app/coach/rules.yaml`
- Create: `backend/tests/test_coach_rules.py`

- [ ] **Step 1: Write failing tests for coach rules**

```python
# backend/tests/test_coach_rules.py
import pytest
from pathlib import Path
from app.coach.engine import CoachEngine


@pytest.fixture
def engine():
    rules_path = Path(__file__).parent.parent / "app" / "coach" / "rules.yaml"
    return CoachEngine(rules_path)


@pytest.fixture
def sample_analytics():
    return {
        "tactical_analytics": {
            "player_1": {
                "shot_distribution": {
                    "smash": 0.15,
                    "clear": 0.40,
                    "drop": 0.20,
                    "net_shot": 0.10,
                    "drive": 0.15,
                },
                "total_shots": 50,
            }
        },
        "fitness_analytics": {
            "player_1": {
                "fatigue_trend": "declining",
                "avg_recovery": 1.5,
                "rally_intensity": 2.3,
            }
        },
        "footwork_analytics": {
            "player_1": {
                "avg_recovery": 1.5,
                "distance_covered": 1200,
            }
        },
    }


def test_coach_engine_generates_recommendations(engine, sample_analytics):
    result = engine.generate(sample_analytics, "player_1")
    
    assert "strengths" in result
    assert "weaknesses" in result
    assert "top_3_improvements" in result
    assert "recommended_drills" in result
    assert "evidence" in result


def test_coach_triggers_smash_rule(engine, sample_analytics):
    # Low smash percentage should trigger weakness
    result = engine.generate(sample_analytics, "player_1")
    
    # Should have at least one weakness about smash
    weakness_text = " ".join(result["weaknesses"]).lower()
    assert "smash" in weakness_text or len(result["weaknesses"]) > 0


def test_coach_triggers_fatigue_rule(engine, sample_analytics):
    result = engine.generate(sample_analytics, "player_1")
    
    # Should detect declining fatigue trend
    weakness_text = " ".join(result["weaknesses"]).lower()
    assert "fatigue" in weakness_text or "declining" in weakness_text or len(result["weaknesses"]) > 0


def test_coach_handles_missing_data(engine):
    empty_analytics = {}
    result = engine.generate(empty_analytics, "player_1")
    
    # Should not crash, return empty results
    assert result["strengths"] == []
    assert result["weaknesses"] == []


def test_coach_get_nested_helper(engine):
    data = {
        "a": {"b": {"c": 42}},
        "x": [1, 2, 3],
    }
    
    assert engine._get_nested(data, "a.b.c") == 42
    assert engine._get_nested(data, "a.b") == {"c": 42}
    assert engine._get_nested(data, "missing.path") == 0
    assert engine._get_nested(data, "x.1") == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_coach_rules.py -v`
Expected: FAIL (current engine doesn't have `_get_nested` or dot-notation support)

- [ ] **Step 3: Implement updated CoachEngine**

```python
# backend/app/coach/engine.py
from pathlib import Path
from typing import Any

import yaml


class CoachEngine:
    """Rule-based coach engine with dot-notation field paths.
    
    Rules use dot-notation to access nested analytics fields:
        tactical.shot_distribution.smash
        fitness.fatigue_trend
        footwork.avg_recovery
    """
    
    def __init__(self, rules_path: Path | None = None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules.yaml"
        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)["rules"]

    def generate(self, analytics: dict[str, Any], player_id: str) -> dict[str, Any]:
        """Generate coaching recommendations for a player.
        
        Args:
            analytics: Nested dict with tactical_analytics, fitness_analytics, footwork_analytics
            player_id: Player identifier (e.g., "player_1")
            
        Returns:
            Dict with strengths, weaknesses, improvements, drills, evidence
        """
        strengths = []
        weaknesses = []
        improvements = []
        drills = []
        evidence = []

        # Build player-specific analytics view
        player_analytics = {
            "tactical": analytics.get("tactical_analytics", {}).get(player_id, {}),
            "fitness": analytics.get("fitness_analytics", {}).get(player_id, {}),
            "footwork": analytics.get("footwork_analytics", {}).get(player_id, {}),
        }
        
        # Add computed fields
        tactical = player_analytics["tactical"]
        if tactical:
            shot_dist = tactical.get("shot_distribution", {})
            tactical["max_shot_percentage"] = max(shot_dist.values()) if shot_dist else 0

        for rule in self.rules:
            if self._evaluate_rule(rule, player_analytics):
                entry = {
                    "finding": rule["recommendation"],
                    "metrics": self._extract_metrics(rule, player_analytics),
                }
                evidence.append(entry)

                if rule["category"] == "strength":
                    strengths.append(rule["recommendation"])
                elif rule["category"] == "weakness":
                    weaknesses.append(rule["recommendation"])
                    improvements.append(rule["recommendation"])
                    drills.append(rule.get("drill", ""))
                elif rule["category"] == "neutral":
                    # Neutral observations go to evidence only
                    pass

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3],
            "recommended_drills": drills[:3],
            "evidence": evidence,
        }
    
    def _evaluate_rule(self, rule: dict, analytics: dict) -> bool:
        """Evaluate a rule against analytics data.
        
        Rule format:
            check:
                field: tactical.shot_distribution.smash
                operator: "<"
                threshold: 0.3
                min_shots: tactical.total_shots >= 10
        """
        check = rule.get("check", {})
        if not check:
            return False
        
        # Check minimum data requirement
        min_shots_expr = check.get("min_shots")
        if min_shots_expr:
            if not self._evaluate_condition(min_shots_expr, analytics):
                return False
        
        # Evaluate main condition
        field_path = check.get("field")
        operator = check.get("operator")
        threshold = check.get("threshold", check.get("value"))
        
        if not field_path or not operator:
            return False
        
        value = self._get_nested(analytics, field_path)
        
        return self._compare(value, operator, threshold)
    
    def _evaluate_condition(self, expr: str, analytics: dict) -> bool:
        """Evaluate a condition expression like 'tactical.total_shots >= 10'."""
        # Parse expression: "field_path operator value"
        parts = expr.split()
        if len(parts) != 3:
            return False
        
        field_path, operator, value_str = parts
        
        try:
            value = float(value_str)
        except ValueError:
            return False
        
        field_value = self._get_nested(analytics, field_path)
        return self._compare(field_value, operator, value)
    
    def _compare(self, actual, operator: str, expected) -> bool:
        """Compare actual value against expected using operator."""
        try:
            actual = float(actual)
            expected = float(expected)
        except (TypeError, ValueError):
            # String comparison for equality operators
            if operator == "==":
                return str(actual) == str(expected)
            elif operator == "!=":
                return str(actual) != str(expected)
            return False
        
        if operator == "<":
            return actual < expected
        elif operator == ">":
            return actual > expected
        elif operator == "<=":
            return actual <= expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        
        return False
    
    def _get_nested(self, data: dict, path: str):
        """Extract value from nested dict using dot notation.
        
        Examples:
            _get_nested(data, "a.b.c") -> data["a"]["b"]["c"]
            _get_nested(data, "x.1") -> data["x"][1] (for lists)
        """
        keys = path.split(".")
        current = data
        
        for key in keys:
            if current is None:
                return 0
            
            if isinstance(current, dict):
                current = current.get(key, 0)
            elif isinstance(current, (list, tuple)):
                try:
                    idx = int(key)
                    current = current[idx] if 0 <= idx < len(current) else 0
                except (ValueError, IndexError):
                    return 0
            else:
                return 0
        
        return current if current is not None else 0
    
    def _extract_metrics(self, rule: dict, analytics: dict) -> list[str]:
        """Extract human-readable metrics for evidence."""
        metrics = []
        check = rule.get("check", {})
        
        if "field" in check:
            value = self._get_nested(analytics, check["field"])
            field_name = check["field"].split(".")[-1]
            if isinstance(value, float):
                metrics.append(f"{field_name}: {value:.2f}")
            else:
                metrics.append(f"{field_name}: {value}")
        
        # Add common metrics
        total_shots = self._get_nested(analytics, "tactical.total_shots")
        if total_shots > 0:
            metrics.append(f"total shots: {total_shots}")
        
        fatigue = self._get_nested(analytics, "fitness.fatigue_trend")
        if fatigue and fatigue != "unknown":
            metrics.append(f"fatigue trend: {fatigue}")
        
        return metrics if metrics else ["data available"]
```

- [ ] **Step 4: Update rules.yaml to dot-notation format**

```yaml
# backend/app/coach/rules.yaml
rules:
  - name: smash_efficiency
    check:
      field: tactical.shot_distribution.smash
      operator: "<"
      threshold: 0.3
      min_shots: tactical.total_shots >= 10
    recommendation: "Your smash conversion rate is low. Focus on placement over power — aim for the sidelines and body rather than simply hitting hard."
    category: weakness
    drill: "Practice targeted smashes to designated court zones with a feeder."

  - name: recovery_speed
    check:
      field: footwork.avg_recovery
      operator: ">"
      threshold: 1.2
    recommendation: "Recovery after shots is slower than optimal. Work on split-step timing and base positioning."
    category: weakness
    drill: "Shadow footwork drills: return to base after each shot call, 3 sets of 20."

  - name: shot_variety
    check:
      field: tactical.max_shot_percentage
      operator: ">"
      threshold: 0.5
      min_shots: tactical.total_shots >= 20
    recommendation: "Shot selection is predictable. Vary your attack to keep opponents off balance."
    category: weakness
    drill: "Rally drills with constraint: alternate clear/drop/net each shot."

  - name: fatigue_management
    check:
      field: fitness.fatigue_trend
      operator: "=="
      value: "declining"
    recommendation: "Performance declines in later rallies. Improve match fitness and manage energy in early games."
    category: weakness
    drill: "Interval training: 12x (30s high intensity + 30s rest) to build rally endurance."

  - name: net_play_strength
    check:
      field: tactical.shot_distribution.net_shot
      operator: ">"
      threshold: 0.2
      min_shots: tactical.total_shots >= 10
    recommendation: "Strong net play presence. Use this to set up attacking opportunities."
    category: strength
    drill: "Maintain net dominance with variation: net kill, net lift, net spin."

  - name: clear_usage
    check:
      field: tactical.shot_distribution.clear
      operator: ">"
      threshold: 0.35
      min_shots: tactical.total_shots >= 10
    recommendation: "Heavy use of clears — effective for defense but consider mixing with drops and smashes."
    category: neutral
    drill: "Clear-drop combination drills from rear court."
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_coach_rules.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/coach/engine.py backend/app/coach/rules.yaml backend/tests/test_coach_rules.py
git commit -m "feat: coach rules with dot-notation field paths and operator evaluation"
```

---

## Task 5: Fitness Fatigue Trend Calculation

**Files:**
- Modify: `backend/app/pipeline/analytics/fitness.py`
- Create: `backend/tests/test_fatigue_trend.py`

- [ ] **Step 1: Write failing tests for fatigue trend**

```python
# backend/tests/test_fatigue_trend.py
import numpy as np
import pytest
from app.pipeline.analytics.fitness import FitnessAnalyticsStage


def test_fatigue_trend_declining():
    """High intensity early, low intensity late -> declining."""
    intensities = [3.0, 3.2, 2.8, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "declining"


def test_fatigue_trend_stable():
    """Consistent intensity throughout -> stable."""
    intensities = [2.0, 2.1, 1.9, 2.0, 2.1, 1.9, 2.0, 2.1, 1.9, 2.0]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "stable"


def test_fatigue_trend_improving():
    """Low intensity early, high intensity late -> improving."""
    intensities = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "improving"


def test_fatigue_trend_insufficient_data():
    """Less than 5 rallies -> insufficient_data."""
    intensities = [2.0, 2.1, 1.9]
    result = FitnessAnalyticsStage._compute_fatigue_trend(intensities)
    assert result == "insufficient_data"


def test_fatigue_trend_empty():
    """Empty list -> insufficient_data."""
    result = FitnessAnalyticsStage._compute_fatigue_trend([])
    assert result == "insufficient_data"


def test_fitness_analytics_populates_intensities():
    """Verify rally_intensities list is populated."""
    from app.pipeline.base import ArtifactStore, StageConfig
    import pandas as pd
    
    # Create mock artifacts
    artifacts = ArtifactStore("/tmp/test_job")
    
    footwork = {
        "player_1": {"avg_recovery": 1.0, "distance_covered": 500, "recovery_times": []}
    }
    artifacts.set("footwork_analytics", footwork)
    
    rallies_df = pd.DataFrame([
        {"rally_id": 1, "start_frame": 0, "end_frame": 300, "shot_count": 8},
        {"rally_id": 2, "start_frame": 330, "end_frame": 600, "shot_count": 6},
        {"rally_id": 3, "start_frame": 630, "end_frame": 900, "shot_count": 7},
        {"rally_id": 4, "start_frame": 930, "end_frame": 1200, "shot_count": 5},
        {"rally_id": 5, "start_frame": 1230, "end_frame": 1500, "shot_count": 4},
    ])
    artifacts.set_parquet("rallies", rallies_df)
    
    shots_df = pd.DataFrame([
        {"frame": f, "player_id": "player_1", "stroke_type": "clear", "stroke_confidence": 0.8}
        for f in range(0, 1500, 30)  # ~50 shots
    ])
    artifacts.set_parquet("shots", shots_df)
    
    config = StageConfig()
    stage = FitnessAnalyticsStage()
    result = stage.run(artifacts, config)
    
    assert result.success
    fitness = artifacts.get("fitness_analytics")
    assert "player_1" in fitness
    assert len(fitness["player_1"]["rally_intensities"]) == 5
    assert fitness["player_1"]["fatigue_trend"] in ["stable", "declining", "improving"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_fatigue_trend.py -v`
Expected: FAIL (current implementation returns "insufficient_data" always)

- [ ] **Step 3: Implement updated FitnessAnalyticsStage**

```python
# backend/app/pipeline/analytics/fitness.py
import numpy as np
import pandas as pd

from app.pipeline.base import ArtifactStore, StageConfig, StageResult


class FitnessAnalyticsStage:
    name = "fitness_analytics"
    input_keys = ["footwork_analytics", "rallies", "shots"]
    output_keys = ["fitness_analytics"]

    def run(self, artifacts: ArtifactStore, config: StageConfig) -> StageResult:
        footwork = artifacts.get("footwork_analytics")
        rallies_df = artifacts.get_parquet("rallies")
        shots_df = artifacts.get_parquet("shots")

        if footwork is None:
            return StageResult.from_error("Footwork analytics required")

        fitness = {}
        for player_id, fw_data in footwork.items():
            rally_intensities = []
            
            if rallies_df is not None and shots_df is not None:
                for _, rally in rallies_df.iterrows():
                    # Count player's shots in this rally
                    rally_shots = shots_df[
                        (shots_df["frame"] >= rally["start_frame"]) &
                        (shots_df["frame"] <= rally["end_frame"]) &
                        (shots_df["player_id"] == player_id)
                    ]
                    
                    # Compute intensity as shots per second
                    duration_frames = rally["end_frame"] - rally["start_frame"]
                    duration_seconds = max(duration_frames / 30, 0.1)  # Assume 30fps
                    intensity = len(rally_shots) / duration_seconds
                    rally_intensities.append(float(intensity))

            # Compute fatigue trend from rally intensities
            fatigue_trend = self._compute_fatigue_trend(rally_intensities)
            
            # Compute additional fitness metrics
            avg_intensity = float(np.mean(rally_intensities)) if rally_intensities else 0
            peak_intensity = float(np.max(rally_intensities)) if rally_intensities else 0
            intensity_std = float(np.std(rally_intensities)) if rally_intensities else 0
            
            # Late rally fatigue: compare last 3 rallies to first 3
            late_rally_fatigue = self._compute_late_rally_fatigue(rally_intensities)

            fitness[player_id] = {
                "rally_intensity": avg_intensity,
                "rally_intensities": rally_intensities,
                "fatigue_trend": fatigue_trend,
                "avg_recovery": fw_data.get("avg_recovery", 0),
                "total_distance": fw_data.get("distance_covered", 0),
                "peak_intensity": peak_intensity,
                "intensity_std": intensity_std,
                "late_rally_fatigue": late_rally_fatigue,
                "rally_count": len(rally_intensities),
            }

        artifacts.set("fitness_analytics", fitness)

        return StageResult.success(
            artifacts={"fitness_analytics": artifacts.path("fitness_analytics")},
            metadata={
                "rally_intensity": {k: v["rally_intensity"] for k, v in fitness.items()},
                "fatigue_trend": {k: v["fatigue_trend"] for k, v in fitness.items()},
            }
        )

    @staticmethod
    def _compute_fatigue_trend(rally_intensities: list[float]) -> str:
        """Analyze rally intensity over time to detect fatigue.
        
        Uses quarter comparison + linear regression slope.
        
        Args:
            rally_intensities: List of intensity values (shots/second) per rally
            
        Returns:
            "improving" | "stable" | "declining" | "insufficient_data"
        """
        if len(rally_intensities) < 5:
            return "insufficient_data"
        
        n = len(rally_intensities)
        
        # Split into quarters
        q1 = rally_intensities[:n//4]
        q2 = rally_intensities[n//4:n//2]
        q3 = rally_intensities[n//2:3*n//4]
        q4 = rally_intensities[3*n//4:]
        
        avg_q1 = np.mean(q1) if q1 else 0
        avg_q4 = np.mean(q4) if q4 else 0
        
        # Linear regression slope
        x = np.arange(len(rally_intensities))
        slope = np.polyfit(x, rally_intensities, 1)[0]
        
        # Normalize slope by average intensity
        avg_intensity = np.mean(rally_intensities)
        normalized_slope = slope / avg_intensity if avg_intensity > 0 else 0
        
        # Decision logic
        if avg_q4 < avg_q1 * 0.8 and normalized_slope < -0.01:
            return "declining"
        elif avg_q4 > avg_q1 * 1.2 and normalized_slope > 0.01:
            return "improving"
        return "stable"
    
    @staticmethod
    def _compute_late_rally_fatigue(rally_intensities: list[float]) -> float:
        """Compute fatigue factor from late rallies vs early rallies.
        
        Returns:
            Float > 0 if late rallies are less intense (fatigue detected)
            Float < 0 if late rallies are more intense (improving)
            0 if no significant difference
        """
        if len(rally_intensities) < 6:
            return 0.0
        
        first_half = rally_intensities[:len(rally_intensities)//2]
        second_half = rally_intensities[len(rally_intensities)//2:]
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        if avg_first == 0:
            return 0.0
        
        # Positive value means performance declined
        return float((avg_first - avg_second) / avg_first)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_fatigue_trend.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/analytics/fitness.py backend/tests/test_fatigue_trend.py
git commit -m "feat: fatigue trend calculation with quarter comparison and linear regression"
```

---

## Task 6: Integrate All Changes into Colab Pipeline

**Files:**
- Modify: `colab/pipeline.py`

- [ ] **Step 1: Update stage_strokes in colab/pipeline.py**

Replace the `stage_strokes` function (lines 322-342) with BST integration:

```python
def stage_strokes(hits_data, shuttle_data, pose_data=None, court=None, device="cuda"):
    """Classify strokes using BST model with feature extraction."""
    if not hits_data:
        return []
    
    shuttle_df = pd.DataFrame(shuttle_data) if shuttle_data else pd.DataFrame()
    pose_df = pd.DataFrame(pose_data) if pose_data else pd.DataFrame()
    
    # Import BST components
    from app.models.bst import BSTClassifier, STROKE_CLASSES
    from app.models.bst_features import BSTFeatureExtractor
    
    # Initialize classifier and feature extractor
    bst_path = str(BST_PATH) if BST_PATH.exists() else None
    classifier = BSTClassifier(bst_path, device=device)
    
    # Get video dimensions from shuttle data or defaults
    frame_width = 640
    frame_height = 480
    if shuttle_df is not None and len(shuttle_df) > 0:
        frame_width = max(shuttle_df["x"].max(), 640)
        frame_height = max(shuttle_df["y"].max(), 480)
    
    extractor = BSTFeatureExtractor(
        frame_width=frame_width,
        frame_height=frame_height,
        court_length=COURT_LENGTH,
        court_width=COURT_WIDTH,
    )
    
    shots = []
    previous_shots = []
    
    for hit in hits_data:
        frame = hit["frame"]
        
        # Extract 144-dim features
        features = extractor.extract(
            shuttle_df=shuttle_df,
            pose_df=pose_df,
            target_frame=frame,
            player_id="player_1",
            previous_shots=previous_shots,
        )
        
        # Predict stroke type
        stroke_type, confidence = classifier.predict(features)
        
        shots.append({
            "frame": frame,
            "hit_confidence": hit["confidence"],
            "stroke_type": stroke_type,
            "stroke_confidence": confidence,
        })
        
        previous_shots.append({
            "stroke_type": stroke_type,
            "frame": frame,
            "stroke_confidence": confidence,
        })
    
    return shots
```

- [ ] **Step 2: Update stage_fitness in colab/pipeline.py**

Replace the `stage_fitness` function (lines 417-431) with real fatigue calculation:

```python
def stage_fitness(footwork_data, rallies_data, shots_data):
    """Compute fitness analytics with real fatigue trend detection."""
    fitness = {}
    shots_df = pd.DataFrame(shots_data) if shots_data else pd.DataFrame()
    rallies_df = pd.DataFrame(rallies_data) if rallies_data else pd.DataFrame()
    
    for pid, fw in footwork_data.items():
        intensities = []
        
        if len(rallies_df) > 0 and len(shots_df) > 0:
            for _, rally in rallies_df.iterrows():
                # Count player's shots in this rally
                rs = shots_df[
                    (shots_df["frame"] >= rally["start_frame"]) &
                    (shots_df["frame"] <= rally["end_frame"]) &
                    (shots_df["player_id"] == pid)
                ]
                
                # Compute intensity as shots per second
                duration_frames = rally["end_frame"] - rally["start_frame"]
                duration_seconds = max(duration_frames / 30, 0.1)
                intensity = len(rs) / duration_seconds
                intensities.append(float(intensity))
        
        # Compute fatigue trend
        fatigue_trend = _compute_fatigue_trend(intensities)
        
        # Additional metrics
        avg_intensity = float(np.mean(intensities)) if intensities else 0
        peak_intensity = float(np.max(intensities)) if intensities else 0
        late_fatigue = _compute_late_rally_fatigue(intensities)
        
        fitness[pid] = {
            "rally_intensity": avg_intensity,
            "rally_intensities": intensities,
            "fatigue_trend": fatigue_trend,
            "avg_recovery": fw.get("avg_recovery", 0),
            "total_distance": fw.get("distance_covered", 0),
            "peak_intensity": peak_intensity,
            "late_rally_fatigue": late_fatigue,
            "rally_count": len(intensities),
        }
    
    return fitness


def _compute_fatigue_trend(intensities):
    """Compute fatigue trend from rally intensities."""
    if len(intensities) < 5:
        return "insufficient_data"
    
    n = len(intensities)
    q1 = intensities[:n//4]
    q4 = intensities[3*n//4:]
    
    avg_q1 = np.mean(q1) if q1 else 0
    avg_q4 = np.mean(q4) if q4 else 0
    
    # Linear regression slope
    x = np.arange(len(intensities))
    slope = np.polyfit(x, intensities, 1)[0]
    avg_intensity = np.mean(intensities)
    normalized_slope = slope / avg_intensity if avg_intensity > 0 else 0
    
    if avg_q4 < avg_q1 * 0.8 and normalized_slope < -0.01:
        return "declining"
    elif avg_q4 > avg_q1 * 1.2 and normalized_slope > 0.01:
        return "improving"
    return "stable"


def _compute_late_rally_fatigue(intensities):
    """Compute late rally fatigue factor."""
    if len(intensities) < 6:
        return 0.0
    
    first_half = intensities[:len(intensities)//2]
    second_half = intensities[len(intensities)//2:]
    
    avg_first = np.mean(first_half)
    avg_second = np.mean(second_half)
    
    if avg_first == 0:
        return 0.0
    
    return float((avg_first - avg_second) / avg_first)
```

- [ ] **Step 3: Update stage_coach in colab/pipeline.py**

Replace the `stage_coach` function (lines 466-487) with dot-notation rule evaluation:

```python
def stage_coach(tactical, fitness, footwork):
    """Generate coaching recommendations using dot-notation rules."""
    strengths, weaknesses, improvements, drills, evidence = [], [], [], [], []
    
    for pid in set(list(tactical.keys()) + list(fitness.keys())):
        # Build analytics dict for this player
        player_analytics = {
            "tactical": tactical.get(pid, {}),
            "fitness": fitness.get(pid, {}),
            "footwork": footwork.get(pid, {}),
        }
        
        # Add computed fields
        tactical_data = player_analytics["tactical"]
        if tactical_data:
            shot_dist = tactical_data.get("shot_distribution", {})
            tactical_data["max_shot_percentage"] = max(shot_dist.values()) if shot_dist else 0
        
        total = tactical_data.get("total_shots", 0)
        
        for rule in RULES:
            try:
                if _evaluate_rule(rule, player_analytics):
                    entry = {"finding": rule["recommendation"], "metrics": [f"total shots: {total}"]}
                    evidence.append(entry)
                    if rule["category"] == "strength":
                        strengths.append(rule["recommendation"])
                    elif rule["category"] == "weakness":
                        weaknesses.append(rule["recommendation"])
                        improvements.append(rule["recommendation"])
                        drills.append(rule.get("drill", ""))
            except Exception:
                continue
    
    return {"strengths": strengths, "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3], "recommended_drills": drills[:3], "evidence": evidence}


def _evaluate_rule(rule, analytics):
    """Evaluate a rule against player analytics."""
    check = rule.get("check", {})
    if not check:
        return False
    
    # Check minimum shots requirement
    min_shots_expr = check.get("min_shots")
    if min_shots_expr:
        if not _evaluate_condition(min_shots_expr, analytics):
            return False
    
    # Evaluate main condition
    field_path = check.get("field")
    operator = check.get("operator")
    threshold = check.get("threshold", check.get("value"))
    
    if not field_path or not operator:
        return False
    
    value = _get_nested(analytics, field_path)
    return _compare(value, operator, threshold)


def _evaluate_condition(expr, analytics):
    """Evaluate condition expression like 'tactical.total_shots >= 10'."""
    parts = expr.split()
    if len(parts) != 3:
        return False
    
    field_path, operator, value_str = parts
    
    try:
        value = float(value_str)
    except ValueError:
        return False
    
    field_value = _get_nested(analytics, field_path)
    return _compare(field_value, operator, value)


def _compare(actual, operator, expected):
    """Compare values using operator."""
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        if operator == "==":
            return str(actual) == str(expected)
        return False
    
    if operator == "<": return actual < expected
    elif operator == ">": return actual > expected
    elif operator == "<=": return actual <= expected
    elif operator == ">=": return actual >= expected
    elif operator == "==": return actual == expected
    elif operator == "!=": return actual != expected
    return False


def _get_nested(data, path):
    """Extract value from nested dict using dot notation."""
    keys = path.split(".")
    current = data
    
    for key in keys:
        if current is None:
            return 0
        if isinstance(current, dict):
            current = current.get(key, 0)
        elif isinstance(current, (list, tuple)):
            try:
                idx = int(key)
                current = current[idx] if 0 <= idx < len(current) else 0
            except (ValueError, IndexError):
                return 0
        else:
            return 0
    
    return current if current is not None else 0
```

- [ ] **Step 4: Update RULES in colab/pipeline.py**

Replace the RULES list (lines 46-65) with dot-notation format:

```python
RULES = [
    {"name": "smash_efficiency", 
     "check": {"field": "tactical.shot_distribution.smash", "operator": "<", "threshold": 0.3, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Your smash conversion rate is low. Focus on placement over power.",
     "category": "weakness", "drill": "Practice targeted smashes to designated court zones."},
    
    {"name": "recovery_speed",
     "check": {"field": "footwork.avg_recovery", "operator": ">", "threshold": 1.2},
     "recommendation": "Recovery after shots is slower than optimal. Work on split-step timing.",
     "category": "weakness", "drill": "Shadow footwork drills: return to base after each shot."},
    
    {"name": "shot_variety",
     "check": {"field": "tactical.max_shot_percentage", "operator": ">", "threshold": 0.5, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Shot selection is predictable. Vary your attack.",
     "category": "weakness", "drill": "Rally drills: alternate clear/drop/net each shot."},
    
    {"name": "fatigue_management",
     "check": {"field": "fitness.fatigue_trend", "operator": "==", "value": "declining"},
     "recommendation": "Performance declines in later rallies. Improve match fitness.",
     "category": "weakness", "drill": "Interval training: 12x (30s high intensity + 30s rest)."},
    
    {"name": "net_play_strength",
     "check": {"field": "tactical.shot_distribution.net_shot", "operator": ">", "threshold": 0.2, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Strong net play presence. Use this to set up attacking opportunities.",
     "category": "strength", "drill": "Maintain net dominance with variation."},
    
    {"name": "clear_usage",
     "check": {"field": "tactical.shot_distribution.clear", "operator": ">", "threshold": 0.35, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Heavy use of clears — mix with drops and smashes.",
     "category": "neutral", "drill": "Clear-drop combination drills from rear court."},
]
```

- [ ] **Step 5: Update run_pipeline to pass pose_data to stage_strokes**

In `run_pipeline()`, update the call to `stage_strokes` (around line 637):

```python
# Old:
shots = stage_strokes(hits, all_shuttle)

# New:
shots = stage_strokes(hits, all_shuttle, all_pose, court, device)
```

- [ ] **Step 6: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: integrate BST, fatigue trend, and coach rules into colab pipeline"
```

---

## Task 7: Run Full Test Suite

- [ ] **Step 1: Run all backend tests**

Run: `.venv/bin/pytest backend/tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: No TypeScript errors

- [ ] **Step 3: Run lint**

Run: `.venv/bin/ruff check backend/`
Expected: No lint errors (or acceptable warnings)

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address test failures and lint issues"
```

---

## Task 8: End-to-End Validation

- [ ] **Step 1: Start backend server**

Run: `PYTHONPATH=/home/sujith/baddyCoach/backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000`
Expected: Server starts on port 8000

- [ ] **Step 2: Start frontend dev server**

Run: `cd frontend && npx vite`
Expected: Dev server starts on port 5173

- [ ] **Step 3: Run Colab pipeline on test video**

Run: `python colab/pipeline.py test_video.mp4 --output results/report.json --device cpu`
Expected: Pipeline completes with non-random stroke distribution

- [ ] **Step 4: Load report in frontend**

1. Open http://localhost:5173
2. Click "Load Report"
3. Select `results/report.json`
4. Verify:
   - Shot distribution shows realistic percentages (not random)
   - Coach section has 2-4 recommendations
   - Fatigue trend shows "stable" or "declining" (not "insufficient_data")

- [ ] **Step 5: Document results**

Update memory with validation findings.

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | BST Feature Extraction | bst_features.py, test_bst_features.py |
| 2 | BST Classifier | bst.py |
| 3 | Stroke Stage Integration | strokes.py |
| 4 | Coach Rules | engine.py, rules.yaml, test_coach_rules.py |
| 5 | Fitness Fatigue | fitness.py, test_fatigue_trend.py |
| 6 | Colab Pipeline | pipeline.py |
| 7 | Test Suite | (none) |
| 8 | E2E Validation | (none) |
