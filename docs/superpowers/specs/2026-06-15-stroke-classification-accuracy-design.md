# Design: Stroke Classification Accuracy & Pipeline Improvements

**Date:** 2026-06-15
**Priority:** CRITICAL — Rally/stroke identification is the most important product metric

## Problem Statement

The current pipeline has three critical gaps that undermine product accuracy:

1. **BST Model Not Integrated**: `stage_strokes` uses random classification based on shuttle y-position instead of the actual BST neural network
2. **Coach Rules Don't Match Data**: Rules check for values that don't exist in the current data format
3. **Fatigue Trend Always Returns "insufficient_data"**: No real per-rally intensity analysis

**Goal:** Accurate rally → stroke identification → correct classification into major types (clear, smash, drop, etc.) → subtypes (flick_serve, net_shot, etc.)

---

## Design Approach

### 1. BST Model Integration with Robust Feature Extraction

**Challenge:** BST model weights exist (`bst_CG_JnB_bone_merged.pt`) but the model architecture is unknown. The checkpoint is a raw state_dict without a bundled model class.

**Solution: Multi-stage feature extraction pipeline**

```
Frame Data → Feature Extractor → 144-dim Vector → BST Classifier → Stroke Type
```

**Feature Extraction Strategy:**

| Feature Group | Source | Dimensions | Description |
|---------------|--------|------------|-------------|
| Shuttle trajectory | TrackNetV3 | 24 | Position, velocity, acceleration over 8-frame window |
| Shuttle position | TrackNetV3 | 6 | Current x, y, speed, direction |
| Pose joints (hitter) | RTMPose | 48 | 17 keypoints × (x, y) normalized by bbox |
| Pose dynamics | RTMPose | 12 | Joint velocities (wrist, elbow, shoulder) |
| Body orientation | RTMPose | 6 | Torso angle, lean, arm extension |
| Court position | Homography | 6 | Normalized court coordinates (x, y) in meters |
| Rally context | Sequence | 42 | Previous 3 shots: type (one-hot 12), confidence, frame gap |

**Total: 144 dimensions**

**BST Classifier Implementation:**

```python
class BSTClassifier:
    def __init__(self, model_path, device="cuda"):
        # Try multiple architecture patterns
        self.model = self._load_with_fallback(model_path, device)
    
    def _load_with_fallback(self, path, device):
        # Pattern 1: Checkpoint contains model object
        # Pattern 2: Checkpoint contains state_dict for known architecture
        # Pattern 3: Try common BST architectures (MLP, ResNet-1D, Transformer)
        # Fallback: Random classification with warning
        pass
    
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        # features: (144,) normalized vector
        # Returns: (stroke_type, confidence)
        pass
```

**Fallback Strategy (if BST fails to load):**

Use rule-based classification using shuttle trajectory + pose features:
- High shuttle speed + downward angle → **smash**
- Shuttle near net (y < 300) + low speed → **net_shot**
- Shuttle trajectory arc (parabolic) → **clear/lift**
- Shuttle fast + horizontal → **drive**
- Shuttle slow + near court → **drop**

### 2. Coach Rules Data Format Alignment

**Current Problem:** Rules check `d.get("smash", 0)` but `d` is a dict with percentage values like `{"smash": 0.25, "clear": 0.30}`.

**Solution:** Rewrite rules to match actual data structure:

```yaml
rules:
  - name: smash_efficiency
    check:
      field: tactical.shot_distribution.smash
      operator: "<"
      threshold: 0.3
      min_shots: tactical.total_shots >= 10
    recommendation: "Your smash conversion rate is low..."
    category: weakness
    drill: "Practice targeted smashes..."
    
  - name: recovery_speed
    check:
      field: footwork.avg_recovery
      operator: ">"
      threshold: 1.2
    recommendation: "Recovery after shots is slower..."
    category: weakness
    drill: "Shadow footwork drills..."
    
  - name: shot_variety
    check:
      field: tactical.max_shot_percentage
      operator: ">"
      threshold: 0.5
      min_shots: tactical.total_shots >= 20
    recommendation: "Shot selection is predictable..."
    category: weakness
    drill: "Rally drills with constraint..."
    
  - name: fatigue_management
    check:
      field: fitness.fatigue_trend
      operator: "=="
      value: "declining"
    recommendation: "Performance declines in later rallies..."
    category: weakness
    drill: "Interval training..."
    
  - name: net_play_strength
    check:
      field: tactical.shot_distribution.net_shot
      operator: ">"
      threshold: 0.2
      min_shots: tactical.total_shots >= 10
    recommendation: "Strong net play presence..."
    category: strength
    drill: "Maintain net dominance..."
    
  - name: clear_usage
    check:
      field: tactical.shot_distribution.clear
      operator: ">"
      threshold: 0.35
      min_shots: tactical.total_shots >= 10
    recommendation: "Heavy use of clears..."
    category: neutral
    drill: "Clear-drop combination drills..."
```

**Engine Implementation:**

```python
class CoachEngine:
    def evaluate_rule(self, rule, analytics):
        check = rule["check"]
        
        # Extract value from nested dict path
        value = self._get_nested(analytics, check["field"])
        
        # Check minimum shot requirement
        if "min_shots" in check:
            min_val = self._get_nested(analytics, check["min_shots"])
            if not min_val:
                return False
        
        # Evaluate condition
        threshold = check.get("threshold", check.get("value"))
        op = check["operator"]
        
        if op == "<": return value < threshold
        elif op == ">": return value > threshold
        elif op == "==": return value == threshold
        elif op == "!=": return value != threshold
        return False
    
    def _get_nested(self, data, path):
        """Extract value from nested dict using dot notation.
        Example: tactical.shot_distribution.smash -> data["tactical"]["shot_distribution"]["smash"]
        """
        keys = path.split(".")
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, 0)
            else:
                return 0
        return data
```

### 3. Fitness Fatigue Trend with Real Per-Rally Intensity

**Current Problem:** Returns "insufficient_data" because recovery_times list is empty.

**Solution:** Compute fatigue trend from rally intensity pattern:

```python
def compute_fatigue_trend(rally_intensities: list[float]) -> str:
    """
    Analyze rally intensity over time to detect fatigue.
    
    Args:
        rally_intensities: List of intensity values (shots/second) per rally, in chronological order
    
    Returns:
        "improving" | "stable" | "declining" | "insufficient_data"
    """
    if len(rally_intensities) < 5:
        return "insufficient_data"
    
    # Split into quarters for trend analysis
    n = len(rally_intensities)
    q1 = rally_intensities[:n//4]
    q2 = rally_intensities[n//4:n//2]
    q3 = rally_intensities[n//2:3*n//4]
    q4 = rally_intensities[3*n//4:]
    
    avg_q1 = np.mean(q1) if q1 else 0
    avg_q4 = np.mean(q4) if q4 else 0
    
    # Calculate trend using linear regression
    x = np.arange(len(rally_intensities))
    slope = np.polyfit(x, rally_intensities, 1)[0]
    
    # Combine quarter comparison with slope
    if avg_q4 < avg_q1 * 0.8 and slope < 0:
        return "declining"
    elif avg_q4 > avg_q1 * 1.2 and slope > 0:
        return "improving"
    return "stable"
```

**Enhanced Fitness Stage Output:**

```python
fitness[player_id] = {
    "rally_intensity": float(np.mean(intensities)),
    "rally_intensities": intensities,
    "fatigue_trend": compute_fatigue_trend(intensities),
    "avg_recovery": compute_avg_recovery(pose_data, player_id),
    "total_distance": compute_distance(pose_data, player_id),
    "peak_intensity": float(np.max(intensities)),
    "intensity_std": float(np.std(intensities)),
    "late_rally_fatigue": compute_late_rally_fatigue(intensities),
}
```

---

## Implementation Plan

### Phase 1: BST Feature Extraction & Classification

**Files to modify:**
- `colab/pipeline.py` — Replace `stage_strokes` with proper BST integration
- `backend/app/pipeline/strokes.py` — Update feature extraction

**New files:**
- `backend/app/models/bst_features.py` — Feature extraction pipeline

**Key changes:**
1. Implement `extract_bst_features()` that computes 144-dim vector from shuttle + pose + court data
2. Update `BSTClassifier` to handle multiple checkpoint formats
3. Add rule-based fallback if BST fails to load
4. Validate feature extraction with test data

### Phase 2: Coach Rules Alignment

**Files to modify:**
- `colab/pipeline.py` — Rewrite `stage_coach` with dot-notation rule evaluation
- `backend/app/coach/engine.py` — Add `_get_nested()` helper
- `backend/app/coach/rules.yaml` — Update to new format

**Key changes:**
1. Rewrite rules to use dot-notation field paths
2. Implement `_get_nested()` for extracting values from analytics dict
3. Update rule evaluation logic to handle operators
4. Test with actual pipeline output

### Phase 3: Fitness Fatigue Trend

**Files to modify:**
- `colab/pipeline.py` — Update `stage_fitness` with real fatigue computation
- `backend/app/pipeline/analytics/fitness.py` — Add `compute_fatigue_trend()`

**Key changes:**
1. Implement `compute_fatigue_trend()` using quarter comparison + linear regression
2. Add `late_rally_fatigue` metric
3. Ensure rally_intensities list is populated correctly
4. Validate with multi-rally test cases

---

## Testing Strategy

### Unit Tests
- `test_bst_features.py` — Feature extraction dimensions, normalization
- `test_coach_rules.py` — Rule evaluation with various analytics inputs
- `test_fatigue_trend.py` — Fatigue detection with synthetic intensity data

### Integration Tests
- Run full pipeline on test video
- Verify stroke classification produces reasonable distribution (not random)
- Verify coach rules trigger with actual data
- Verify fatigue trend shows "stable" or "declining" (not "insufficient_data")

### Validation Metrics
- Stroke classification: Compare BST output distribution against expected badminton shot patterns
- Coach rules: Count triggered rules (should be 2-4 per player, not 0)
- Fatigue trend: Should be "stable" or "declining" for 30+ minute videos

---

## Success Criteria

1. **BST Classification:** Stroke types follow realistic badminton distribution (clear 20-30%, smash 10-20%, drop 15-25%, net_shot 10-15%, etc.)
2. **Coach Rules:** At least 2 rules trigger per player (strengths + weaknesses)
3. **Fatigue Trend:** Returns "stable" or "declining" for videos with 5+ rallies
4. **End-to-End:** Colab pipeline → JSON report → Frontend displays correct stroke types and coach recommendations

---

## Risk Mitigation

**BST Model Load Failure:**
- Fallback to rule-based classification using shuttle trajectory + pose
- Log warning but continue pipeline
- Mark stroke_confidence as low (0.3-0.5)

**Feature Extraction Errors:**
- Validate input data before extraction
- Return zero vectors with low confidence if data missing
- Log which features were unavailable

**Coach Rules Triggering Too Many/Too Few:**
- Start with conservative thresholds
- Adjust based on real match data
- Add `min_shots` requirement to prevent false positives

---

## Next Steps

1. Implement BST feature extraction pipeline
2. Test BST classifier with checkpoint
3. Rewrite coach rules engine
4. Implement fatigue trend computation
5. Run full pipeline on test video
6. Validate results match success criteria
