# IMPROVE HIT DETECTION & PLAYER ATTRIBUTION — Phone-Recorded Pause-And-Record Video

## 1. Problem Analysis (Pause-and-Record → Low Confidence, Systematic Bias)

| Current Assumption | Phone-Recording Reality | Impact on Your Pipeline |
|-------------------|------------------------|-------------------------|
| **Continuous video** | Sudden scene-cuts (pause → record) between rallies | `rally_gap_threshold` fails: post-pause rally frames follow instantly. |
| **Fixed camera** | Slight phone tilt/conversion changes between rallies | Court coordinates drift, `homography` validity decays, side-assignment breaks. |
| **High frame rate** | Variable FPS (phone auto-adjusts) | `settings.fps` mismatch between shuttle speed calculation and clip extraction. |
| **Fast rally alternation** | Slower amateur rallies, errors, time between shots | `_find_peaks(distance=3)` may merge distinct hits. |
| **Near-far side dynamic** | Camera often moves closer to one side | `image_to_court(H, ...)` provides a false sense of spatial truth since `H` is static per-video.. |

### Why Your Stroke/Attribution Counts Are Unfair:
1. **Missing strokes**: The phone stops rolling between points. Tennis scoring relies on the "near-far" geometry, but that geometry is invalidated when the user repositions.
2. **Player bias**: If Bobby Murray (the closer opponent) is in frame more often, his pose features dominate. BST defaults to the dominant class if features lack discriminative signal.
3. **Scene cuts as noise**: Sudden jumps in shuttle coordinates are misinterpreted as `"reversal"` signals.

---

## 2. Targeted Technical Recommendations (Code-Level)

### 2.1 Hit Detection: Temporal Dedup Logic
**Current `hits.py` logic error:**

```python
min_gap = max(3, int(fps * settings.hit_dedup_gap_seconds))
# For fps=30, gap = max(3, 0.2*30) = 6 frames.
```

**Issue:** On amateur footage, a rally might have 2-3 shots within 1 second. An `0.2s` dedup window (6 frames) might drop valid stroke detections.

**Fix in `hits.py`** (line 79-88):

```python
def run(...):
    fps = float(config.processing_fps or settings.fps)
    if hasattr(config, 'source_fps') and config.source_fps > 0:
        fps = config.source_fps
        
    dedup_seconds = settings.hit_dedup_gap_seconds
    if config.video_source == "phone_paused":
        dedup_seconds = 0.5  # Lower frame VFX rate means fewer samples
        
    min_gap = max(3, int(fps * dedup_seconds))
```

- If you expose video metadata to the config, you can dynamically gate based on phone footage.

### 2.2 Scene-Cut Detection (Pause/Record Boundaries)
**Currently:** The `rally_ending` logic completely relies on `stroke_type`, which misclassifies `scene_cut` (line 183-184).

```python
# In rallies.py line 183: Affects Rally Segmentation
scene_cut = disp.max() > 50 * np.median(disp)
```

**Problem:** This heuristic is too simplistic. The shuttle doesn't jump 50x its median when paused. Instead, the pause causes a frame skip of unknown length.

**Robust fix (Scene Segmentation):**
Instead of looking at spatial jumps, detect *temporal* jumps via frame numbers.

```python
import numpy as np

def detect_temporal_gaps(shuttle_df, threshold_frames=60):
    """Returns True where a scene cut (pause/record) is likely."""
    if shuttle_df is None or len(shuttle_df) < 5:
        return np.zeros(len(shuttle_df), dtype=bool)
    
    # Frame number difference
    frame_nums = shuttle_df['frame'].values
    dt = np.diff(frame_nums, prepend=frame_nums[0])
    
    # Where frame gap is unexpectedly large (pause/record jump)
    gaps = dt > threshold_frames
    return gaps
```

**Where to apply:**
1. **During `RallySegmentation`**: `scene_cut = detect_temporal_gaps(...)`. Instead of using a static threshold, align with actual frame number deltas.
2. **During `HitFrameLocalization`**: Gate `find_peaks` on gaps.

### 2.3 Player Attribution: Leveraging Asymmetric Court Layout
**Current logic** (`attribution.py, 56-64`):

```python
def _shuttle_direction_at(frame):
    # ...
    for lb in range(1, LOOKBACK + 1):
        # y_prev is 5 frames behind
```

**Problem:** This assumes the shuttle moves vertically (down for `player_1`, up for `player_2`).

**Phone-specific fix:**
If the server is always recording from one side of the net:
- **`player_1`** (bottom of frame in raw pixels but in-phone, air, *push up* is `dy < 0` — this is exactly what your literal comment describes).
- **`player_2`** (top of frame — compute absolutely instead of using side-assignment if confidence is low).

So inside `_shuttle_direction_at`:

```python
# Instead of absolute player assignment, use verticality when confidence is < 0.2:
if (abs(y_prev) > 1 and ...):
    dy = y_at - y_prev
    if abs(dy) > 2:
        return "player_1" if dy >  jouer else "player_2"
# Add:
    if abs(dy) <= 2: # Nearly horizontal, near net
        return None # Force unknown, avoid false-confidence assignment
```

Then in `Tier 3` (rally alternation), handle `None` safely:

```python
if last_player is not None:
    # ...existing logic
else:
    # We don't have a prior yet
    if r_frame - shots_df.iloc[prev_idx]["frame"] > LARGE_GAP:
        shots_df.at[idx, "player_id"] = None # Don't force attribution
        shots_df.at[idx, "attribution_tier"] = "unclassified_scene_cut"
```

### 2.4 Stroke Classification: Physics Ensemble
You already use this in `strokes.py`, but for pause-record video:

```python
# In train_model, add phone-specific physics hints:
def apply_phone_record_ensemble(shuttle_vx, shuttle_vy, pose_features):
    """
    For phone footage where playback is paused, the shuttle speed 
    and direction have physical meaning we should trust over 
    weak model confidence (because model was trained on continuous footage).
    """
    speed = np.sqrt(shuttle_vx**2 + shuttle_vy**2)
    
    # If speed is very low and pose is "reaching", classify as net_shot
    if speed < 1.5 and pose_features["arm_extension"] > 0.8:
        return "net_shot", 0.7  # High-confidence override
        
    # If speed is very high vertical (>2m/s in Y), classify as smash
    if shuttle_vy > 2.0:
        return "smash", 0.65
        
    return None, 0.0
```

Then add this to the confidence-weighted ensemble in `bst.py` (or wherever `apply_physics_ensemble` is called).

---

## 3. Template-Level / UI Suggestions
As a quick UI fix for user trust:
- **`player_1` vs `player_2` → Show confidence badges**: Green for `bst_side`, Orange for `shuttle_direction`, Red for `rally_alternation`.
- **"Scene cut detected" warning**: On the report, flag rallies where `rally_id` jumps by a >60 frame gap so the coach knows the model had to infer.

---

## 4. Summary of Recommendations Table

| Area | Current Approach | Improved Approach (Phone/Amateur) |
|------|-----------------|----------------------------------|
| **Hit Detection** | `distance=0.2s` dedup; `find_peaks` | Reduce dedup gap to `0.5s`, add frame-number gap detection for pause/record. |
| **Rally Segmentation** | Static `gap_threshold=90` | Dynamic gap based on temporal frame numbers. |
| **Player Attribution** | `bst_min_conf=0.5` + `shuttle_direction` | Lower conf. to `0.3` for amateurs; if `dy` is small (<2), return unknown. |
| **Body Reversal** | Relies on spatial threshold | Increase threshold for amateur/phone shakes. |
| **Edge Case** | Assumes continuous rally | Explicit "scene-cut" flags with `None` player_id. |

---

## Bottom Line

Pause-and-record video breaks your core assumption that Yolo and TrackNet enjoy temporally consistent video. You don't need better models — you need **smarter scene segmentation, physics-based rule overrides, and a UI that exposes the model's uncertainty** so the user can manually correct the 15% of edge-case strokes.
