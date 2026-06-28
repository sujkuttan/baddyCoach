# Additional Hit Detection & Stroke Classification Improvements

## Context
Improvements for phone-recorded, pause-and-record badminton video. These are high-impact, validated fixes for the current pipeline.

---

## 1. Critical: Scene-Cut Detection (Frame Gaps)

**Problem:** Pause-and-record creates frame gaps where the shuttle "teleports" between rallies. Current detection CoDe(`rallies.py:183-184`) uses spatial displacement (`disp.max() > 50 * median`), but pauses often have no shuttle points at all between segments, so `len(seg) < 5` is skipped.

**Impact:** Every scene cut produces 1-3 false hit detections from `reversal_score` (teleport looks like direction reversal) and `speedieee` (massive frame-to-frame displacement reads as speed peak).

**Fix:** Add temporal gap detection based on frame numbers, not just spatial displacement:

```python
def detect_scene_cuts(shuttle_df, threshold_frames=60):
    """Returns mask of frame indices where a scene cut (pause/record) is likely."""
    frame_nums = shuttle_df['frame'].values
    dt = np.diff(frame_nums, prepend=frame_nums[0])
    return dt > threshold_frames
```

**Apply in:**
- `hits.py`: Zero out signal at cut boundaries
- `strokes.py`: Never build clips across a cut
- `rallies.py`: Force a new rally at every cut

---

## 2. Critical: Fix Between-2-Hits Clip Construction

**Problem:** `strokes.py._build_clip` default: `end_frame = min(frame + seq_len, next_hit_frame)`. For long pauses, `next_hit` is 5+ seconds (150+ frames) away; clip is `seq_len` frames (100) with most being zero-padded or single-direction shuttle flight.

**Impact:** BST sees a 100-frame clip with 2-3s of dead air. Produces `unknown` and `drive` misclassifications.

**Fix:** Construct clips based on **post-hit trajectory** only, not gap to next hit:

```python
# Use a fixed window from the hit (e.g., 30-60 frames post-hit)
clip_frames = range(hit_frame, min(hit_frame + 60, len(frames)))
```

For physics fallback, use only the **outgoing** trajectory:

```python
post_hit_shuttle = clip_shuttle[clip_shuttle["frame"] >= hit_frame]
# Compute trajectory features from post_hit_shuttle only
```

---

## 3. High Impact: Racket-Arm Proximity Attribution

**Problem:** `attribution.py:56-64` uses `dy > 0 -> player_1, dy < 0 -> player_2`. Assumes fixed camera angle. The balance check then **blindly flips ALL attributions** in a rally if counts are imbalanced.

**Fix:** Use **racket-arm proximity** at hit frame. Camera-angle-independent:

```python
def attribute_by_racket_proximity(hit_frame, shuttle_pos, poses):
    """Return player_id whose wrist is closest to shuttle at hit frame."""
    min_dist = float('inf')
    hitter = None
    for player_id, pose in poses.items():
        wrist = pose.get_wrist_at(hit_frame)
        if wrist is None:
            continue
        dist = euclidean_distance(wrist, shuttle_pos)
        if dist < min_dist:
            min_dist = dist
            hitter = player_id
    return hitter
```

- `wrist` = midpoint of left/right wrist keypoints from RTMPose
- If pose missing, fall back to player bbox center
- Only use shuttle direction as **Tier 3**, not primary

---

## 4. High Impact: Resolution-Independent Thresholds

**Problem:** Hardcoded pixel thresholds perform inconsistently across phone resolutions.

| Code | Value | @720p | @4K |
|------|-------|-------|-----|
| `reversal` | `abs(dy) > 1.0` | Strict | Noise |
| `dead_shuttle` | `4.0 px/frame` | Reasonable | Too strict |
| `proximity` | `dist /  hints0` | Very sensitive | Insensitive |

**Fix:** Normalize by video diagonal:

```python
DIAG = np.sqrt(frame_width**2 + frame_height**2)

# Reversal threshold: 0.1% of video diagonal
reversal_threshold = 0.001 * DIAG  # ~2.5px @1080p, ~4.6px @4K

# Dead shuttle: 0.3% of diagonal, frame-rate adjusted
dead_speed_threshold = 0.003 * DIAG / fps
```

---

## 5. Medium Impact: Data Quality Flags Per Shot

**Problem:** No way to express uncertainty. Fallback to rule-based, missing pose, or interpolation across a cut is invisible.

**Fix:** Add `data_quality` column to `shots_df` with bit flags:

```python
QUALITY_FLAGS = {
    "pose_missing": 0b0001,
    "bbox_interpolated": 0b0010,
    "across_scene_cut": 0b0100,
    "rule_based_fallback": 0b1000,
}
```

Expose in UI for user review.

---

## 6. Medium Impact: Adaptive Clip Length

**Problem:** `bst_min_clip_frames = 15` too low, `seq_len = 100` (3.3s) too long for fast exchanges. Zero-padded short clips = weak signal.

**Fix:** Use variable lengths based on typical flight times:

| Stroke | Duration | Clip Length |
|--------|----------|-------------|
| Serve | 1.5-2.0s | 45-60 frames |
| Net/drive | 0.3-0.5s | 10-15 frames |
| Clear/smash | 0.8-1.2s | 24-36 frames |

Or simply: **clip to next hit or 60 frames, whichever is shorter**, then pad to `seq_len`.

---

## Summary Table

| # | Improvement | Priority | Expected Impact |
|---|-------------|----------|-----------------|
| 1 | **Scene-cut detection** (temporal frame gaps) | Critical | Removes 30-50% of false hits |
| 2 | **Post-hit-only clip construction** | Critical | Fixes "between-2-hits" invalid clips |
| 3 | **Racket-arm proximity attribution** | High | Camera-angle-independent |
| 4 | **Resolution-independent thresholds** | High | Consistent across resolutions |
| 5 | **Per-shot data quality flags** | Medium | Enables user review |
| 6 | **Adaptive clip lengths** | Medium | Better model inputs |

---

## What NOT to Bother With (For Now)

| Item | Why Skip |
|------|----------|
| Widen dedup gap (0.2s → 0.5s) | Already rejected — harms rapid exchanges |
| BST retraining | Too expensive without phone footage dataset |
| Court detection overhaul | Secondary — attribution works without perfect court |
| Frame interpolation | Current linear interpolation is sufficient |
