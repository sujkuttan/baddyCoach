# Stroke Classification Fix Plan

## Overview

Five root causes were identified through analysis of two pipeline runs (Run 1: 199 shots, Run 2: 204 shots), cross-referencing parquet data in `results/mmpose_results/debug/` against source code in `backend/app/`.

## Fix 0: Debug Logging Instrumentation

Add structured debug capture across all pipeline stages, gated by `debug_level` in `StageConfig`, so Fixes 1-5 produce measurable output.

### Common Pattern

```python
@dataclass
class StageConfig:
    gpu_enabled: bool = True
    processing_fps: int = settings.processing_fps
    extra: dict[str, Any] = field(default_factory=dict)
    debug_level: int = 0  # 0=off, 1=basic, 2=verbose, 3=dump tensors
```

Debug data is written as parquet/JSONL to `{job_dir}/debug/` — not stdout — using batch/parquet I/O. Each point shows: what to capture, where in code, what format, and why.

### A. Hit Detection (`hits.py`) — Level 2+

| What | Where | Format |
|------|-------|--------|
| Per-frame component scores (raw, before normalization & weighting) | After `_compute_trajectory_change`, `_compute_speed_peaks`, etc. | `debug_hit_scores.parquet` — columns: `frame, trajectory_raw, speed_raw, proximity_raw, swing_raw, combined, is_peak` |
| Peak detection results (prominence, height) | At `find_peaks` call (line 34) | Add columns to above |
| Dedup decisions (which hits merged/removed and why) | In dedup loop (lines 51-58) | Log per merge event |

**Why:** If trajectory score dominates but is noisy, we see it. If proximity is always zero (pose missing), we see it. If peaks look reasonable but dedup removes correct ones, we see it.

### B. Clip Construction (`strokes.py:69-123`) — Level 1+

| What | Where |
|------|-------|
| Per clip: hit frame, start/end, seq_len, original n_frames | Lines 196-231 |
| Missing frame stats: frames with missing bbox, missing pose, missing shuttle | Lines 78-107 |
| Player track ID changes per clip | Lines 58-67 |
| Bbox interpolation tracking | At `det_bbox_lookup.get()` calls (line 106) |

**Storage:** `debug_clips.parquet` — columns: `shot_id, hit_frame, n_frames, n_missing_bbox_far, n_missing_bbox_near, n_missing_pose_far, n_missing_pose_near, n_track_id_changes_far, n_track_id_changes_near, player_order`

**Why:** Directly validates Fix 2 (temporal detection smoothing). The 166 unique track IDs per 18k detections means gaps are everywhere — this measures whether bbox availability improves from ~60% to >95%.

### C. BST Model Inputs (`bst.py:278-294`) — Level 2+

| What | Where |
|------|-------|
| JnB tensor: min, max, mean, zero-fraction per batch | After line 279 (`np.stack`) |
| Shuttle tensor: min, max, zero-fraction | After line 282 |
| Position tensor: min, max | After line 285 |

**Storage:** `debug_bst_features.jsonl` — one JSON line per batch:
```json
{"batch":0, "n_clips":32, "JnB_min":-0.5, "JnB_max":0.6, "JnB_zero_frac":0.12,
 "shuttle_min":-0.2, "shuttle_max":1.5, "shuttle_zero_frac":0.05}
```
Level 3: dump first batch's feature tensors to parquet for offline numpy inspection.

**Why:** If JnB features are mostly zero, the model sees nearly identical input for all clips → same output. Zero-fraction validates Fix 2.

### D. BST Model Outputs (`bst.py:293-316`) — Level 1+ (Critical)

| What | Where |
|------|-------|
| Full softmax distribution over all 25 classes (not just argmax) | After line 294 |
| Pre-softmax logits for class 0 vs max class | After line 293 |
| Temperature value used | Line 294 |
| When pred_idx==0: 2nd-best class ID, confidence, whether 0.3 threshold met | Lines 300-311 |

**Storage:** `debug_bst_outputs.parquet` — columns: `shot_id, pred_class_id, pred_confidence, logit_class_0, logit_max, second_best_class_id, second_best_confidence, is_second_best_override, prob_unknown, prob_smash, prob_clear, prob_lift, prob_drop, prob_net_shot, ...` (top-N as individual cols, full 25-class distribution as JSON blob).

**Why (critical):** The current code throws away 24/25 probabilities per shot. With full distribution we can:
- See if net_shot/drop have non-trivial probability but never win argmax
- Detect uniform distributions (model uncertain) vs peaked at wrong class
- Verify temperature scaling is working
- See if the same few classes dominate regardless of input

### E. Rule-Based Fallback (`bst.py:335-376`) — Level 1+

| What | Where |
|------|-------|
| Input shuttle stats at decision: mean_speed, max_speed, mean_dy, end_y, n_valid | Lines 358-361 |
| Predicted stroke type and which threshold branch triggered | Lines 363-376 |

**Storage:** Add columns to shots.parquet for rule-based shots: `[mean_speed, max_speed, mean_dy, end_y, n_shuttle_valid, trigger_branch]`

**Why:** Validates Fix 1 (normalization correction) — confirms rule-based classifier now produces diverse stroke types.

### F. Player Attribution (`attribution.py:70-140`) — Level 1+

| What | Where |
|------|-------|
| Attribution tier used per shot: bst_side, shuttle_direction, rally_alternation, final_fallback | Tracks 1-4 (lines 76, 97, 119, 139) |
| BST side mapping if used | Lines 85-92 |

**Storage:** Add `attribution_tier` column to shots.parquet.

### G. Rally End Reason (`shared/utils.py:160-175`) — Level 1+

| What | Where |
|------|-------|
| For last shot of each rally: stroke_type, confidence, triggered branch | Lines 169-175 |
| Frame gap after last shot (next_gap) | From `rallies.py` |

**Storage:** Add `end_reason_debug` columns to rallies.parquet.

### H. `print()` Cleanup

Migrate ~15 `print()` calls in `bst.py` to `logger.info()` / `logger.debug()` gated on debug_level.

### Debug Output Layout

```
results/{job_id}/debug/
├── debug_hit_scores.parquet     # Per-frame component scores
├── debug_clips.parquet           # Per-clip construction metadata
├── debug_bst_features.jsonl      # Per-batch feature tensor stats
├── debug_bst_outputs.parquet     # Per-shot full softmax distribution
└── debug_rallies.json            # Per-rally end-reason details
```

---

## Fix 1: Rule-Based Classifier Normalization

**Files:**
- `backend/app/models/bst.py:335-376` — `_rule_based_predict`
- `backend/app/pipeline/strokes.py:124-129` — `_build_clip` return dict

**Problem:**
Clip construction (`strokes.py:82-83`) normalizes shuttle by court dimensions (`x/13.4`, `y/6.1`), but rule-based thresholds (`bst.py:363-376`) were designed for pixel-space normalization (`x/1920`, `y/1080` → range [0,1]).

- `end_y` ALWAYS negative after court-normalization → "lift" (needs `end_y > 0.5`) and "drop" (needs `end_y > 0.7`) can NEVER trigger
- `mean_speed` ALWAYS > 0.03 → "net_shot" can NEVER trigger
- Most trajectories fall through to "drive" or "unknown"

**Secondary issue:** Between-2-hits clips span ~3.3s (100 frames at 30fps), covering BOTH incoming shuttle (toward player) and outgoing shuttle (away from player). The trajectory direction reverses at hit point → V-shaped average → "drive"-like signal.

**Fix steps:**
1. Add `vid_w`, `vid_h` to the clip dict returned by `_build_clip`
2. In `_rule_based_predict`, before threshold checks:
   - Denormalize shuttle by `(court_length, court_width)`
   - Renormalize by `(vid_w, vid_h)`
   - Extract only POST-HIT frames (midpoint to end of clip)
3. Apply existing pixel-space thresholds on the renormalized post-hit trajectory

**Validation:** Run on `results/mmpose_results/debug/` — verify 69 rule-based shots produce varied types (clear, smash, drive, net_shot, drop) instead of all "drive".

```python
def _rule_based_predict(self, clip: dict) -> str:
    shuttle = clip['shuttle']  # (seq_len, 2), normalized by court dims
    vid_w, vid_h = clip['vid_w'], clip['vid_h']
    court_len = clip.get('court_length', 13.4)
    court_wid = clip.get('court_width', 6.1)

    # Denormalize back to court-space meters, then to pixel-space
    court_shuttle = shuttle * [court_len, court_wid]
    pixel_shuttle = court_shuttle / [vid_w, vid_h]

    # Take only post-hit half
    mid = len(pixel_shuttle) // 2
    post_hit = pixel_shuttle[mid:]

    # Apply existing thresholds on post_hit
    ...
```

---

## Fix 2: Temporal Detection Smoothing

**Files:**
- `backend/app/pipeline/strokes.py:69-76` — `det_bbox_lookup` construction
- `backend/app/pipeline/strokes.py:101-107` — bbox usage in clip construction

**Problem:**
Per-frame YOLO tracking produces 166 unique track IDs for 18,000 detections across just 2 players. When `frame_player_map` or `det_bbox_lookup` fails:
- `_get_keypoints_for_frame` returns None → joints left as zeros
- `normalize_joints` falls back to keypoint bbox (less stable)
- BST features become garbled → low confidence / wrong class

**Fix steps:**
1. Build per-player temporal bbox buffers: for each known player ID, collect frames where a bbox exists
2. At clip construction, linearly interpolate bbox for missing frames
3. Apply interpolation to `frame_player_map` too (player side assignment)
4. When `det_bbox_lookup` returns None, use interpolated bbox

```python
def _interpolate_player_bboxes(lookup: dict, player_id: str, frames: list):
    dets = lookup.get(player_id, {})
    existing = sorted(dets.keys())
    if not existing:
        return {}
    result = {}
    for f in frames:
        if f in dets:
            result[f] = dets[f]
        else:
            before = [ef for ef in existing if ef <= f]
            after = [ef for ef in existing if ef >= f]
            if before and after:
                bf, af = before[-1], after[0]
                if bf == af:
                    result[f] = dets[bf]
                else:
                    ratio = (f - bf) / (af - bf)
                    result[f] = tuple(
                        dets[bf][i] + ratio * (dets[af][i] - dets[bf][i])
                        for i in range(4)
                    )
            elif before:
                result[f] = dets[before[-1]]
            elif after:
                result[f] = dets[after[0]]
    return result
```

**Validation:** After fix, `normalize_joints` should have valid `det_bbox` for ≥95% of frames (currently ~60-70%). Verify via debug_clips.parquet.

---

## Fix 3: BST Class Ordering Verification

**Files:**
- `backend/app/models/bst.py:17-24` — `SHUTTLESET_CLASSES` array
- `backend/app/models/bst.py:27-35` — `map_to_coach_class`
- Checkpoint at `BST/weight/bst_CG_JnB_bone_merged.pt`

**Problem:**
BST outputs only 7 of 25 class IDs (3, 4, 5, 16, 17, 18, 23) across both runs — never net_shot, drop, push, block, rush, cross_court. Mean confidence: 0.26. Hypothesis: class ordering mismatch between `SHUTTLESET_CLASSES` and training checkpoint.

**Fix steps:**
1. Load checkpoint, inspect MLP head weight matrix shape to confirm `n_classes=25`
2. Find/create test sample per stroke type with known ground truth
3. Run feature extraction → BST inference → compare predicted class ID vs expected
4. If mismatch: reorder `SHUTTLESET_CLASSES` to match checkpoint ordering
5. If no labeled samples: add validation script that logs class activation distribution across all clips via debug_bst_outputs.parquet

```python
import torch
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt if isinstance(ckpt, dict) else ckpt.get('state_dict', ckpt)
mlp_weight = state['mlp_head.mlp.mlp.3.weight']
print(f'n_classes={mlp_weight.shape[0]}, in_dim={mlp_weight.shape[1]}')
```

---

## Fix 4: Temporal Smoothing Scope

**Files:**
- `backend/app/pipeline/strokes.py:269-285`

**Problem:**
Line 273: `if stype != "unknown": continue` — only "unknown" strokes get smoothed. Low-confidence determinate predictions (BST "drive" at conf=0.089) remain untouched even when surrounded by "smash" or "clear".

**Fix steps:**
1. Change condition:
   ```python
   if stype != "unknown" and conf > 0.2:
       continue
   ```
2. Shots with conf < 0.2 get majority-vote smoothing alongside "unknown"

**Validation:** Isolated "drive" (conf=0.089) surrounded by smashes/clears should be corrected to majority type.

---

## Fix 5: Rally Winner Confidence Threshold

**Files:**
- `backend/app/pipeline/shared/utils.py:160-175` — `_infer_end_reason`
- `backend/app/pipeline/rallies.py:28` — `_compute_rally_winner_after_attribution`

**Problem:**
`_infer_end_reason` requires conf ≥ 0.5 for "winner", max BST conf is 0.633, rule-based max is 0.3. 13/14 rallies → "unforced_error". Winner = "player who didn't hit last shot" — misses genuine winners.

**Fix steps:**
1. Lower winner threshold from 0.5 to 0.3 for smash/drop/kill
2. Add trajectory-speed-based winner detection (smash with speed > 8 m/s or within 2m of net = winner)
3. `_compute_rally_winner_after_attribution` already handles winner correctly (winner = last_pid when end_reason == "winner")

**Validation:** Rallies ending in smashes should show `end_reason = "winner"` instead of "unforced_error".

---

## Implementation Order

| Order | Fix | Files | Dependencies | Effort |
|-------|-----|-------|-------------|--------|
| 0 | Debug logging infra | `base.py`, `bst.py`, `strokes.py`, `hits.py`, `attribution.py`, `utils.py` | None (do first) | 3h |
| 1 | Rule-based normalization | `bst.py`, `strokes.py` | None | 2h |
| 2 | Temporal detection smoothing | `strokes.py` | None | 3h |
| 3 | BST class ordering verification | `bst.py` | None (research) | 4h |
| 4 | Temporal smoothing scope | `strokes.py` | None | 30m |
| 5 | Rally winner threshold | `utils.py`, `rallies.py` | Fix 1 (better types) | 1h |

Fix 0 must come first — every subsequent fix needs debug output to validate correctness. Fixes 1, 2, 4 are independent and can be parallelized.

## Verification

```bash
cd backend && python -m pytest -m "not gpu and not model"

python3 -c "
import pandas as pd
shots = pd.read_parquet('results/mmpose_results/debug/shots.parquet')
print('Rule-based types:', shots[shots['is_rule_based']]['stroke_type'].value_counts().to_dict())
print('BST types:', shots[~shots['is_rule_based']]['stroke_type'].value_counts().to_dict())
print('Mean confidence:', shots['stroke_confidence'].mean())
"

ls -la results/{job_id}/debug/
```

**Acceptance criteria:**
- Rule-based shots show ≥3 different stroke types (not all "drive")
- BST includes net_shot, drop, or push (at least occasionally)
- Mean BST confidence ≥ 0.40 (was 0.26)
- Rally end_reasons include "winner" where appropriate (not 100% "unforced_error")
- Temporal smoothing corrects low-confidence outliers
