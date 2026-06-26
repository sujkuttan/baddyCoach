# Pipeline Quality Spec — Shuttle Cleaning, Rally Segmentation, Track Stitching

Implementation spec for the three highest-leverage data-quality fixes identified from the
phone-footage Colab run (`logs/debug/*.parquet`). All three are **GPU-free** to implement and
unit-test, and each lifts BST stroke quality, hit detection, and the planned rule-based
classifier simultaneously.

**Suggested order:** (1) Shuttle cleaning → unblocks everything downstream. (2) Track stitching →
stabilizes BST's joint channel. (3) Rally + winner → independent, gives honest stats.

---

## Evidence baseline (from the latest run)

| Artifact | Metric | Value | Implication |
|---|---|---|---|
| `shuttle.parquet` | confidence mean | 0.45 | weak tracking |
| | frac conf > 0.5 | 0.39 | only 39% high-conf |
| | single-frame jumps > 200px | **281** (80% low-conf) | teleport noise |
| | frac conf < 0.3 | 11% | gate-droppable points |
| `rallies.parquet` | rally 11 | **69 shots / 78s** | mega-rally, no internal gap > 90 |
| | rally 3 | **44 shots / 55s** | mega-rally |
| | winner split | **player_2 = 11/13** | implausible skew |
| `player_detections.parquet` | distinct track_ids | **166** (141 far, 75 near) | heavy ID churn for 2 players |
| `pose.parquet` | keypoint / wrist conf | 0.71 / 0.70 | healthy — not a bottleneck |

---

# Spec 1 — Shuttle trajectory cleaning (highest leverage)

## Problem
TrackNet output is noisy: 281 single-frame >200px jumps (80% low-confidence), 11% of frames below
conf 0.3. This noise propagates into hit detection (`debug_hit_scores.trajectory_raw`/`speed_raw`),
BST's shuttle channel, rally-dead detection, and any rule-based classifier. Fix once, everything
downstream improves.

## Where
New post-processing step in `backend/app/pipeline/shuttle.py` → `_store_data`, applied to the
dataframe **before** `artifacts.set_parquet("shuttle", df)`. This is the single shared insertion
point used by both the backend and colab pipelines.

## Algorithm — `_clean_trajectory(df)` (in order)
1. **Confidence gate** — mark frames with `confidence < settings.shuttle_clean_min_conf` (0.30) as
   missing (`x = y = NaN`). Removes the 11% low-conf points that account for 80% of teleports.
2. **Physical-velocity outlier reject** — walk the series; if a point's displacement from the last
   *valid* point exceeds `shuttle_max_jump_px` (200) **and** the next valid point jumps back toward
   the prior trajectory (a there-and-back spike, not a sustained move), drop it as a teleport.
   Only reject the *spike* shape — never a sustained fast move — so real smashes survive.
3. **Gap interpolation** — linearly interpolate `x,y` across missing runs **only up to
   `shuttle_max_interp_gap` frames (≤ 7)**; longer gaps stay `NaN` (genuinely untracked — do not
   fabricate trajectories).
4. **Light smoothing** — centered moving-median (window `shuttle_smooth_window` = 3) on `x,y` to
   de-jitter without lag. Do **not** smooth across `NaN` gaps.
5. **Provenance** — keep a `was_interpolated` boolean column so downstream consumers (hit detection,
   quality gate) can distinguish real vs filled points.

## New settings (`backend/app/config/settings.py`)
```python
shuttle_clean_enabled: bool = True
shuttle_clean_min_conf: float = 0.30
shuttle_max_jump_px: float = 200.0
shuttle_max_interp_gap: int = 7
shuttle_smooth_window: int = 3
```

## Validation (no GPU)
- Re-run: `teleport jumps > 200px` should drop from 281 → near-0; `frac conf > 0.5` rises; peaks in
  `debug_hit_scores` get cleaner.
- Unit test: synthetic track with 3 injected spikes + one 4-frame gap → assert spikes removed, gap
  filled, and a genuine 250px fast-shot move **preserved**.

## Risk / mitigation
Over-aggressive rejection eating real fast shots → mitigated by the there-and-back spike check and
the ≤ 7-frame interp cap. `shuttle_clean_enabled` allows clean A/B.

---

# Spec 2 — Rally segmentation + winner fix

## Problem
- **Mega-rallies:** rally 11 (69 shots / 78s) and rally 3 (44 shots / 55s) have **no internal gap
  > 90 frames** (max 76), so the `rally_gap_threshold` rule cannot split them. Splitting then falls
  back to `_is_rally_ending_shot`, which keys off the **unreliable stroke type**.
- **Winner skew (11/13 → player_2):** `_compute_rally_winner_after_attribution`
  (`rallies.py:43-49`) derives the winner from `last_pid` + a stroke-type-inferred `end_reason` —
  both noisy. The `shuttle_df` is not even passed at the `rallies.py:125` call site, so the
  smash-speed path is dead code.

## Part A — split mega-rallies using the shuttle, not the stroke type
A rally ends when the shuttle goes **dead** (hits floor / leaves play / stops moving) — visible in
the shuttle track independently of BST. In `backend/app/pipeline/rallies.py`:
- Between consecutive shots, scan the (cleaned) shuttle track for a **dead-shuttle window**:
  ≥ `rally_dead_frames` (≈ 25) consecutive frames where shuttle speed ≈ 0 (< `rally_dead_speed_px`)
  **or** confidence collapses (lost) **or** the shuttle is at floor level and static.
- Split the rally at that window **even when `frame_gap ≤ 90`** — catches the case where players
  resume quickly (gap < 3s) but the shuttle clearly died.
- Keep the existing 90-frame gap rule as the coarse signal; the dead-shuttle check is the fine one.

## Part B — winner from shuttle landing, not stroke inference
- Determine the losing side from **where the shuttle ended**: which half of the court (`court_y`
  relative to the net) the shuttle was in when it went dead, and whether it landed in/out. The side
  the shuttle died on = the side that failed to return → opponent wins.
- Use `last_pid` only as a tiebreaker, and **gate on attribution confidence**: if the last shot's
  `attribution_tier` is `rally_fallback` / `final_fallback`, don't trust it.
- **Pass `shuttle_df`** into `_compute_rally_winner_after_attribution` (the `rallies.py:125` call
  currently omits it).

## New settings
```python
rally_dead_frames: int = 25       # min still/lost-shuttle window to end a rally
rally_dead_speed_px: float = 4.0  # per-frame shuttle speed below this = "dead"
```

## Validation
- Mega-rallies (44, 69) split into plausible lengths; winner distribution de-skews from 11/2 toward
  balanced.
- Spot-check a few rally boundaries against shuttle-dead frames.
- Unit test: synthetic shot sequence with a dead-shuttle window mid-rally → asserts a split there.

## Note
De-couples rally structure and winners from BST quality — valuable regardless of how stroke
classification evolves.

---

# Spec 3 — Track-ID stitching (stabilize the 2 players)

## Problem
`player_detections.parquet` has **166 track-IDs for 2 players** (141 far, 75 near).
`players.py:125-150` groups by Ultralytics `track_id`, so each fragment becomes a separate "player"
entry; identity is held together only by the per-frame `center_y` vs court-midline side test.
Consequence: in `_build_clip`, `det_bbox_lookup` is keyed per fragment-id, so bbox-diagonal joint
normalization is computed from fragmented, short-lived tracks — adding noise to BST's most important
channel.

## Design
Insert a **stitching pass** in `_process_detections` that collapses the N fragments into **exactly 2
persistent identities** before assigning `player_1` / `player_2`.

## Algorithm
1. For each frame there are ≤ 2 detections, each with a side (near/far) from `center_y`.
2. Maintain two persistent tracks (`far`, `near`). For each new detection, assign to the persistent
   track of the **same side** whose last-known bbox center is within `track_stitch_max_dist_px`
   (centroid distance or IoU) — greedy nearest-neighbor association across the gap.
3. On a fragment break (new `track_id`), **inherit the persistent identity** from side + position
   continuity rather than starting a new player.
4. Output exactly 2 players, each with the full concatenated detection list → stable per-player bbox
   across the whole match.

## New settings
```python
track_stitch_enabled: bool = True
track_stitch_max_dist_px: float = 150.0  # max centroid jump to keep same identity
```

## Validation
- `players` artifact reports **2** ids with ~9000 detections each (vs 166 fragments).
- In `shots.parquet`, `bbox_diag_player_*_std` drops (more stable normalization).
- Confirm BST `clip_jnb_std` stops being bit-identical across clips if normalization was the cause.
- Unit test: synthetic detections with mid-stream `track_id` changes on a smoothly-moving player →
  asserts a single stitched identity.

## Risk / mitigation
The two players crossing sides (rare in singles — the net separates them) → handled by the side test
plus the distance gate; a hard "2 persistent tracks" cap prevents fragment explosion.

---

## Cross-cutting validation checklist
After each fix, re-run the phone clip and confirm BST internals do not regress:
1. Logit spread (top1 − top2 mean logit) stays flat (~0.2), not re-inflating toward 1.45.
2. Stroke variety preserved; no single class > ~35%.
3. Mean stroke confidence trends up (target > 0.317, the pre-min-window anchored baseline).
4. `hit_offset` within clips remains 0 (anchoring intact).
