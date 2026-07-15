# BaddyCoach — Conversation Summary (updated 2026-07-15)

## Objective
- Raise BST clip eligibility above 50.6% by feeding InpaintNet-repaired shuttle coords to BST and counting them as "present" in the quality gate (instead of zeroing + triple-penalizing). Verify stroke accuracy does not regress via Colab phone-video re-run.

## Important Details
- Main repo `/home/sujith/baddyCoach` on `master`, HEAD `681c6ab` (stitch_tracks hardening) + local WIP for P0/P1/P2 (uncommitted).
- Locked decisions: `bst_shuttle_norm`="resolution"; `bst_joint_norm`="bbox"; `hit_candidate_threshold`=0.50; court-rejected enters BST tensor in resolution mode only.
- Eligibility analysis (parquet, 172 shots, 87 eligible=50.6%): ineligible reasons long_shuttle_gap 62, too_many_repaired 48, low_observed 33, too_many_interpolated 11, long_bbox_gap 2, clip_too_short 1, low_pose_coverage 1, low_quality_score 79. Ineligible clips: observed median 0.45, repaired median 0.55, interpolated ~0, pose coverage median 1.0.
- BUG confirmed (root cause of exclusion): strokes.py zeroed InpaintNet-repaired shuttle from BST tensor when `bst_shuttle_require_raw_observation=True`; scorer counted repaired as missing → triple penalty.
- User decision: implement P0+P1+P2 together.

## Work State
### Completed this session (P0/P1/P2, uncommitted)
- P0: `bst_input_quality.py` — `present = observed | repaired`; `present_fraction` gates `low_observed_shuttle`; `max_shuttle_gap` measured on `present`; `too_many_repaired_shuttle` hard gate REMOVED. `strokes.py` `_build_clip` — feeds repaired coords to BST tensor when `bst_shuttle_use_repaired=True` (default), keeps interpolated zeroed unless `bst_shuttle_use_interpolated`.
- P1: `bst_repaired_shuttle_penalty=0.50` (mild) vs `bst_interpolated_shuttle_penalty=0.80` (heavier) — repaired (model output) penalized less than interpolated (fabric). `bst_max_interpolated_shuttle_fraction` raised 0.25→0.50.
- P2: `bst_contact_gap_window=15`; `max_shuttle_gap_frames` = longest absent run WITHIN contact window only (not full clip) — gap far from contact (pre-serve tail) no longer disqualifies.
- Settings added: `bst_shuttle_use_repaired=True`, `bst_shuttle_use_interpolated=False`, `bst_repaired_shuttle_penalty=0.50`, `bst_interpolated_shuttle_penalty=0.80`, `bst_contact_gap_window=15`. Removed: `bst_max_repaired_shuttle_fraction`. `bst_max_interpolated_shuttle_fraction` 0.25→0.50.
- Tests: `test_bst_input_quality.py` updated (removed too_many_repaired reason string; added present_shuttle_fraction asserts, contact-window gap tests, repaired-counts-as-present test). `test_strokes.py` updated `test_build_clip_skips_repaired_and_interpolated_when_require_raw` to assert repaired IS fed; added `test_build_clip_skips_repaired_when_use_repaired_false`.
- Verification: backend suite `not gpu and not model and not integration` → 4 failed (PRE-EXISTING: test_colab_pipeline×2, test_strokes×2 stale court-rejected tests), 515 passed (was 511 baseline; +4 new tests). No regressions. Integration test flaky (passes isolated).
- Eligibility reconstruction from parquet: old 87/172 (50.6%) → estimated ~100% for this phone sample (TrackNet dropouts were InpaintNet-repaired, so present≈1.0). Conservative floor from earlier rigorous counterfactual: +22 (gap relaxation) with P0 gap-filling adding more. Real number requires Colab re-run.

### Pre-existing (before this session)
- Merged `bst-input-quality`→`master`; deleted 4 branches + worktrees; preserved plan doc to `docs/superpowers/plans/2026-07-10-bst-input-quality-gate.md`.
- stitch_tracks hardening (scene-cut reset + court-side resolution) committed `681c6ab`, pushed.

### Blocked
- None.

## Next Move
1. Commit + push P0/P1/P2 (user to confirm).
2. Colab phone-video re-run to confirm eligibility rises AND stroke accuracy (frame error / exact match on manual labels) does not regress.

## Relevant Files
- `backend/app/pipeline/shared/bst_input_quality.py` — `evaluate_bst_clip_quality` (present-based scoring, contact-window gap).
- `backend/app/pipeline/strokes.py` — `_build_clip` L277-292 (feed repaired/interpolated per toggles).
- `backend/app/config/settings.py` — L184-188 (use_repaired/use_interpolated), L200-207 (penalties, gap window, interp gate).
- `backend/tests/test_bst_input_quality.py`, `backend/tests/test_strokes.py` — updated/added tests.
- `results/hybrid_results/debug/debug_bst_input_quality.parquet` — source analysis (172 shots).
- `docs/superpowers/plans/2026-07-10-bst-input-quality-gate.md` — design plan.
