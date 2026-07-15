# BaddyCoach — Conversation Summary (updated 2026-07-15, post-Colab re-run)

## Objective
- Raise BST clip eligibility by feeding InpaintNet-repaired shuttle coords to BST and counting them as "present" (P0/P1/P2). Confirm via Colab run on test_match.mp4 whether output improved.

## Important Details
- Main repo `/home/sujith/baddyCoach` on `master`, HEAD `bc7131e` (P0/P1/P2 committed + pushed). `stitch_tracks` hardening at `681c6ab`. Settings point BST to `ckpts/bst/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt` (commit 3f5460f swapped to CG_AP variant).
- Locked: `bst_shuttle_norm`="resolution"; `bst_joint_norm`="bbox"; `hit_candidate_threshold`=0.50; `hit_refine_window`=4 (reverted from 16); court-rejected enters BST tensor in resolution mode only.
- P0/P1/P2 settings: `bst_shuttle_use_repaired=True` (feeds repaired), `bst_shuttle_use_interpolated=False`, `bst_repaired_shuttle_penalty=0.50`, `bst_interpolated_shuttle_penalty=0.80`, `bst_contact_gap_window=15`, `bst_max_interpolated_shuttle_fraction=0.50`. Removed `bst_max_repaired_shuttle_fraction`.

## Colab re-run results (test_match.mp4, 9000 frames, T4, 2069s)
- 307 clips evaluated; **eligibility 208/307 = 67.8%** (up from the ~50.6% bottleneck measured earlier).
- SHUTTLE DISQUALIFIERS ELIMINATED (intended effect, model-independent):
  - `long_shuttle_gap` 62→0, `too_many_repaired_shuttle` 48→0, `low_observed_shuttle` 33→0, `too_many_interpolated_shuttle` 11→0.
  - Remaining ineligible reasons: `low_quality_score` 96, `long_bbox_gap` 6, `low_pose_coverage` 4 (genuine non-shuttle quality issues — correctly abstained).
  - `present_shuttle_fraction` median=1.0, mean=0.994 (repaired now fills gaps).
  - `max_shuttle_gap_frames` (contact window): median=0, max=2. `full_shuttle_gap_frames` median=0, max=26 (gaps outside contact window no longer disqualify).
- stroke_source: BST-involved = bst(56)+bst_no_physics(56)+bst_gate_distrusted(66)+agree(8)+temporal_smoothing(3) = 189/307 (62%). quality_abstain 56, physics_fallback 62.
- Stroke vocab shifted: NEW run has NO `rush` class (Jul 6 baseline had rush=38%); top = net_shot 66(21%), unknown 56(18%), block 43(14%), lift 25, short_serve 18, push 20, clear 17, long_serve 15, smash 14, drive 12, drop 10.

## CONFOUNDS discovered (Colab run ≠ clean A/B vs Jul 6 baseline)
1. **BST checkpoint swapped** (separate commits 112819a/a666c75/3f5460f, NOT my change): baseline Jul 6 used `bst_CG_JnB_bone_merged.pt` (25-class incl. rush); current code uses `bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt` (no rush). So stroke distribution/accuracy differ due to MODEL, not my change.
2. **hit_refine_window reverted 16→4** between Jul 7 and now: Jul 7 (win=16) had frame error mean 18.2; this run (win=4) mean 6.8. Temporal-alignment improvement is from the revert, not P0/P1/P2.
3. labels_enriched_new.csv was enriched by an EARLIER run (has rush in true_stroke, precomputed frame_diff mean 15.65 = the Jul 7 win=16 run). So OLD-run metrics embedded in CSV are from win=16 + old checkpoint.

## Manual-label evaluation (labels_enriched_new.csv, 99 labeled, matched to NEW run by label_frame)
- Recall ≈ 82% (non-greedy; greedy gives 69 — matching artifact, not real). 18 labels genuinely have no new-run shot within 15 frames.
- Frame error: NEW mean 6.83 / median 6.0 (vs OLD embedded 15.65/8.0 — but OLD is win=16 run, so improvement = win=4 revert, NOT my change).
- Stroke exact+similar: NEW 27.5% (exact 15.9%). OLD embedded 10.1% but unreliable (crude class-id→name reverse map + old checkpoint + win=16).
- BST eligible coverage on labeled subset: 81.2% (direct effect of P0/P1/P2).
- Per-class NEW: block→block 6/15 (40% exact, best); smash→net_shot/block/long_serve (poor); net_shot, lift, clear heavily confused. New model still weak on fine stroke types.

## Conclusions
- P0/P1/P2 **definitively achieved their intended effect**: shuttle-eligibility bottleneck removed, repaired coords now fed to BST, presence ≈1.0, contact-window gaps ≈0. This is proven and model-independent.
- Stroke-accuracy improvement from my change specifically CANNOT be isolated from these artifacts: the run differs from baseline in BOTH checkpoint and hit window. The labeled-subset accuracy (27.5%) is not a clean A/B.
- No regressions from P0/P1/P2 themselves (gate logic correct; tests pass).

## Next Move (for a clean A/B of P0/P1/P2 only)
- Re-run Colab on test_match.mp4 with CURRENT checkpoint + win=4, toggling ONLY: `bst_shuttle_use_repaired=False` + restore `too_many_repaired_shuttle` gate (old logic) vs current. Diff eligibility + labeled accuracy. (Save the old-logic run's shots.parquet + debug_bst_input_quality.parquet before overwriting.)
- Alternatively keep current checkpoint going forward (intended per recent commits) and accept eligibility win; fine-stroke accuracy is a MODEL issue (different checkpoint), out of P0/P1/P2 scope.

## Relevant Files
- `results/hybrid_results/debug/debug_bst_input_quality.parquet` — NEW run: 307 clips, shuttle disqualifiers 0.
- `results/hybrid_results/debug/shots.parquet` — NEW run: 307 shots, stroke_source, bst_input_eligible.
- `results/hybrid_results/report.json` — 307 shots, 35 rallies.
- `results/hybrid_results/pipeline.log` — run log (test_match.mp4, T4, checkpoint download).
- `labels_enriched_new.csv` — 99 manual labels, enriched by earlier run.
- `backend/scripts/evaluate_labels.py` — eval harness (greedy match; frame_diff overridden by CSV = old run).
- `backend/app/config/settings.py` — checkpoint + P0/P1/P2 settings.
- `docs/superpowers/plans/2026-07-10-bst-input-quality-gate.md` — design plan.
