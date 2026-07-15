# Task 2.3 — Verify near/far attribution convention vs gold labels

**Status:** DONE (convention already correct — no behavior change needed)
**Branch:** accuracy-improvement

## What was checked
1. The `side` write path from `OwnerDecision.side` → `shots_df["side"]`.
2. Whether any place in the pipeline inverts the near/far mapping.
3. The committed-only attribution match rate against the gold labels (`labels_enriched_new.csv` `side` column) using the SAME matching logic as the benchmark harness.

## Trace result (side write path)
- `ownership_quality.assign_rally_owners` returns `OwnerDecision(side="near"|"far", ...)`
  — `backend/app/pipeline/shared/ownership_quality.py:80` (Viterbi path) and `:101`/`:121` (anchor/bridge paths). No `near`/`far` flip.
- `attribution.py:106` writes `shots_df.at[idx, "side"] = decision.side` **verbatim** — no inversion.
- Viterbi (`ownership_scorer.assign_hit_owners_viterbi`, `ownership_scorer.py:417`) emits the literal state strings `"near"`/`"far"` and is consumed directly.
- `strokes.py` local `side` variables (`:147`, `:335`, `:591`) are only for BST clip p0/p1 player ordering (Far=p0, Near=p1), NOT for writing `shots_df["side"]`. No inversion there.
- `rallies.py` does not write `shots_df["side"]`.
- Canonical convention source: `players.py::_resolve_sides` (`:181`) assigns `player_1`→"near" (larger median court-y = camera-near/lower-court) and `player_2`→"far". This matches the human-label meaning of near/far.

**No inversion found.** The convention already matches the gold labels.

## Test results
- Added `tests/test_attribution.py::test_near_far_convention_vs_labels`
  (marked `@pytest.mark.integration` and `@pytest.mark.benchmark`). Skips cleanly
  when `results/hybrid_results/debug/shots.parquet` or `labels_enriched_new.csv`
  is absent. When present, it reads both, builds gold labels from
  `label_status=="labeled"` rows, runs greedy nearest-frame match (radius 15),
  and asserts the **committed-only** match rate (`side in {near,far}` AND
  `owner_uncertain==False`, `side == label.side`) is **> 50%**.
- With real files present (`-m integration`): **PASSED**.
  Measured committed-only rate = **76.9% (10/13)**, consistent with the
  benchmark harness's ~62.5% (15/24) overall committed rate. An inverted
  convention would yield ~50% (random), so this guards against a future sign-flip.
- Regression run `tests/test_attribution.py -q`: **11 passed**.
  2 failures (`test_unanchored_rally_stays_unknown`,
  `test_short_compatible_gap_bridges_between_anchors`) are **pre-existing**
  (confirmed via `git stash`): they assert the old `owner_source` values
  `local_anchor`/`viterbi_bridge`/`unknown`, but Task 2.1/2.2 changed the
  owner source to `viterbi_rally`. They are unrelated to this task and were
  already failing before my edits (2 failed, 10 passed on stashed tree).

## Files changed
- `backend/app/pipeline/attribution.py` — added CANONICAL NEAR/FAR CONVENTION
  doc block (top of module) + inline comment at the `side` write (no logic change).
- `backend/tests/test_attribution.py` — added `test_near_far_convention_vs_labels`
  guard test (integration/benchmark markers, clean skip without real files).
- `backend/pytest.ini` — registered `benchmark` marker (silences warning).

## Concerns
- None blocking. The 2 pre-existing `test_attribution.py` failures stem from the
  2.1/2.2 owner-source rename (`viterbi_rally` vs `local_anchor`/`viterbi_bridge`)
  and should be updated as part of / after those tasks, not here.
- `aimplayer_alpha` remains near-random (mean ≈ 0.499) so the `attention_owner_match`
  diagnostic disagrees with Viterbi on ~46% of shots — this is a model-signal issue
  (documented in AGENTS.md), not a convention inversion, and does not affect the
  `side` column written by this stage.
