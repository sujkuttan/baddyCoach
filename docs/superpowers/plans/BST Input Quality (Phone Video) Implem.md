# BST Input Quality (Phone Video) Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.>> **Do not start implementation until the user explicitly asks to execute.** This document is the implementation plan only.
**Goal:** Make phone-video shuttle, court geometry, hit timing, and joint features produce BST-admissible clips so stroke classification stops abstaining on ~68% of shots.
**Architecture:** Keep TrackNet → `shuttle_raw` / cleaned `shuttle` → hits → `_build_clip` → `evaluate_bst_clip_quality` → BST. Fix the admission path in order: (1) stop blanking valid pixel shuttle for court OOB when using resolution norm, (2) make court geometry video-specific and reliability-gated, (3) drive hits from `shuttle_raw` and sanity-check contact alignment, (4) unify Colab TrackNet with backend and reduce fake fill, (5) harden joint normalization, (6) retune the quality gate only after upstream metrics improve. Backend shared modules are the source of truth; Colab inherits via imports except where it still inlines TrackNet/court ingest.
**Tech Stack:** Python 3, NumPy, pandas, PyTorch TrackNetV3, pydantic-settings, pytest, Colab `pipeline.py`.
**Baseline (latest Colab run in `../logs/` on `test_match.mp4`):**
| Metric | Baseline ||--------|----------|| Shots | 301 || `quality_abstain` | 206 (68.4%) || `bst_input_eligible` | 32 (10.6%) || BST inference (`bst_raw_confidence` set) | 13 (4.3%) || `court_rejected_shuttle` hard-flag | 207 shots || Mean `court_rejected_shuttle_fraction` | 0.46 || Mean `clip_shuttle_valid` | ~49.8 || Shuttle conf ≥ 0.5 | 29.6% || InpaintNet repaired frames | 35.5% || `y_frac` extreme warnings | 33 || Physics overrides | 0 (unchanged target) |
**Success criteria (same video after all phases):**
| Metric | Target ||--------|--------|| `court_rejected_shuttle` hard-fail shots | &lt; 40 || `bst_input_eligible` | ≥ 50% || BST inference rate | ≥ 40% || Mean `clip_shuttle_valid` | ≥ 80 || `y_frac` extreme warnings | &lt; 10 || Post-norm joint \|mean\| | &lt; 0.5 on majority of clips || Physics override guard behavior | unchanged |
**Locked design decisions (no open options during implementation):**
1. Default `bst_shuttle_norm` stays `"resolution"`. Court-rejected points **still enter** the BST shuttle tensor in resolution mode; they are blanked only when `bst_shuttle_norm == "court"`.2. `court_rejected_shuttle` is a **hard** quality reason only when `bst_shuttle_norm == "court"`; otherwise soft score penalty only (`−0.20` if any observed rejected).3. Hit Phase-1 trajectory detection prefers `shuttle_raw` when present; cleaned `shuttle` is fallback only.4. Colab deletes its inline `TrackNetV3` and uses `app.models.tracknet.TrackNetV3` (same as `get_tracknet()`).5. `default_corners.json` is used only with explicit `--use-default-corners`; otherwise prefer CLI / `manual_corners.json` / auto-detect / proportional fallback.6. Shuttle-in-court reliability threshold is `0.60` of high-conf detections inside court±margin; below that set `court["geometry_reliable"]=False` and skip court rejection.7. Do **not** widen `hit_refine_window` beyond 4. Do **not** disable the quality gate to force BST.8. Phase 6 retunes thresholds only after a Colab re-run shows Phase 1–5 metrics.
---
## File map
| File | Responsibility ||------|----------------|| `backend/app/pipeline/strokes.py` | `_build_clip` shuttle fill rules; joint masking || `backend/app/pipeline/shared/bst_input_quality.py` | Mode-aware court-reject hard fail; optional joint-mean soft penalty || `backend/app/pipeline/shuttle.py` | Court enrichment; reliability gate; shuttle-in-court fraction helper || `backend/app/pipeline/shared/court.py` | Existing `court_geometry_reliable`; maybe export helper used by shuttle || `backend/app/pipeline/hits.py` | Prefer `shuttle_raw`; contact `y_frac` sanity nudge || `backend/app/pipeline/shared/bst_preproc.py` | Joint norm (bbox diagonal + center_align) || `backend/app/config/settings.py` | New/retuned thresholds || `colab/pipeline.py` | Court fallback order; backend TrackNet; debug parquet dumps || `backend/tests/test_strokes.py` | Clip shuttle fill / provenance || `backend/tests/test_bst_input_quality.py` | Gate reasons by norm mode || `backend/tests/test_shuttle.py` | Court enrichment + reliability || `backend/tests/test_hits.py` | Raw-shuttle preference + y_frac nudge || `docs/superpowers/specs/2026-07-10-bst-input-quality-design.md` | Amend court-reject tensor semantics for resolution mode |
---
## Phase 1 — Uncouple court rejection from resolution-mode BST shuttle
### Task 1: Resolution-mode `_build_clip` keeps court-rejected pixel shuttle
**Files:**- Modify: `backend/app/pipeline/strokes.py` (shuttle fill loop ~L254–283)- Modify: `backend/tests/test_strokes.py` (`test_build_clip_zeros_court_rejected_shuttle_and_records_provenance` and new court-mode test)- Amend note in: `docs/superpowers/specs/2026-07-10-bst-input-quality-design.md` (Clip provenance / Inference routing bullets that say court-rejected always encode as `[0,0]`)
- [ ] **Step 1: Rewrite the existing test for resolution mode and add a court-mode twin**
Replace `test_build_clip_zeros_court_rejected_shuttle_and_records_provenance` with two tests:
```pythondef test_build_clip_resolution_mode_keeps_court_rejected_pixel_shuttle(monkeypatch):    from app.pipeline.strokes import _build_clip    from app.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "resolution")
    frames = [0, 1, 2]    shuttle = pd.DataFrame({        "frame": frames,        "x": [100.0, 200.0, 300.0],        "y": [100.0, 200.0, 300.0],        "confidence": [0.9, 0.9, 0.9],        "was_interpolated": [False, True, False],        "court_rejected": [False, True, False],    })    shuttle_raw = pd.DataFrame({        "frame": frames,        "x": [100.0, np.nan, 300.0],        "y": [100.0, np.nan, 300.0],        "confidence": [0.9, 0.0, 0.9],        "was_repaired": [False, True, False],    })    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])    pose = pd.DataFrame([        {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}        for frame in frames for player in ("player_1", "player_2")    ])    players = [        {"id": "player_1", "side": "near", "detections": [            {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames        ]},        {"id": "player_2", "side": "far", "detections": [            {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames        ]},    ]
    clip = _build_clip(        frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,        player_detections=players, player_ids=["player_1", "player_2"],        shuttle_raw=shuttle_raw,    )
    # Court-rejected frame still contributes resolution-normalized pixels    np.testing.assert_allclose(clip["shuttle"][1], [200.0 / 640.0, 200.0 / 480.0], atol=1e-6)    assert clip["_bst_provenance"]["shuttle_court_rejected"] == [False, True, False]    assert clip["_bst_provenance"]["shuttle_observed"] == [True, False, True]

def test_build_clip_court_mode_zeros_court_rejected_shuttle(monkeypatch):    from app.pipeline.strokes import _build_clip    from app.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "court")
    frames = [0, 1, 2]    shuttle = pd.DataFrame({        "frame": frames,        "x": [100.0, 200.0, 300.0],        "y": [100.0, 200.0, 300.0],        "confidence": [0.9, 0.9, 0.9],        "was_interpolated": [False, False, False],        "court_rejected": [False, True, False],    })    # Minimal pose/players same as above...    keypoints = np.column_stack([np.full(17, 50.0), np.full(17, 50.0), np.ones(17)])    pose = pd.DataFrame([        {"frame": frame, "player_id": player, "keypoints": keypoints.tolist()}        for frame in frames for player in ("player_1", "player_2")    ])    players = [        {"id": "player_1", "side": "near", "detections": [            {"frame": frame, "bbox": [0, 0, 100, 100]} for frame in frames        ]},        {"id": "player_2", "side": "far", "detections": [            {"frame": frame, "bbox": [200, 0, 300, 100]} for frame in frames        ]},    ]
    clip = _build_clip(        frames, shuttle, pose, 640, 480, 13.4, 6.1, 3,        player_detections=players, player_ids=["player_1", "player_2"],        homography=np.eye(3),    )
    np.testing.assert_array_equal(clip["shuttle"][1], [0.0, 0.0])    assert clip["_bst_provenance"]["shuttle_court_rejected"][1] is True```
- [ ] **Step 2: Run tests — expect resolution-mode test FAIL**
```powershell$env:PYTHONPATH = "$PWD\backend"$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"python -m pytest backend/tests/test_strokes.py::test_build_clip_resolution_mode_keeps_court_rejected_pixel_shuttle -v```
Expected: FAIL because `_build_clip` still skips `court_rejected` rows unconditionally (`assert_allclose` sees `[0,0]`).
- [ ] **Step 3: Implement mode-aware shuttle fill**
In `backend/app/pipeline/strokes.py`, replace the fill condition:
```python        # Provenance always records court_rejected.        # Resolution/pixel BST inputs keep image-space coords even when        # court projection is OOB (homography mismatch must not blank        # the only channel BST was trained to see in resolution mode).        # Court-normalized mode still blanks rejected points.        use_for_tensor = clean_row is not None and (            not court_rejected or settings.bst_shuttle_norm != "court"        )        if use_for_tensor:            s_conf = float(clean_row.get("confidence", 1.0))            if s_conf >= settings.shuttle_min_conf:                sx = float(clean_row["x"])                sy = float(clean_row["y"])                if not np.isfinite(sx) or not np.isfinite(sy):                    pass                elif settings.bst_shuttle_norm == "court" and homography is not None:                    sx, sy = image_to_court(homography, (sx, sy))                    shuttle[t, 0] = max(0.0, min(1.0, sx / court_length if court_length > 0 else 0))                    shuttle[t, 1] = max(0.0, min(1.0, sy / court_width if court_width > 0 else 0))                else:                    shuttle[t, 0] = max(0.0, min(1.0, sx / vid_w if vid_w > 0 else 0))                    shuttle[t, 1] = max(0.0, min(1.0, sy / vid_h if vid_h > 0 else 0))```
Also update any later `clip_shuttle_valid` counting that skips `court_rejected` in resolution mode (search for `court_rejected` in the same file ~L368) so valid-count includes resolution-mode rejected pixels.
- [ ] **Step 4: Re-run both clip tests**
```powershellpython -m pytest backend/tests/test_strokes.py::test_build_clip_resolution_mode_keeps_court_rejected_pixel_shuttle backend/tests/test_strokes.py::test_build_clip_court_mode_zeros_court_rejected_shuttle -v```
Expected: PASS
- [ ] **Step 5: Commit**
```bashgit add backend/app/pipeline/strokes.py backend/tests/test_strokes.py docs/superpowers/specs/2026-07-10-bst-input-quality-design.mdgit commit -m "$(cat <<'EOF'fix: keep resolution-mode BST shuttle when court-rejected
Court OOB is a homography signal, not missing pixels. Blanking thoseframes starved BST on phone footage with mismatched default corners.EOF)"```
### Task 2: Mode-aware quality hard-fail for court rejection
**Files:**- Modify: `backend/app/pipeline/shared/bst_input_quality.py`- Modify: `backend/tests/test_bst_input_quality.py`
- [ ] **Step 1: Update / add tests**
```pythondef test_quality_hard_rejects_court_rejected_only_in_court_norm_mode(monkeypatch):    from app.config import settings as settings_mod    from app.pipeline.shared.bst_input_quality import evaluate_bst_clip_quality
    rejected = [True] * 6 + [False] * 14  # 0.30 > 0.25
    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "resolution")    result_res = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))    assert result_res["eligible"] is True or "court_rejected_shuttle" not in result_res["reasons"]    assert result_res["score"] == pytest.approx(0.8)  # soft −0.20 only    assert "court_rejected_shuttle" not in result_res["reasons"]
    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "court")    result_court = evaluate_bst_clip_quality(_provenance(shuttle_court_rejected=rejected))    assert result_court["eligible"] is False    assert "court_rejected_shuttle" in result_court["reasons"]```
Update existing `test_quality_rejects_clip_with_too_many_court_rejected_points` to monkeypatch `bst_shuttle_norm="court"` so it still documents court-mode hard fail.
Update `test_quality_accumulates_all_failed_hard_checks_and_clamps_score` similarly (force court norm for the court reason, or drop that reason from the expected list under resolution default).
- [ ] **Step 2: Run — expect FAIL under default resolution**
```powershellpython -m pytest backend/tests/test_bst_input_quality.py -v```
Expected: existing hard-reject test fails once rewritten for mode awareness, or new test fails until code changes.
- [ ] **Step 3: Implement**
In `evaluate_bst_clip_quality`:
```python    if observed_rejected.any():        score -= 0.20    if (        settings.bst_shuttle_norm == "court"        and rejected_fraction > settings.bst_max_court_rejected_shuttle_fraction    ):        hard_reasons.append("court_rejected_shuttle")```
Remove the previous unconditional hard-reason branch for court rejection.
- [ ] **Step 4: Re-run quality tests — PASS**
- [ ] **Step 5: Commit**
```bashgit add backend/app/pipeline/shared/bst_input_quality.py backend/tests/test_bst_input_quality.pygit commit -m "$(cat <<'EOF'fix: hard-fail court-rejected shuttle only in court-norm BST modeEOF)"```
### Task 3: Persist enriched shuttle (+ court columns) and shuttle_raw in Colab debug
**Files:**- Modify: `colab/pipeline.py` (~L1250–1260)
- [ ] **Step 1: Change debug dumps to post-enrichment artifacts**
After `store.set_parquet("shuttle", shuttle_df)` and `shuttle_raw` is set:
```python        # Dump the same artifacts the stage uses — not pre-clean all_shuttle.        store.get_parquet("shuttle").to_parquet(debug_dir / "shuttle.parquet", index=False)        raw_dbg = store.get_parquet("shuttle_raw")        if raw_dbg is not None:            raw_dbg.to_parquet(debug_dir / "shuttle_raw.parquet", index=False)```
Remove (or stop using) `pd.DataFrame(all_shuttle).to_parquet(debug_dir / "shuttle.parquet", ...)`.
- [ ] **Step 2: Confirm columns exist after enrichment path**
When court is valid, enriched `shuttle_df` must include `court_rejected`, `x_court`, `y_court` before dump. No unit test in CI for Colab file I/O; verify with a short local snippet importing the enrichment helper:
```powershellpython -c "from app.pipeline.shuttle import _add_court_space_columns; import inspect; print('court_rejected' in inspect.getsource(_add_court_space_columns))"```
Expected: `True`
- [ ] **Step 3: Commit**
```bashgit add colab/pipeline.pygit commit -m "$(cat <<'EOF'fix: dump enriched shuttle and shuttle_raw in Colab debug artifactsEOF)"```
**Phase 1 exit check:** Offline, recompute eligibility on a synthetic provenance vector with 46% court_rejected under resolution mode → eligible if other gates pass. Document in commit message / PR notes.
---
## Phase 2 — Court geometry that matches the phone video
### Task 4: Add shuttle-in-court fraction helper + settings
**Files:**- Modify: `backend/app/config/settings.py`- Modify: `backend/app/pipeline/shuttle.py` (or `shared/court.py`)- Modify: `backend/tests/test_shuttle.py`
- [ ] **Step 1: Add settings**
```python    court_shuttle_in_bounds_min_fraction: float = 0.60    court_shuttle_reliability_min_conf: float = 0.50```
- [ ] **Step 2: Write failing test**
```pythondef test_shuttle_in_court_fraction_and_reliability_flag():    from app.pipeline.shuttle import compute_shuttle_in_court_fraction
    # Identity H: pixel == court metres. Points at x=20 are OOB for length 13.4.    df = pd.DataFrame({        "x": [1.0, 2.0, 20.0, 3.0],        "y": [1.0, 1.0, 1.0, 1.0],        "confidence": [0.9, 0.9, 0.9, 0.2],  # last ignored by conf gate    })    frac = compute_shuttle_in_court_fraction(df, np.eye(3), min_conf=0.5, oob_margin=1.0)    # 2 of 3 high-conf points in bounds    assert frac == pytest.approx(2 / 3, abs=1e-6)```
- [ ] **Step 3: Implement helper**
```pythondef compute_shuttle_in_court_fraction(    df: pd.DataFrame,    H: np.ndarray,    *,    min_conf: float,    oob_margin: float,) -> float:    from app.pipeline.shared.court import image_to_court, COURT_LENGTH, COURT_WIDTH
    considered = 0    inside = 0    for _, row in df.iterrows():        if float(row.get("confidence", 0.0)) < min_conf:            continue        x, y = row.get("x"), row.get("y")        if not (np.isfinite(x) and np.isfinite(y)):            continue        considered += 1        cx, cy = image_to_court(H, (float(x), float(y)))        oob = (            not np.isfinite(cx) or not np.isfinite(cy)            or cx < -oob_margin or cx > COURT_LENGTH + oob_margin            or cy < -oob_margin or cy > COURT_WIDTH + oob_margin        )        if not oob:            inside += 1    return float(inside / considered) if considered else 0.0```
- [ ] **Step 4: Tests PASS + commit**
```bashgit commit -m "$(cat <<'EOF'feat: measure shuttle-in-court fraction for homography reliabilityEOF)"```
### Task 5: Gate court rejection on geometry reliability
**Files:**- Modify: `backend/app/pipeline/shuttle.py` (`ShuttleTrackingStage.run` and/or `_add_court_space_columns` callers)- Modify: `colab/pipeline.py` (post-homography block ~L1246–1248)- Test: `backend/tests/test_shuttle.py`
- [ ] **Step 1: Failing behavior test**
```pythondef test_add_court_space_skips_rejection_when_geometry_unreliable(monkeypatch):    from app.pipeline.shuttle import _add_court_space_columns
    monkeypatch.setattr("app.pipeline.shuttle.settings.shuttle_oob_margin_meters", 0.25)    df = pd.DataFrame({        "frame": [0, 1],        "x": [20.0, 21.0],  # OOB under identity H        "y": [1.0, 1.0],        "confidence": [0.9, 0.9],    })    enriched = _add_court_space_columns(        df, np.eye(3), fps=30.0, geometry_reliable=False    )    assert enriched["court_rejected"].tolist() == [False, False]    assert enriched["x"].tolist() == [20.0, 21.0]```
- [ ] **Step 2: Implement `geometry_reliable` kwarg**
```pythondef _add_court_space_columns(    df: pd.DataFrame, H: np.ndarray, fps: float, *, geometry_reliable: bool = True) -> pd.DataFrame:    ...            if geometry_reliable and out_of_bounds:                court_rejected[i] = True                ...                continue            # When unreliable: still write x_court/y_court for diagnostics,            # but never set court_rejected / never NaN-out for speed veto.```
Call site in backend stage + Colab:
```python    frac = compute_shuttle_in_court_fraction(        shuttle_df, H,        min_conf=settings.court_shuttle_reliability_min_conf,        oob_margin=settings.shuttle_oob_margin_meters,    )    geom_ok = bool(court.get("valid")) and court_geometry_reliable(court.get("corners_pixel") or court.get("corners"))    geom_ok = geom_ok and frac >= settings.court_shuttle_in_bounds_min_fraction    court["geometry_reliable"] = geom_ok    court["shuttle_in_court_fraction"] = frac    shuttle_df = _add_court_space_columns(        shuttle_df, np.array(court["homography"]), float(video_fps),        geometry_reliable=geom_ok,    )    logger.info("Court shuttle reliability", fraction=frac, reliable=geom_ok)```
Use the correct corner key already present on the court dict in each path (`corners` / `corners_pixel`).
- [ ] **Step 3: Tests PASS + commit**
```bashgit commit -m "$(cat <<'EOF'fix: skip court shuttle rejection when homography is unreliableEOF)"```
### Task 6: Demote silent `default_corners.json` in Colab
**Files:**- Modify: `colab/pipeline.py` court fallback (~L1001–1033) and argparse- Modify: notebook CLI docs if present in the same file’s `argparse` block
- [ ] **Step 1: Change fallback order**
New order:1. CLI `--court-corners` / `court_corners` arg2. `{output_dir}/manual_corners.json`3. Auto-detect (`court_kpRCNN` / color+line)4. Proportional rectangle fallback (already exists)5. **Only if** `--use-default-corners`: load `backend/app/config/default_corners.json`
```python    parser.add_argument(        "--use-default-corners",        action="store_true",        help="Allow repo default_corners.json (disabled by default; often wrong for phone framing)",    )```
When default corners would have been used previously, print a clear warning that geometry may be unreliable and recommend CourtCornerSetup / `manual_corners.json`.
- [ ] **Step 2: Smoke-check argparse**
```powershellpython colab/pipeline.py --help```
Expected: help text includes `--use-default-corners`.
- [ ] **Step 3: Commit**
```bashgit commit -m "$(cat <<'EOF'fix: stop silently applying default_corners.json on Colab phone videosEOF)"```
**Phase 2 exit check:** With reliable corners for `test_match.mp4`, `shuttle_in_court_fraction` ≥ 0.60 and mean `court_rejected_shuttle_fraction` ≪ 0.46 on next run.
---
## Phase 3 — Hit timing for contact-centered clips
### Task 7: Prefer `shuttle_raw` for trajectory hit detection
**Files:**- Modify: `backend/app/pipeline/hits.py` (`HitFrameLocalizationStage.run` ~L695–702)- Modify: `backend/tests/test_hits.py`
- [ ] **Step 1: Failing test — cleaned shuttle must not win when raw exists**
```pythondef test_hit_stage_prefers_shuttle_raw_over_cleaned(tmp_job_dir, monkeypatch):    """Cleaned trajectory is flat (no reversal); raw has a clear reversal at frame 20."""    monkeypatch.setattr("app.pipeline.hits.settings.audio_hit_enabled", False)    monkeypatch.setattr("app.pipeline.hits.settings.wrist_hit_enabled", False)    monkeypatch.setattr("app.pipeline.hits.settings.hit_refine_window", 0)    monkeypatch.setattr("app.pipeline.hits.settings.hit_frame_calibration_offset", 0)
    store = ArtifactStore(tmp_job_dir)    n = 40    frames = list(range(n))    # Raw: V-shaped reversal at 20    x_raw = [100.0 + t * 5.0 for t in range(20)] + [200.0 - (t - 20) * 8.0 for t in range(20, n)]    y_raw = [200.0 - t * 2.0 for t in range(20)] + [160.0 + (t - 20) * 4.0 for t in range(20, n)]    store.set_parquet("shuttle_raw", pd.DataFrame({        "frame": frames, "x": x_raw, "y": y_raw, "confidence": [0.95] * n,    }))    # Cleaned: nearly constant — would hide the hit if preferred    store.set_parquet("shuttle", pd.DataFrame({        "frame": frames, "x": [150.0] * n, "y": [180.0] * n, "confidence": [0.95] * n,    }))    store.set_parquet("pose", pd.DataFrame({        "frame": [], "player_id": [], "keypoints": [],    }))
    result = HitFrameLocalizationStage().run(store, StageConfig())    assert result.status == "success"    hits = store.get_parquet("hits")    assert hits is not None and len(hits) > 0    assert any(abs(int(f) - 20) <= 5 for f in hits["frame"].tolist())```
- [ ] **Step 2: Run — FAIL while stage prefers cleaned flat shuttle**
- [ ] **Step 3: Implement preference**
```python        shuttle_raw_df = artifacts.get_parquet("shuttle_raw")        shuttle_clean_df = artifacts.get_parquet("shuttle")        if shuttle_raw_df is not None and len(shuttle_raw_df) > 0:            shuttle_df = shuttle_raw_df.copy()            # Keep NaNs — do not ffill for hit detection (preserves gaps/reversals)        elif shuttle_clean_df is not None and len(shuttle_clean_df) > 0:            shuttle_df = shuttle_clean_df            logger.warning("Hit detection falling back to cleaned shuttle; shuttle_raw missing")        else:            return StageResult.from_error("Shuttle tracking data required")```
Update `input_keys` docstring/comment; keep `"shuttle"` in inputs for fallback compatibility or list both.
Use **cleaned** shuttle only for optional post-hit analytics if needed later — not for Phase-1 detect.
- [ ] **Step 4: Tests PASS + commit**
```bashgit commit -m "$(cat <<'EOF'fix: detect hit frames from shuttle_raw to preserve reversalsEOF)"```
### Task 8: Contact `y_frac` sanity nudge after refine+calib
**Files:**- Modify: `backend/app/pipeline/hits.py`- Modify: `backend/app/config/settings.py`- Modify: `backend/tests/test_hits.py`
- [ ] **Step 1: Add settings**
```python    hit_contact_yfrac_min: float = 0.15    hit_contact_yfrac_max: float = 0.85    hit_contact_sanity_enabled: bool = True```
- [ ] **Step 2: Helper + test**
Reuse `_direction_reversal_angle` already in `hits.py`. New helper:
```pythondef _contact_y_frac(shuttle_df: pd.DataFrame, frame: int, window: int = 15) -> float | None:    """Return frame-0 y position as fraction of local trajectory y-range, or None."""    ...```
```pythondef test_contact_sanity_nudges_extreme_yfrac(monkeypatch):    # Build a shuttle arc where frame 10 is endpoint (extreme) but frame 14    # has stronger reversal; with window=4 sanity should move toward 14.    ...```
- [ ] **Step 3: Wire after Phase 3 calib (~L779–786)**
```python        if getattr(settings, "hit_contact_sanity_enabled", True) and refine_window > 0:            nudged = 0            for c in candidates:                yf = _contact_y_frac(shuttle_df, c["frame"])                if yf is None:                    continue                if settings.hit_contact_yfrac_min <= yf <= settings.hit_contact_yfrac_max:                    continue                better = _find_nearest_wrist_frame(  # already combines reversal+wrist                    c["frame"], pose_df, shuttle_df,                    search_window=refine_window,                    min_shuttle_conf=settings.shuttle_min_conf,                )                if better != c["frame"]:                    c["_sanity_offset"] = better - c["frame"]                    c["frame"] = better                    nudged += 1            if nudged:                candidates = non_max_suppression(candidates, detector.min_gap_frames)                logger.info("Contact y_frac sanity nudges", count=nudged)```
Do **not** change `hit_refine_window` default (stays 4). Do **not** change `hit_frame_calibration_offset` in this task — only re-measure after Task 7+8 against labels (Task 9).
- [ ] **Step 4: Commit**
```bashgit commit -m "$(cat <<'EOF'feat: nudge hit frames when contact y_frac is at trajectory extremeEOF)"```
### Task 9: Re-evaluate calibration offset against labels (analysis only, then optional setting change)
**Files:**- Possibly modify: `backend/app/config/settings.py` (`hit_frame_calibration_offset`)- Script (create if missing): `scripts/eval_hit_frame_offset.py`
- [ ] **Step 1: Write a small offline evaluator**
```python# scripts/eval_hit_frame_offset.py# Load labels_enriched.csv + hits.parquet (or re-run detector on shuttle_raw)# Report median(pred - label) for offsets in {0,4,6,8,10}```
- [ ] **Step 2: Run on available labels**
```powershellpython scripts/eval_hit_frame_offset.py --labels labels_enriched.csv --hits ../logs/debug/hits.parquet```
- [ ] **Step 3: If median error after Tasks 7–8 is no longer ~+8, set `hit_frame_calibration_offset` to the new median (integer). If still ~8, leave at 8.**
- [ ] **Step 4: Commit script + any settings change separately**
```bashgit commit -m "$(cat <<'EOF'chore: re-fit hit_frame_calibration_offset after raw-shuttle hit detectionEOF)"```
**Phase 3 exit check:** `y_frac` extreme warnings &lt; 10 on next Colab run; label median frame error not worse than baseline.
---
## Phase 4 — TrackNet signal quality (Colab parity + less fake fill)
### Task 10: Route Colab through backend `TrackNetV3`
**Files:**- Modify: `colab/pipeline.py` (delete or stop using inline class ~L248–493; construct backend model ~L1063)- Test: `backend/tests/test_tracknet.py` (existing — run as regression); add `backend/tests/test_colab_tracknet_import.py` if helpful
- [ ] **Step 1: Replace construction**
```python    from app.models.tracknet import TrackNetV3 as BackendTrackNetV3
    tracknet = BackendTrackNetV3(        str(TRACKNET_PATH),        device=device,        chunk_size=gpu_cfg["tracknet_chunk"],        inpaintnet_path=str(INPAINTNET_PATH) if INPAINTNET_PATH.exists() else None,    )```
Ensure `predict_batch(frames, original_size=(ow, oh))` return schema still matches what Colab appends (`x`, `y`, `confidence`, optional `was_repaired`). Backend already returns this (see `tracknet.py` ~L623–635).
- [ ] **Step 2: Remove dead inline `class TrackNetV3` from `colab/pipeline.py` only after the call site compiles and a dry import works**
```powershell$env:PYTHONPATH = "$PWD\backend"python -c "from app.models.tracknet import TrackNetV3; print(TrackNetV3)"```
- [ ] **Step 3: Commit**
```bashgit commit -m "$(cat <<'EOF'fix: use backend TrackNetV3 in Colab for decode parityEOF)"```
### Task 11: Prefer sparse real shuttle in BST over repaired/interpolated fill
**Files:**- Modify: `backend/app/config/settings.py`- Modify: `backend/app/pipeline/strokes.py`- Modify: `backend/tests/test_strokes.py`
**Locked rule:** When `bst_shuttle_require_raw_observation: bool = True` (new, default **True**), write shuttle tensor coords only if `raw_observed` is True for that frame (confidence-qualified raw TrackNet, not repaired). Provenance still records repaired/interpolated. This prefers sparsity over fake continuity (matches existing comment at strokes.py ~L285–289 and quality design).
- [ ] **Step 1: Failing test**
```pythondef test_build_clip_skips_repaired_and_interpolated_when_require_raw(monkeypatch):    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_norm", "resolution")    monkeypatch.setattr(settings_mod.settings, "bst_shuttle_require_raw_observation", True)    # Frame 1: cleaned has xy + interpolated, raw repaired → tensor stays 0    ...    np.testing.assert_array_equal(clip["shuttle"][1], [0.0, 0.0])    assert clip["_bst_provenance"]["shuttle_repaired"][1] is True```
- [ ] **Step 2: Implement in fill loop**
```python        if use_for_tensor:            if settings.bst_shuttle_require_raw_observation and not raw_observed:                continue  # leave zeros; provenance already recorded            ...```
- [ ] **Step 3: Tests PASS + commit**
```bashgit commit -m "$(cat <<'EOF'fix: feed BST only raw-observed shuttle frames by defaultEOF)"```
### Task 12: Persist TrackNet acceptance diagnostics in Colab log summary
**Files:**- Modify: `colab/pipeline.py` (after shuttle batch loop / cleaning print)
- [ ] **Step 1: Log counts**
```python    n_rep = int(shuttle_df.get("was_repaired", pd.Series(dtype=bool)).fillna(False).sum()) if "was_repaired" in shuttle_df else 0    # Prefer counting from shuttle_raw if present    print(f"  Shuttle repaired frames: {n_rep}/{len(shuttle_df)}")```
No threshold retune in this task beyond what backend TrackNet already uses. If after a Colab re-run jump/spike counts remain extreme, open a follow-up to tune `tracknet_low_conf_max_jump_px` / `tracknet_detection_min_conf` with before/after tables — do not blind-tune here.
- [ ] **Step 2: Commit**
```bashgit commit -m "$(cat <<'EOF'chore: log TrackNet repaired-frame counts in Colab shuttle summaryEOF)"```
**Phase 4 exit check:** repaired fraction ↓ vs 35%; `observed_shuttle_fraction` ↑; `clip_shuttle_valid` mean ↑ toward 80.
---
## Phase 5 — Joint / pose features into BST
### Task 13: Treat sparse partial keypoints as absent for a frame/side
**Files:**- Modify: `backend/app/config/settings.py`- Modify: `backend/app/pipeline/strokes.py` (joint loop ~L321–349)- Modify: `backend/tests/test_strokes.py`
- [ ] **Step 1: Setting**
```python    bst_min_valid_keypoints_fraction: float = 0.5  # of 17 COCO joints```
- [ ] **Step 2: Failing test**
When only 2/17 keypoints are valid, `pose_present_*` is False and joint row is all zeros (not a wild center_align from two points).
```pythondef test_build_clip_marks_sparse_keypoints_absent(monkeypatch):    ...    # 2 valid keypoints only on far player    assert clip["_bst_provenance"]["pose_present_far"][0] is False    assert np.allclose(clip["joints"][0, 0], 0.0)```
- [ ] **Step 3: Implement**
```python                n_valid = int(valid_keypoints.sum())                dense_enough = n_valid >= int(17 * settings.bst_min_valid_keypoints_fraction)                if not dense_enough:                    provenance[f"pose_present_{side}"].append(False)                    provenance[f"pose_keypoint_confidence_{side}"].append(0.0)                    joints[t, p_idx] = 0.0                    continue```
Keep `det_bbox` path when dense enough; do not switch `bst_joint_norm` away from `"bbox"`.
- [ ] **Step 4: Commit**
```bashgit commit -m "$(cat <<'EOF'fix: zero BST joints when keypoint coverage is too sparse per frameEOF)"```
### Task 14: Soft quality penalty for extreme post-norm joint mean (diagnostic)
**Files:**- Modify: `backend/app/pipeline/shared/bst_input_quality.py` — only if provenance gains `joint_abs_mean` from `_build_clip`- Modify: `backend/app/pipeline/strokes.py` to record `joint_abs_mean` in provenance- Modify: `backend/tests/test_bst_input_quality.py`
**Locked rule:** Soft penalty only (`score −0.10` when `joint_abs_mean > 1.0`); **never** a hard reason in Phase 5.
- [ ] **Step 1: Record in provenance after joints filled**
```python    finite = joints[:video_len][np.isfinite(joints[:video_len])]    provenance["joint_abs_mean"] = float(np.mean(np.abs(finite))) if finite.size else 0.0```
- [ ] **Step 2: Soft penalty in evaluator**
```python    joint_abs_mean = float(provenance.get("joint_abs_mean", 0.0))    if joint_abs_mean > settings.bst_joint_abs_mean_soft_max:  # default 1.0        score -= 0.10        reasons.append("extreme_joint_mean")  # soft — not hard_reasons```
- [ ] **Step 3: Tests + commit**
```bashgit commit -m "$(cat <<'EOF'feat: soft-penalize BST clips with extreme post-norm joint meansEOF)"```
**Phase 5 exit check:** joint-mean validator warnings drop; anatomical violation rate stays ~natural; eligible `jnb_*` stats centered near 0.
---
## Phase 6 — Recalibrate quality gate (after Phases 1–5 re-run)
### Task 15: Measure new distributions, then retune settings
**Prerequisite:** One full Colab re-run of `test_match.mp4` with Phases 1–5 shipped; artifacts in `logs/debug/debug_bst_input_quality.parquet`.
**Files:**- Modify: `backend/app/config/settings.py` (only fields justified by data)- Modify: `backend/tests/test_bst_input_quality.py` if expected scores change- Optional script: `scripts/summarize_bst_input_quality.py`
- [ ] **Step 1: Summarize**
```powershellpython scripts/summarize_bst_input_quality.py --parquet ../logs/debug/debug_bst_input_quality.parquet```
Report distributions for: `observed_shuttle_fraction`, `repaired_shuttle_fraction`, `interpolated_shuttle_fraction`, `court_rejected_shuttle_fraction`, `score`, reason frequencies, eligible %.
- [ ] **Step 2: Retune rules (apply only what data supports)**
Allowed knobs (change the minimum set):
| Setting | Current | When to change ||---------|---------|----------------|| `bst_quality_score_min` | 0.70 | If score mass sits 0.55–0.69 but labels show those clips are fine || `bst_max_repaired_shuttle_fraction` | 0.50 | If TrackNet still repairs more than observed but clips are good || `bst_max_interpolated_shuttle_fraction` | 0.25 | Only if cleaning interp is honest and not fake || `bst_min_observed_shuttle_fraction` | 0.35 | Rarely — keep strict |
**Forbidden:** setting `bst_input_quality_enabled=False`; removing long-gap / low-observed hard checks.
- [ ] **Step 3: Update unit tests for any new defaults**
- [ ] **Step 4: Commit**
```bashgit commit -m "$(cat <<'EOF'chore: retune BST input quality thresholds from post-fix phone runEOF)"```
---
## Phase 7 — Verification checklist (same video)
### Task 16: Colab re-run comparison table
- [ ] **Step 1: Run pipeline on `test_match.mp4` with per-video `manual_corners.json` (not default corners)**
- [ ] **Step 2: Fill comparison table**
| Metric | Baseline | After ||--------|----------|-------|| `quality_abstain` % | 68.4 | || `bst_input_eligible` % | 10.6 | || BST inference % | 4.3 | || `court_rejected_shuttle` hard-flag count | 207 | || Mean `clip_shuttle_valid` | 49.8 | || Shuttle conf≥0.5 % | 29.6 | || Repaired % | 35.5 | || `y_frac` extreme warnings | 33 | || Joint \|mean\| majority &lt; 0.5 | no | || Top stroke / unknown % | 76% unknown | |
- [ ] **Step 3: Confirm success criteria met or file follow-ups for missed targets only**
- [ ] **Step 4: Update `HANDOFF.md` current-state section with the new numbers (docs-only commit if user wants)**
---
## Out of scope
- BST fine-tuning / VideoBadminton swap (`scripts/spike_videobadminton`)- Physics override re-enable- Auth, job persistence, frontend redesign- Widening `hit_refine_window` beyond 4
---
## Spec coverage self-review
| Spec / plan requirement | Task ||-------------------------|------|| Resolution mode keeps court-rejected pixels in BST tensor | Task 1 || Court mode still blanks rejected | Task 1 || Quality hard-fail court reject only in court norm | Task 2 || Colab debug persists enriched shuttle + raw | Task 3 || Shuttle-in-court reliability + skip rejection | Tasks 4–5 || Stop silent default_corners | Task 6 || Hits prefer shuttle_raw | Task 7 || Contact y_frac sanity; no widen refine window | Task 8 || Recalibrate hit offset from labels | Task 9 || Colab backend TrackNet parity | Task 10 || BST sparse raw-observed shuttle | Task 11 || Repaired diagnostics | Task 12 || Sparse keypoint → absent joints | Task 13 || Soft extreme joint-mean penalty | Task 14 || Gate retune after evidence | Task 15 || End-to-end verify vs baseline | Task 16 || Amend BST input quality design doc semantics | Task 1 |
**Placeholder scan:** none intentional — thresholds and code paths are specified.
**Type consistency:** `geometry_reliable` / `shuttle_in_court_fraction` on court dict; `bst_shuttle_require_raw_observation`; `joint_abs_mean` in provenance; `court_rejected` provenance unchanged.

RegardsSujith