# Coaching Engine — Design & Implementation Spec

Turning the pipeline from **analytics** into **coaching**. Five workstreams, anchored to the
existing contracts: `Metric`/`MetricResult`/`run_metrics` (`shuttle_coach/metrics/base.py`),
`Finding` + `derive_findings`/`evaluate_yaml_rules` (`shuttle_coach/feedback/rules.py`),
`MatchModel.from_tables` (`shuttle_coach/events.py`), `analyze_from_pipeline` (`shuttle_coach/engine.py`),
and `storage/progress.py`.

Current shots parquet columns (produced by `strokes.py` + `attribution.py` + `rallies.py`):
`shot_id, rally_id, player_id, frame, start_ts, stroke_type, stroke_confidence,
shuttleset_class_id, is_rule_based, is_bst_fallback, hit_confidence, court_x, court_y`.
Rallies: `rally_id, start_frame, end_frame, shot_count, end_reason, winner_player_id,
serving_player_id, start_ts, end_ts`.

---

## Build order (dependencies)

```
Item 5 (trust)  ──► everything (gates which insights are shown / trusted)
   │
   ├─ 5a strict model load + model_health.json
   ├─ 5b benchmark harness (offline, labeled clips)
   └─ 5c DataQualityStage → quality.json + per-capability trust flags
        │
        ▼
Shared: ShotContextStage → shot_events (the enriched rally graph)
        │
        ├─► Item 1 (causal pattern engine)         reads shot_events
        ├─► Item 4 (dynamic drills)                reads findings(1) + trends(3)
        │
Item 2 (technique reference)  ── needs per-stroke FEATURE persistence + reference store + Item 3 history
Item 3 (progress surfaced)    ── needs stable player identity (see Open Dependencies)
```

Phasing: **Phase A** = Item 5 (all) + ShotContextStage. **Phase B** = Item 1 + Item 3.
**Phase C** = Item 2 + Item 4. Item 4 depends on 1 and 3; Item 2 depends on 3.

---

## Open dependencies (resolve before Phase B/C)

1. **Stable player identity across sessions.** Today `player_id` is per-job `player_1/player_2`
   (and `near/far`). Cross-session items (2, 3) need a durable key. **Decision:** add an optional
   `player_key` to the `/upload` and `/process` request (e.g. a user-chosen name/id). Pipeline
   stores `player_key_map` in the job (`{"player_1": "alex", "player_2": "opponent"}`). When absent,
   fall back to job-scoped keys and mark cross-session features `unavailable`. All progress storage
   is keyed by `player_key`, not `player_id`.
2. **court_x/court_y availability.** Set only when court is valid AND pose foot resolves
   (`attribution.py:140-159`). Zone/pressure features degrade gracefully and are gated by Item 5c.

---

# Item 5 — Trust (foundation)

### 5a. Strict model-load verification
**Files:** `models/tracknet.py:275`, `models/bst.py:131`.

Replace silent `load_state_dict(state_dict, strict=False)` with checked loading:

```python
def _checked_load(model, state_dict, *, core_prefixes, max_missing_frac=0.05):
    incompat = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(incompat.missing_keys), list(incompat.unexpected_keys)
    total = sum(1 for _ in model.state_dict())
    missing_frac = len(missing) / max(total, 1)
    core_missing = [k for k in missing if any(k.startswith(p) for p in core_prefixes)]
    status = {
        "loaded": not core_missing and missing_frac <= max_missing_frac,
        "missing_frac": round(missing_frac, 4),
        "n_missing": len(missing), "n_unexpected": len(unexpected),
        "core_missing": core_missing[:10],
    }
    return status
```

- TrackNet `core_prefixes = ("enc1","enc5","out")`; BST `core_prefixes = ("tcn_pose","mlp_head","embedding_tem")`.
- If `status["loaded"]` is False → set `self.model = None` (forces honest fallback) and log a WARNING.
- Persist a merged `model_health.json` artifact per job: `{tracknet:{...}, bst:{...}, rtmpose:{loaded:bool}, court:{loaded:bool}}`.

### 5b. Offline accuracy benchmark harness
**New:** `backend/benchmarks/` + `scripts/run_benchmark.py`.

**Labeled-clip manifest** `benchmarks/manifest/*.json`:
```json
{ "clip_id": "rally_001", "video": "clips/rally_001.mp4", "fps": 30,
  "court_corners_px": [[..bl..],[..br..],[..tl..],[..tr..]],
  "hits": [ {"frame": 42, "stroke_type": "smash", "player_side": "far"},
            {"frame": 78, "stroke_type": "lift",  "player_side": "near"} ] }
```

**Runner** computes, per clip and aggregate:
| Component | Metric | Tolerance | Release gate (initial) |
|---|---|---|---|
| Hit detection | precision / recall / F1 | ±3 frames | F1 ≥ 0.80 |
| Stroke classification | accuracy, macro-F1, confusion matrix | matched hits | macro-F1 ≥ 0.60 |
| Attribution | % correct side | per matched hit | ≥ 0.90 |
| Court homography | mean reprojection error | held-out court pts | ≤ 0.30 m |
| Shuttle tracking | detection rate, mean px error | where GT exists | rate ≥ 0.70 |

Output `benchmarks/results/{ISO_DATE}.json` + `benchmarks/results/{ISO_DATE}.md`.
Pytest: `tests/test_benchmark.py` marked `@pytest.mark.benchmark`, auto-skips when no manifest/data
present, fails when results fall below the gate. Run in CI nightly, not on every commit.

### 5c. Runtime data-quality gate
**New:** `app/pipeline/quality.py` → `DataQualityStage` (run last, before report).
Writes `quality.json`:

```python
{
  "court_valid": bool,                       # court["valid"]
  "shuttle_detection_rate": float,           # shuttle.conf>thr / n
  "pose_coverage": float,                    # frames with nonzero kps / expected
  "bst_fallback_rate": float,                # mean(shots.is_bst_fallback)
  "mean_stroke_confidence": float,
  "court_xy_coverage": float,                # shots with court_x present / n_shots
  "n_shots": int, "n_rallies": int,
  "model_health": {...},                     # from 5a
  "quality_score": float,                    # 0..1 weighted blend
  "tier": "high|medium|low",
  "capability_trust": {                      # per-insight-family gating
     "tactical": bool, "patterns": bool, "technique": bool,
     "movement": bool, "progress": bool
  },
  "caveats": ["BST fell back on 41% of shots", ...]
}
```

Trust rules (defaults in settings):
- `patterns` trustworthy ⇔ `court_valid AND bst_fallback_rate < 0.30 AND court_xy_coverage > 0.6 AND n_shots ≥ 20`.
- `technique` ⇔ `pose_coverage > 0.5 AND rtmpose.loaded`.
- `tactical` ⇔ `bst_fallback_rate < 0.4 AND n_shots ≥ 15`.

**Integration:** every `Finding`/insight carries `data_confidence` and `suppressed` flags.
`analyze_from_pipeline` (and the report) **drops or down-ranks** insights whose capability isn't
trustworthy and always emits `report["data_quality"]` so the UI can show a banner
("Low-confidence run — court not detected; tactical insights hidden").

---

# Shared — `ShotContextStage` (the rally graph)

**New:** `app/pipeline/analytics/shot_context.py`. Runs after `rally_segmentation` (so winner +
rally_id exist) and after attribution. Produces `shot_events.parquet` — one enriched row per shot.

Added columns (on top of existing shot columns):
```
zone                  # front/mid/rear × left/center/right, from court_x/court_y
shot_index_in_rally   # 0-based
is_last_in_rally      # bool
prev_stroke_type, prev_player_id, next_stroke_type, next_player_id
prev_gap_s, next_gap_s            # time to adjacent shots
reaction_time_s       # = prev_gap_s (time since opponent's shot landed-to-hit proxy)
displacement_m        # court distance from this player's PREVIOUS shot position to this one
under_pressure        # reaction_time_s < settings.pressure_time_s OR displacement_m > settings.pressure_dist_m
shot_outcome          # winner | unforced_error | forced_error | net | neutral
won_point             # player_id == rally.winner_player_id AND is_last_in_rally
lost_point            # is_last_in_rally AND player hit last AND lost
led_to_loss_within_k  # rally ended in a loss for this player within k shots of this one
```

Zone helper reuses `court_position.py:_get_zone_from_court`. `under_pressure` is an **approximation**
(documented): true reaction pressure would need ball-trajectory timing; we proxy with inter-shot time
and required court displacement. Pressure thresholds live in settings (`pressure_time_s=0.9`,
`pressure_dist_m=2.5`).

This table is the single join the original critique called for; Items 1 and 4 read it.

---

# Item 1 — Causal pattern engine (conditional outcome stats)

**New metric:** `app/shuttle_coach/metrics/patterns.py`

```python
@register
class ConditionalShotOutcome(Metric):
    metric_id = "patterns.conditional_outcome"
    requires = {"shots", "tactical"}   # needs court_x/y → zones

    def compute(self, m) -> list[MetricResult]:
        ev = load_shot_events(m)        # from shot_events.parquet
        out = []
        for pid in m.player_ids:
            pe = ev[ev.player_id == pid]
            baseline_loss = pe["lost_point"].mean()
            # context key = (stroke_type, zone, pressure_bucket)
            for (stroke, zone, pressed), g in pe.groupby(
                    ["stroke_type","zone","under_pressure"]):
                n = len(g)
                if n < settings.pattern_min_samples:    # default 5
                    continue
                loss = g["led_to_loss_within_k"].mean()
                win  = g["won_point"].mean()
                out.append(MetricResult(
                    metric_id=self.metric_id, player_id=pid,
                    value={"n": n, "loss_rate": round(loss,3),
                           "win_rate": round(win,3),
                           "baseline_loss": round(baseline_loss,3),
                           "wilson_loss_lb": wilson_lower_bound(loss, n),
                           "stroke": stroke, "zone": zone, "pressed": bool(pressed)},
                    unit="rate", sample_size=n,
                    confidence=sample_confidence(n, m),   # scaled by n and (1-fallback)
                    context={"stroke": stroke, "zone": zone, "pressed": bool(pressed)},
                ))
        return out
```

Also emit a **2-shot transition** variant keyed on `(prev_stroke_type → stroke_type)` with the same
loss-rate machinery (catches "your net→lift transition leaks points").

`wilson_lower_bound(p, n, z=1.96)` ranks patterns so small-n flukes don't dominate.
`sample_confidence(n, m) = min(1, n/20) * (1 - bst_fallback_rate)`.

**Findings:** `app/shuttle_coach/feedback/patterns.py`
```python
def derive_pattern_findings(results, quality) -> list[Finding]:
    if not quality["capability_trust"]["patterns"]:
        return []
    findings = []
    for r in results:                       # patterns.conditional_outcome
        v = r.value
        excess = v["loss_rate"] - v["baseline_loss"]
        if v["wilson_loss_lb"] > settings.pattern_loss_floor and \
           excess > settings.pattern_excess_loss:        # default 0.15
            findings.append(Finding(
              code=f"pattern::{v['stroke']}::{v['zone']}::{'pressed' if v['pressed'] else 'free'}",
              player_id=r.player_id,
              severity=round(min(1.0, excess * 2 * r.confidence), 2),
              headline=phrase(v),   # see below
              detail=(f"When you play a {v['stroke']} from the {pretty(v['zone'])}"
                      f"{' under pressure' if v['pressed'] else ''}, you lose the point "
                      f"{v['loss_rate']:.0%} of the time vs {v['baseline_loss']:.0%} overall "
                      f"(n={v['n']})."),
              evidence=[r.metric_id],
            ))
    return findings
```

`phrase()` yields the critique's target line: *"You lift cross-court under pressure and get punished."*
The `code` (`pattern::...`) is the join key the drill matcher (Item 4) uses.

**Settings:** `pattern_min_samples=5`, `pattern_excess_loss=0.15`, `pattern_loss_floor=0.45`,
`pattern_lookahead_k=2`.

**Wire-in:** register metric (import in `metrics/__init__.py`), add `derive_pattern_findings` to
`derive_findings`. Report exposes `report["patterns"]` (sorted by severity, suppressed if untrusted).

---

# Item 2 — Technique reference comparison

### 2a. Persist features, not just the scalar
**File:** `analytics/technical.py`. Currently emits `avg_score` per stroke. Add per-stroke feature
aggregates and persist them:
```python
"technique_features": {              # per player → per stroke
  "smash": {"elbow_extension": {"p50":.., "mean":.., "std":.., "n":..},
            "min_knee_angle": {...}, "hip_shoulder_sep": {...}, "follow_through": {...}},
  ...
}
```
These come from the temporal arrays already computed in `_analyze_swing_mechanics`
(`utils.py` joint angles). Also split each feature by rally-intensity bucket (fast vs slow rally,
using fitness `rally_intensity`) to support the "flattens under pressure" insight.

### 2b. Reference store
**New:** `data/reference/{tier}.json` (`tier ∈ {self, beginner, intermediate, advanced, pro}`):
```json
{ "smash": { "min_knee_angle": {"p10":120,"p50":140,"p90":158}, ... }, ... }
```
Builder `scripts/build_reference.py` aggregates many sessions' `technique_features` into percentile
tables. Ship a seed `intermediate.json` (hand-curated or from labeled clips); `self` is built from
the player's own history (Item 3).

### 2c. Comparison metric + finding
**New:** `app/shuttle_coach/metrics/technique_ref.py`
```python
@register
class TechniqueReference(Metric):
    metric_id = "technique.reference"
    requires = {"technique"}
    def compute(self, m):
        hist = player_history(m.player_key)            # Item 3
        ref  = choose_reference(hist, tier=settings.technique_reference_tier)
        # own-history if >= settings.technique_min_history_sessions else tier file
        results = []
        for pid in m.player_ids:
            feats = current_features(pid)
            for stroke, fmap in feats.items():
                for fname, cur in fmap.items():
                    pctl = percentile_vs_ref(cur["p50"], ref[stroke][fname])
                    degr = pressure_degradation(stroke, fname)   # fast vs slow Δ
                    results.append(MetricResult(
                        metric_id=self.metric_id, player_id=pid,
                        value={"stroke":stroke,"feature":fname,"current":cur["p50"],
                               "ref_p50":ref[stroke][fname]["p50"],"percentile":pctl,
                               "degrades_under_pressure":degr["flag"],"delta_fast":degr["delta"]},
                        unit="percentile", sample_size=cur["n"], confidence=...,
                        context={"reference":ref_source}))
        return results
```
Graceful degradation: own-history → tier file → absolute bounds (today's behaviour) with a
`reference="absolute"` flag so the UI can label it "no baseline yet".

**Finding example:** *"Your smash knee bend (148°) is shallower than 80% of your reference, and it
flattens by 12° in fast rallies — load your legs earlier."*

**Settings:** `technique_reference_tier="intermediate"`, `technique_min_history_sessions=3`,
`technique_pressure_delta_deg=8`.

---

# Item 3 — Progress tracking surfaced (close the loop)

### 3a. Structured snapshot
Define `SessionSnapshot` written by `save_player_session` (extend `storage/progress.py`). Keyed by
**player_key** (Open Dep #1). Stable metric keys:
```python
{ "job_id":..., "timestamp": ISO,
  "shot_distribution": {...}, "error_rate_by_zone": {...}, "error_rate_by_side":{...},
  "top_loss_patterns": [{"code":"pattern::lift::rear_left::pressed","loss_rate":.62,"n":13}, ...],
  "technique_features": {...},             # for Item 2 self-reference
  "fitness": {"distance_m":..,"avg_recovery_s":..,"rally_intensity":..},
  "rally_stats": {...},
  "data_quality": {"tier":"high","quality_score":0.86} }   # weight trends by quality
```

### 3b. Trend engine (generalise existing `compute_trends`)
```python
def compute_metric_trend(player_key, key_path, window=5) -> dict:
    # dot-path access into snapshots; supports scalar + nested dict leaves
    # returns: {direction, slope, pct_change, values[], n_sessions,
    #           first_value, last_value, sparkline[]}
def compare_last_n(player_key, n=5) -> list[dict]:
    # headline movements: [{"metric":"error_rate_by_side.backhand",
    #   "pct_change":-0.12,"direction":"improving","detail":"-12% over last 5 sessions"}]
```
Weight/skip sessions with `data_quality.tier == "low"` so a bad run doesn't fake a trend.

### 3c. API + report + UI
- **Endpoint** (`routes.py`): `GET /api/players/{player_key}/progress?window=5` →
  `{ "n_sessions":.., "trends":[...], "headlines":[...], "sparklines":{...} }`.
  404/empty-state when `< 2` sessions.
- **Report**: add `report["progress"]` = `compare_last_n(...)` top movements when ≥2 sessions.
- **Frontend**: `frontend/src/views/ProgressView.tsx` + `components/ProgressPanel.tsx`
  (Recharts line + sparkline per headline metric), `api.ts: getPlayerProgress(playerKey, window)`.
  Empty state: "Analyze ≥2 sessions for the same player to see trends." Player selector maps
  `player_key`. Add a nav entry in `App.tsx` state machine (`upload|processing|report|progress`).

---

# Item 4 — Dynamic, prescriptive drills

### 4a. Drill catalog
**New:** `app/shuttle_coach/feedback/drills.yaml`
```yaml
drills:
  - id: shadow_six_corner
    targets: ["slow_recovery", "pattern::*::rear_*::pressed"]   # glob match on finding codes
    focus: "court coverage + recovery to base"
    levels:
      foundational: {dosage: "3×30s, walk-through", success: "return to base before next feed"}
      intermediate: {dosage: "4×45s, jog",          success: "<1.0s recovery, 8/10 reps"}
      advanced:     {dosage: "5×60s, sprint + shadow swing", success: "<0.8s recovery under fatigue"}
  - id: net_kill_block_reflex
    targets: ["weak_shot::block", "pattern::block::front_*::pressed"]
    focus: "front-court reflex + tight blocks"
    levels: {...}
```

### 4b. Matcher (replaces static strings)
**New:** `app/shuttle_coach/feedback/drill_matcher.py`
```python
def select_drills(findings, trends, quality, top_n=3) -> list[dict]:
    chosen = []
    for f in sorted(findings, key=lambda x: x.severity, reverse=True):
        for drill in catalog_matching(f.code):          # exact + glob targets
            level = pick_level(f.severity, trend_for(f, trends))
            #   declining trend or severity>0.7 → 'foundational' (remediate)
            #   improving / severity<0.4        → advance one level (progress)
            chosen.append({
              "drill_id": drill["id"], "name": drill["id"].replace("_"," ").title(),
              "focus": drill["focus"], "level": level,
              "dosage": drill["levels"][level]["dosage"],
              "success_criteria": drill["levels"][level]["success"],
              "rationale": f.detail,                      # WHY this drill
              "linked_finding": f.code,
              "trend": trend_for(f, trends),
            })
    return dedup_by_drill_keep_highest_severity(chosen)[:top_n]
```

### 4c. Wiring + back-compat
- `analyze_from_pipeline` (`engine.py:50`) calls `select_drills(...)` instead of appending the
  rule's static drill string.
- Keep `report["recommended_drills"]` (flat strings, rendered from structured) for the current UI;
  add `report["recommended_drills_detailed"]` (structured objects). Frontend `CoachPanel.tsx`
  upgraded to render level/dosage/rationale/success when present, else fall back to strings.
- Trend input comes from Item 3, so drill **level** reflects whether the weakness is improving.

---

## Settings additions (`config/settings.py`)
```python
# Trust / quality
quality_shuttle_conf_thr: float = 0.5
quality_min_shots_tactical: int = 15
quality_max_fallback_patterns: float = 0.30
model_max_missing_frac: float = 0.05
# Shot context / pressure
pressure_time_s: float = 0.9
pressure_dist_m: float = 2.5
pattern_lookahead_k: int = 2
# Patterns
pattern_min_samples: int = 5
pattern_excess_loss: float = 0.15
pattern_loss_floor: float = 0.45
# Technique reference
technique_reference_tier: str = "intermediate"
technique_min_history_sessions: int = 3
technique_pressure_delta_deg: float = 8.0
# Progress
progress_default_window: int = 5
```

## Report schema (additive, back-compat)
```jsonc
{ ...existing...,
  "data_quality": { "tier":"high", "quality_score":0.86, "caveats":[...],
                    "capability_trust": {...} },
  "patterns":   [ {"code":..,"headline":..,"detail":..,"severity":..,"data_confidence":..} ],
  "technique_reference": [ {...} ],
  "progress":   [ {"metric":..,"pct_change":..,"direction":..,"detail":..} ],
  "recommended_drills_detailed": [ {"drill_id":..,"level":..,"dosage":..,"rationale":..} ] }
```

## Test plan
| Item | Tests |
|---|---|
| 5a | `test_model_health.py`: corrupted/renamed state_dict → `loaded=False`, model falls back; good weights → `loaded=True`. |
| 5b | `test_benchmark.py` (marked, skips w/o data): runner emits all metrics; gate comparison logic unit-tested with synthetic GT. |
| 5c | `test_quality_gate.py`: synthetic jobs (no court / high fallback / good) → correct tier + capability flags + suppression. |
| Shared | `test_shot_context.py`: hand-built shots/rallies → correct zone, prev/next, pressure, outcome labels; degrades when court_x absent. |
| 1 | `test_patterns.py`: constructed loss pattern (lift/rear/pressed loses 8/10) surfaces a finding; sub-`min_samples` doesn't; Wilson ranking. |
| 2 | `test_technique_ref.py`: percentile vs seeded ref; own-history vs tier fallback; pressure-degradation flag. |
| 3 | `test_progress_api.py`: 1 session → empty-state; ≥2 → trend direction/pct; low-quality session excluded. |
| 4 | `test_drill_matcher.py`: finding+severity+trend → expected drill id/level/dosage; glob target match; dedup + top-N. |
| E2E | `test_coaching_e2e.py`: synthetic match → report contains patterns/progress/drills/data_quality, suppressed correctly on a low-quality fixture. |

## Effort estimate
- Phase A (Item 5 + ShotContext): ~4–6 dev-days. **Highest leverage** — establishes trust + the join.
- Phase B (Items 1, 3): ~5–7 dev-days (1 is the core coaching value; 3 includes FE).
- Phase C (Items 2, 4): ~4–6 dev-days (2 needs reference data; 4 needs the drill catalog authored).

## Non-goals / explicit limitations
- `under_pressure` is a timing/displacement proxy, not true shuttle-flight reaction time.
- Reference tiers are only as good as the seed data; ship `self`-baseline first, curated tiers later.
- Cross-session features require the `player_key`; without it, Items 2/3 degrade to "unavailable".
