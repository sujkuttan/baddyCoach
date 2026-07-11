# Owner Attribution Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace forced-alternation owner assignment with evidence-gated attribution that can abstain, preserve all shots for rally/global analytics, and restrict player-specific coaching metrics to confidently owned shots only.

**Architecture:** Keep `OwnershipScorer` focused on local geometry/pose evidence and move sequence decisions into a separate anchor/bridge assignment module. `PlayerAttributionStage` will persist both local evidence and final assignment metadata, while downstream player analytics will consume a confident-owner filtered view instead of assuming every shot belongs to a player.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, pytest, FastAPI pipeline artifacts, Shuttle Coach metric engine.

---

## File Structure

- Modify: `backend/app/config/settings.py`
  - Add explicit owner-quality thresholds and bridge controls.
- Create: `backend/app/pipeline/shared/ownership_quality.py`
  - Anchor eligibility, bounded bridge assignment, helper to filter confident-owner shots.
- Modify: `backend/app/pipeline/shared/ownership_scorer.py`
  - Remove BST/turn from production emission, keep BST alpha/class-prefix as diagnostics only.
- Modify: `backend/app/pipeline/attribution.py`
  - Persist local evidence, assign anchors/bridges/unknown, stop filling default sides, keep alpha as post-hoc diagnostic only.
- Modify: `backend/app/shuttle_coach/events.py`
  - Build `player_ids` from stable match entities but filter `shots_of()` to confident owners only.
- Modify: `backend/app/shuttle_coach/engine.py`
  - Use the confident-owner view for player-specific rally/coaching summaries.
- Modify: `colab/pipeline.py`
  - Mirror the backend’s new attribution metadata and coach-input filtering behavior.
- Create: `backend/scripts/evaluate_owner_attribution.py`
  - Offline held-out evaluation and optional logistic calibration report.
- Modify: `backend/tests/test_attribution.py`
  - Replace alpha-as-owner expectations with abstain/anchor/bridge/diagnostic tests.
- Modify: `backend/tests/test_shuttle_coach_events.py`
  - Verify unknown-owner shots do not leak into per-player analytics.
- Create: `backend/tests/test_evaluate_owner_attribution.py`
  - Cover matching, metrics, fold splitting, and deployment recommendation logic.

### Task 1: Diagnostic-Only Ownership Emissions

**Files:**
- Modify: `backend/app/config/settings.py`
- Modify: `backend/app/pipeline/shared/ownership_scorer.py`
- Test: `backend/tests/test_attribution.py`

- [ ] **Step 1: Write the failing scorer tests**

```python
def test_bst_alpha_is_diagnostic_only_not_emission():
    scorer = OwnershipScorer(
        trajectory_weight=1.0,
        court_side_weight=0.0,
        proximity_weight=0.0,
        motion_weight=0.0,
        pose_feasibility_weight=0.0,
        turn_prior_weight=0.0,
        bst_weight=0.0,
        calib_near_mean=0.5,
        calib_near_std=1.0,
        calib_far_mean=0.5,
        calib_far_std=1.0,
    )
    shuttle_df = pd.DataFrame(
        {"frame": [7, 10, 13], "x": [640.0, 660.0, 700.0], "y": [300.0, 310.0, 320.0], "confidence": [0.9, 0.9, 0.9]}
    )
    players = {"players": [{"id": "p1", "side": "near", "detections": []}, {"id": "p2", "side": "far", "detections": []}]}
    court = {"homography": np.eye(3).tolist()}

    low_alpha = scorer.score(shuttle_df, None, players, court, frame=10, shot={"aimplayer_alpha": 0.10})
    high_alpha = scorer.score(shuttle_df, None, players, court, frame=10, shot={"aimplayer_alpha": 0.90})

    assert low_alpha["near_score"] == pytest.approx(high_alpha["near_score"])
    assert low_alpha["far_score"] == pytest.approx(high_alpha["far_score"])
    assert low_alpha["bst_diag_near"] != pytest.approx(high_alpha["bst_diag_near"])


def test_turn_prior_is_reported_but_not_used_in_local_score():
    scorer = OwnershipScorer(
        trajectory_weight=0.0,
        court_side_weight=1.0,
        proximity_weight=0.0,
        motion_weight=0.0,
        pose_feasibility_weight=0.0,
        turn_prior_weight=0.0,
        bst_weight=0.0,
        calib_near_mean=0.5,
        calib_near_std=1.0,
        calib_far_mean=0.5,
        calib_far_std=1.0,
    )
    shuttle_df = pd.DataFrame({"frame": [10], "x": [500.0], "y": [200.0], "confidence": [0.9]})
    players = {"players": [{"id": "p1", "side": "near", "detections": []}, {"id": "p2", "side": "far", "detections": []}]}
    court = {"homography": np.eye(3).tolist()}

    first = scorer.score(shuttle_df, None, players, court, frame=10, prev_owner=None, shot={})
    after_near = scorer.score(shuttle_df, None, players, court, frame=10, prev_owner="p1", shot={})

    assert first["near_score"] == pytest.approx(after_near["near_score"])
    assert first["far_score"] == pytest.approx(after_near["far_score"])
    assert after_near["turn_near"] != pytest.approx(after_near["turn_far"])
```

- [ ] **Step 2: Run the tests to verify current behavior fails**

Run: `cd backend && python -m pytest tests/test_attribution.py -k "diagnostic_only or turn_prior_is_reported" -v`

Expected: FAIL because `OwnershipScorer.score()` still changes `near_score`/`far_score` when `aimplayer_alpha` or `prev_owner` changes.

- [ ] **Step 3: Add settings for anchor/bridge thresholds and deprecate BST emission weight**

```python
# backend/app/config/settings.py
ownership_turn_prior_weight: float = 0.0
ownership_bst_weight: float = 0.0
ownership_min_anchor_confidence: float = 0.68
ownership_min_anchor_margin: float = 0.18
ownership_min_anchor_signals: int = 2
ownership_signal_neutral_epsilon: float = 0.08
ownership_viterbi_bridge_enabled: bool = True
ownership_viterbi_max_bridge_shots: int = 2
ownership_calibration_match_tolerance_frames: int = 15
ownership_calibration_min_accuracy_lift: float = 0.03
ownership_calibration_min_coverage_lift: float = 0.05
```

- [ ] **Step 4: Make BST and turn prior diagnostic-only in the scorer**

```python
# backend/app/pipeline/shared/ownership_scorer.py
result.update({
    "turn_near": round(turn_n, 4),
    "turn_far": round(turn_f, 4),
    "bst_diag_near": round(bst_n, 4),
    "bst_diag_far": round(bst_f, 4),
})

near_score = (
    w_traj * traj_n +
    w_court * court_n +
    w_prox * prox_n +
    w_mot * mot_n +
    w_pose * pose_n
)
far_score = (
    w_traj * traj_f +
    w_court * court_f +
    w_prox * prox_f +
    w_mot * mot_f +
    w_pose * pose_f
)
```

- [ ] **Step 5: Run the scorer tests**

Run: `cd backend && python -m pytest tests/test_attribution.py -k "diagnostic_only or turn_prior_is_reported" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/config/settings.py backend/app/pipeline/shared/ownership_scorer.py backend/tests/test_attribution.py
git commit -m "refactor: keep bst owner signals diagnostic only"
```

### Task 2: Anchor and Bridge Assignment With Explicit Unknowns

**Files:**
- Create: `backend/app/pipeline/shared/ownership_quality.py`
- Modify: `backend/app/pipeline/attribution.py`
- Test: `backend/tests/test_attribution.py`

- [ ] **Step 1: Write failing assignment tests**

```python
def test_unanchored_rally_stays_unknown(tmp_job_dir):
    store = ArtifactStore(tmp_job_dir)
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    store.set("court", {"valid": False})
    store.set_parquet("rallies", pd.DataFrame({"rally_id": [1], "start_frame": [0], "end_frame": [20]}))
    store.set_parquet("shots", pd.DataFrame({"frame": [0, 10, 20], "rally_id": [1, 1, 1], "stroke_type": ["clear"] * 3, "stroke_confidence": [0.8] * 3}))
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0, 10, 20], "x": [100.0, 100.0, 100.0], "y": [200.0, 200.0, 200.0], "confidence": [0.1, 0.1, 0.1]}))

    PlayerAttributionStage().run(store, StageConfig())
    shots = store.get_parquet("shots")

    assert shots["player_id"].isna().all()
    assert set(shots["side"]) == {"unknown"}
    assert shots["owner_confident"].eq(False).all()
    assert set(shots["owner_source"]) == {"unknown"}


def test_short_compatible_gap_bridges_between_anchors(tmp_job_dir, monkeypatch):
    store = ArtifactStore(tmp_job_dir)
    store.set("players", {"players": [{"id": "player_1", "side": "near"}, {"id": "player_2", "side": "far"}]})
    store.set("court", {"valid": False})
    store.set_parquet("rallies", pd.DataFrame({"rally_id": [1], "start_frame": [0], "end_frame": [30]}))
    store.set_parquet("shots", pd.DataFrame({"frame": [0, 10, 20, 30], "rally_id": [1, 1, 1, 1], "stroke_type": ["clear"] * 4, "stroke_confidence": [0.8] * 4}))
    store.set_parquet("shuttle", pd.DataFrame({"frame": [0, 10, 20, 30], "x": [0.0] * 4, "y": [0.0] * 4, "confidence": [0.9] * 4}))

    scripted = [
        {"near_score": 0.82, "far_score": 0.18, "trajectory_near": 0.9, "trajectory_far": 0.1, "court_side_near": 0.8, "court_side_far": 0.2, "proximity_near": 0.8, "proximity_far": 0.2, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.4, "bst_diag_far": 0.6},
        {"near_score": 0.52, "far_score": 0.48, "trajectory_near": 0.5, "trajectory_far": 0.5, "court_side_near": 0.5, "court_side_far": 0.5, "proximity_near": 0.5, "proximity_far": 0.5, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.5, "bst_diag_far": 0.5},
        {"near_score": 0.49, "far_score": 0.51, "trajectory_near": 0.5, "trajectory_far": 0.5, "court_side_near": 0.5, "court_side_far": 0.5, "proximity_near": 0.5, "proximity_far": 0.5, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.5, "bst_diag_far": 0.5},
        {"near_score": 0.19, "far_score": 0.81, "trajectory_near": 0.2, "trajectory_far": 0.8, "court_side_near": 0.2, "court_side_far": 0.8, "proximity_near": 0.2, "proximity_far": 0.8, "motion_near": 0.5, "motion_far": 0.5, "pose_near": 0.5, "pose_far": 0.5, "turn_near": 0.5, "turn_far": 0.5, "bst_diag_near": 0.6, "bst_diag_far": 0.4},
    ]
    seq = iter(scripted)
    monkeypatch.setattr(OwnershipScorer, "score", lambda self, **kwargs: next(seq))

    PlayerAttributionStage().run(store, StageConfig())
    shots = store.get_parquet("shots")

    assert shots["side"].tolist() == ["near", "far", "near", "far"]
    assert shots["owner_source"].tolist() == ["local_anchor", "viterbi_bridge", "viterbi_bridge", "local_anchor"]
    assert shots["owner_confident"].tolist() == [True, True, True, True]
```

- [ ] **Step 2: Run the tests to verify current Viterbi alternation fails them**

Run: `cd backend && python -m pytest tests/test_attribution.py -k "unanchored_rally_stays_unknown or short_compatible_gap_bridges_between_anchors" -v`

Expected: FAIL because the current stage always assigns a player and never emits `"unknown"` sources.

- [ ] **Step 3: Create the owner-quality helper module**

```python
# backend/app/pipeline/shared/ownership_quality.py
from dataclasses import dataclass


@dataclass
class OwnerDecision:
    side: str
    player_id: str | None
    confident: bool
    source: str
    reason: str


def count_independent_signals(score: dict, neutral_epsilon: float) -> int:
    signal_pairs = [
        ("trajectory_near", "trajectory_far"),
        ("court_side_near", "court_side_far"),
        ("proximity_near", "proximity_far"),
        ("motion_near", "motion_far"),
        ("pose_near", "pose_far"),
    ]
    return sum(abs(score[n] - score[f]) >= neutral_epsilon for n, f in signal_pairs)


def is_anchor(score: dict, min_confidence: float, min_margin: float, min_signals: int, neutral_epsilon: float) -> bool:
    confidence = max(score["near_score"], score["far_score"])
    margin = abs(score["near_score"] - score["far_score"])
    return (
        confidence >= min_confidence
        and margin >= min_margin
        and count_independent_signals(score, neutral_epsilon) >= min_signals
    )
```

- [ ] **Step 4: Implement bounded bridge assignment**

```python
# backend/app/pipeline/shared/ownership_quality.py
def assign_rally_owners(indices: list[int], scores: list[dict], players_by_side: dict[str, str], settings) -> dict[int, OwnerDecision]:
    decisions = {
        idx: OwnerDecision(side="unknown", player_id=None, confident=False, source="unknown", reason="no_anchor")
        for idx in indices
    }
    anchors = []
    for pos, (idx, score) in enumerate(zip(indices, scores)):
        if not is_anchor(score, settings.ownership_min_anchor_confidence, settings.ownership_min_anchor_margin, settings.ownership_min_anchor_signals, settings.ownership_signal_neutral_epsilon):
            continue
        side = "near" if score["near_score"] >= score["far_score"] else "far"
        anchors.append((pos, idx, side))
        decisions[idx] = OwnerDecision(side=side, player_id=players_by_side[side], confident=True, source="local_anchor", reason="local_evidence")

    for (left_pos, left_idx, left_side), (right_pos, right_idx, right_side) in zip(anchors, anchors[1:]):
        gap = right_pos - left_pos - 1
        if not settings.ownership_viterbi_bridge_enabled or gap <= 0 or gap > settings.ownership_viterbi_max_bridge_shots:
            continue
        if right_side != (left_side if (right_pos - left_pos) % 2 == 0 else ("far" if left_side == "near" else "near")):
            continue
        cur_side = "far" if left_side == "near" else "near"
        for pos in range(left_pos + 1, right_pos):
            idx = indices[pos]
            decisions[idx] = OwnerDecision(side=cur_side, player_id=players_by_side[cur_side], confident=True, source="viterbi_bridge", reason=f"bounded_bridge:{left_idx}->{right_idx}")
            cur_side = "far" if cur_side == "near" else "near"

    return decisions


def confident_owner_shots(shots_df: pd.DataFrame) -> pd.DataFrame:
    if shots_df.empty or "owner_confident" not in shots_df.columns:
        return shots_df.iloc[0:0].copy()
    mask = shots_df["owner_confident"].fillna(False) & shots_df["player_id"].notna()
    return shots_df[mask].copy()
```

- [ ] **Step 5: Replace the rally Viterbi call in the attribution stage**

```python
# backend/app/pipeline/attribution.py
from app.pipeline.shared.ownership_quality import assign_rally_owners

players_by_side = {p["side"]: p["id"] for p in players_data.get("players", []) if p.get("side") in {"near", "far"}}

for rally_id, emissions_list in rally_emissions.items():
    indices = rally_candidates[rally_id]
    decisions = assign_rally_owners(indices, emissions_list, players_by_side, settings)
    for idx in indices:
        decision = decisions[idx]
        shots_df.at[idx, "player_id"] = decision.player_id
        shots_df.at[idx, "side"] = decision.side
        shots_df.at[idx, "owner_confident"] = decision.confident
        shots_df.at[idx, "owner_source"] = decision.source
        shots_df.at[idx, "owner_reason"] = decision.reason
        shots_df.at[idx, "owner_uncertain"] = not decision.confident
```

- [ ] **Step 6: Remove the default near-side fill and keep attention checks diagnostic-only**

```python
# backend/app/pipeline/attribution.py
if "side" not in shots_df.columns:
    shots_df["side"] = "unknown"
else:
    shots_df["side"] = shots_df["side"].fillna("unknown")

if alpha is None or side not in {"near", "far"}:
    shots_df.at[idx, "attention_owner_match"] = None
    shots_df.at[idx, "attention_alpha_owner"] = None
```

- [ ] **Step 7: Run the attribution tests**

Run: `cd backend && python -m pytest tests/test_attribution.py -v`

Expected: PASS with explicit unknown-owner rows and bounded bridge assignment.

- [ ] **Step 8: Commit**

```bash
git add backend/app/pipeline/shared/ownership_quality.py backend/app/pipeline/attribution.py backend/tests/test_attribution.py
git commit -m "feat: gate owner attribution with anchors and bridges"
```

### Task 3: Filter Player-Specific Coaching and Colab Outputs

**Files:**
- Modify: `backend/app/shuttle_coach/events.py`
- Modify: `backend/app/shuttle_coach/engine.py`
- Modify: `colab/pipeline.py`
- Test: `backend/tests/test_shuttle_coach_events.py`

- [ ] **Step 1: Write failing coach/event tests**

```python
def test_player_ids_ignore_unknown_owner_rows():
    tables = _make_tables()
    tables["shots"] = pd.DataFrame({
        "rally_id": [1, 1, 2],
        "player_id": ["p1", None, "p2"],
        "owner_confident": [True, False, True],
        "shot_type": ["smash", "clear", "drop"],
    })
    model = MatchModel.from_tables(tables)
    assert model.player_ids == ["p1", "p2"]
    assert list(model.shots_of("p1")["shot_type"]) == ["smash"]


def test_shots_of_filters_unconfident_rows():
    tables = _make_tables()
    tables["shots"]["owner_confident"] = [True, False, True]
    model = MatchModel.from_tables(tables)
    p2 = model.shots_of("p2")
    assert p2.empty
```

- [ ] **Step 2: Run the tests to confirm current event filtering is missing**

Run: `cd backend && python -m pytest tests/test_shuttle_coach_events.py -v`

Expected: FAIL because `MatchModel.from_tables()` currently derives `player_ids` from raw `shots["player_id"]` and `shots_of()` returns all rows.

- [ ] **Step 3: Add a confident-owner filtered view to `MatchModel`**

```python
# backend/app/shuttle_coach/events.py
from app.pipeline.shared.ownership_quality import confident_owner_shots

@dataclass
class MatchModel:
    ...
    owner_shots: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_tables(cls, tables: dict[str, pd.DataFrame], match_id: str = "") -> "MatchModel":
        shots = tables.get("shots", pd.DataFrame())
        owner_shots = confident_owner_shots(shots)
        player_ids = sorted(
            pid for pid in pd.unique(owner_shots.get("player_id", pd.Series(dtype=object)))
            if pd.notna(pid)
        )
        return cls(..., shots=shots, owner_shots=owner_shots, player_ids=player_ids)

    def shots_of(self, player_id: str) -> pd.DataFrame:
        if "player_id" not in self.owner_shots.columns:
            return pd.DataFrame()
        return self.owner_shots[self.owner_shots["player_id"] == player_id]
```

- [ ] **Step 4: Update coach summaries to use owner-confident shots**

```python
# backend/app/shuttle_coach/engine.py
from app.pipeline.shared.ownership_quality import confident_owner_shots

def _compute_rally_stats(analytics: dict, player_id: str) -> dict:
    rallies_df = analytics.get("_rallies_df")
    shots_df = confident_owner_shots(analytics.get("_shots_df", np.empty(0)))
    if rallies_df is None or not hasattr(shots_df, "iterrows"):
        return {"avg_length": 0, "max_length": 0, "min_length": 0, "first_shot_win_rate": 0, "long_rally_pct": 0}
```

- [ ] **Step 5: Mirror the metadata in the Colab output adapter**

```python
# colab/pipeline.py
entry = {
    "frame": s["frame"],
    "player_id": s.get("player_id"),
    "side": s.get("side", "unknown"),
    "owner_confident": bool(s.get("owner_confident", False)),
    "owner_source": s.get("owner_source", "unknown"),
    "owner_reason": s.get("owner_reason", "missing"),
}
```

- [ ] **Step 6: Run the focused tests**

Run: `cd backend && python -m pytest tests/test_shuttle_coach_events.py tests/test_attribution.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/shuttle_coach/events.py backend/app/shuttle_coach/engine.py colab/pipeline.py backend/tests/test_shuttle_coach_events.py
git commit -m "feat: filter coaching metrics to confident owner shots"
```

### Task 4: Offline Evaluation and Calibration Report

**Files:**
- Create: `backend/scripts/evaluate_owner_attribution.py`
- Create: `backend/tests/test_evaluate_owner_attribution.py`

- [ ] **Step 1: Write failing evaluator tests**

```python
def test_compute_owner_metrics_tracks_assigned_and_abstained_accuracy():
    matched = pd.DataFrame({
        "label_side": ["near", "far", "near", "far"],
        "pred_side": ["near", "far", "unknown", "near"],
        "owner_source": ["local_anchor", "viterbi_bridge", "unknown", "local_anchor"],
    })
    metrics = compute_owner_metrics(matched)
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["assigned_accuracy"] == pytest.approx(2 / 3)
    assert metrics["overall_accuracy"] == pytest.approx(0.5)
    assert metrics["abstention_rate"] == pytest.approx(0.25)


def test_recommendation_requires_accuracy_and_coverage_lift():
    recommendation = recommend_deploy(
        baseline={"assigned_accuracy": 0.70, "coverage": 0.60},
        candidate={"assigned_accuracy": 0.75, "coverage": 0.68},
        min_accuracy_lift=0.03,
        min_coverage_lift=0.05,
    )
    assert recommendation["deploy"] is True
```

- [ ] **Step 2: Run the evaluator tests**

Run: `cd backend && python -m pytest tests/test_evaluate_owner_attribution.py -v`

Expected: FAIL because the evaluator script and helpers do not exist yet.

- [ ] **Step 3: Create the evaluation script with leave-one-rally-out support**

```python
# backend/scripts/evaluate_owner_attribution.py
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = [
    "ownership_trajectory_near", "ownership_trajectory_far",
    "ownership_court_side_near", "ownership_court_side_far",
    "ownership_proximity_near", "ownership_proximity_far",
    "ownership_motion_near", "ownership_motion_far",
    "ownership_pose_near", "ownership_pose_far",
]


def compute_owner_metrics(matched: pd.DataFrame) -> dict[str, float]:
    assigned = matched[matched["pred_side"].isin(["near", "far"])].copy()
    coverage = len(assigned) / len(matched) if len(matched) else 0.0
    assigned_accuracy = float((assigned["pred_side"] == assigned["label_side"]).mean()) if len(assigned) else 0.0
    overall_accuracy = float((matched["pred_side"] == matched["label_side"]).mean()) if len(matched) else 0.0
    return {
        "coverage": coverage,
        "assigned_accuracy": assigned_accuracy,
        "overall_accuracy": overall_accuracy,
        "abstention_rate": 1.0 - coverage,
        "source_breakdown": matched.groupby("owner_source")["pred_side"].count().to_dict(),
    }


def recommend_deploy(baseline: dict, candidate: dict, min_accuracy_lift: float, min_coverage_lift: float) -> dict:
    accuracy_lift = candidate["assigned_accuracy"] - baseline["assigned_accuracy"]
    coverage_lift = candidate["coverage"] - baseline["coverage"]
    return {"deploy": accuracy_lift >= min_accuracy_lift and coverage_lift >= min_coverage_lift, "accuracy_lift": accuracy_lift, "coverage_lift": coverage_lift}
```

- [ ] **Step 4: Add CLI entry and held-out calibration flow**

```python
# backend/scripts/evaluate_owner_attribution.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-tolerance", type=int, default=settings.ownership_calibration_match_tolerance_frames)
    args = parser.parse_args()

    matched = load_and_match(args.shots, args.labels, args.match_tolerance)
    baseline = compute_owner_metrics(matched.rename(columns={"side": "pred_side", "manual_side": "label_side"}))
    candidate = run_leave_one_rally_out_calibration(matched, FEATURE_COLUMNS)
    report = {"baseline": baseline, "candidate": candidate, "recommendation": recommend_deploy(baseline, candidate, settings.ownership_calibration_min_accuracy_lift, settings.ownership_calibration_min_coverage_lift)}
    Path(args.output).write_text(json.dumps(report, indent=2))
```

- [ ] **Step 5: Run the evaluator tests**

Run: `cd backend && python -m pytest tests/test_evaluate_owner_attribution.py -v`

Expected: PASS.

- [ ] **Step 6: Run a repo-safe verification sweep**

Run: `cd backend && python -m pytest tests/test_attribution.py tests/test_shuttle_coach_events.py tests/test_evaluate_owner_attribution.py -m "not gpu and not model" -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/scripts/evaluate_owner_attribution.py backend/tests/test_evaluate_owner_attribution.py
git commit -m "feat: add offline owner attribution evaluation"
```

## Self-Review

- Spec coverage check:
  - Diagnostic-only BST/alpha handling is covered in Task 1.
  - Anchor, bridge, abstain, and unknown owner persistence are covered in Task 2.
  - Excluding unknown-owner shots from player-specific coaching metrics is covered in Task 3.
  - Offline held-out evaluation and deployment recommendation are covered in Task 4.
- Placeholder scan:
  - All tasks include concrete file paths, commands, and code snippets; no deferred-work markers remain.
- Type consistency:
  - Final attribution metadata uses `owner_confident`, `owner_source`, `owner_reason`, `side`, and `player_id` consistently across Tasks 2-4.
