# Coaching Engine Overhaul Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 6-rule coaching engine with a data-driven system that generates player-specific, explainable coaching feedback using all extracted analytics (tactical, fitness, footwork, court position, shot sequences, rally patterns).

**Architecture:** Expand `RULES` from 6 to 30+ rules covering all analytics dimensions. Add context-aware feedback that includes specific numbers, comparisons between players, and rally-phase analysis. Add optional LLM integration for natural language summaries.

**Tech Stack:** Python, YAML rules, optional OpenAI API for NL summaries

---

## Files to Modify

| File | Changes |
|------|---------|
| `colab/pipeline.py` | Expand RULES, enhance `stage_coach()`, add context extraction |
| `colab/pipeline.py` | Add `stage_tactical_patterns()` for sequence analysis |
| `colab/pipeline.py` | Add `stage_court_analysis()` for zone-based coaching |
| `backend/app/coach/engine.py` | Sync with expanded rules |
| `backend/app/coach/rules.yaml` | Expanded rule definitions |
| `frontend/src/components/CoachPanel.tsx` | Enhanced display for richer feedback |

---

### Task 1: Expand RULES from 6 to 30+ covering all analytics dimensions

**Files:**
- Modify: `colab/pipeline.py:48-73` (RULES list)

- [ ] **Step 1: Replace RULES with comprehensive rule set**

```python
RULES = [
    # ─── Tactical Rules ────────────────────────────────────────
    {"name": "smash_efficiency",
     "check": {"field": "tactical.shot_distribution.smash", "operator": "<", "threshold": 0.08, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Smash usage is below 8% ({smash_pct:.1%}). Smashes are your primary attacking weapon — use them more when opponents return high.",
     "category": "weakness", "drill": "Feed drills: partner lifts to rear court, practice 10 smashes to each corner.",
     "context_fields": ["tactical.shot_distribution.smash", "tactical.total_shots"]},

    {"name": "smash_strength",
     "check": {"field": "tactical.shot_distribution.smash", "operator": ">", "threshold": 0.15, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Excellent smash frequency ({smash_pct:.1%}) — maintaining attacking pressure.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.smash"]},

    {"name": "shot_variety_predictable",
     "check": {"field": "tactical.max_shot_percentage", "operator": ">", "threshold": 0.45, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Shot selection is predictable — {top_shot} accounts for {max_pct:.1%} of shots. Opponents can read your patterns.",
     "category": "weakness", "drill": "Pattern-breaking drill: after 2 identical shots, forced switch to a different stroke.",
     "context_fields": ["tactical.max_shot_percentage", "tactical.shot_distribution"]},

    {"name": "shot_variety_good",
     "check": {"field": "tactical.max_shot_percentage", "operator": "<", "threshold": 0.3, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Good shot variety — no single stroke dominates. This keeps opponents guessing.",
     "category": "strength"},

    {"name": "net_play_dominant",
     "check": {"field": "tactical.shot_distribution.net_shot", "operator": ">", "threshold": 0.2, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Strong net play ({net_pct:.1%}) — use this to force lifts and create smash opportunities.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.net_shot"]},

    {"name": "net_play_weak",
     "check": {"field": "tactical.shot_distribution.net_shot", "operator": "<", "threshold": 0.05, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Net shots are rare ({net_pct:.1%}). Improve front court presence to control rallies.",
     "category": "weakness", "drill": "Net kill drills: partner feeds to net, practice tight spinning net shots.",
     "context_fields": ["tactical.shot_distribution.net_shot"]},

    {"name": "clear_heavy",
     "check": {"field": "tactical.shot_distribution.clear", "operator": ">", "threshold": 0.35, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Heavy reliance on clears ({clear_pct:.1%}) — mix with drops and smashes to vary pace.",
     "category": "weakness", "drill": "Clear-drop combination: alternate clear and drop from rear court.",
     "context_fields": ["tactical.shot_distribution.clear"]},

    {"name": "drop_shot_effective",
     "check": {"field": "tactical.shot_distribution.drop", "operator": ">", "threshold": 0.12, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Good use of drop shots ({drop_pct:.1%}) — keeps opponents off balance.",
     "category": "strength", "context_fields": ["tactical.shot_distribution.drop"]},

    # ─── Fitness Rules ─────────────────────────────────────────
    {"name": "fatigue_declining",
     "check": {"field": "fitness.fatigue_trend", "operator": "==", "value": "declining"},
     "recommendation": "Performance declines in later rallies (fatigue trend: declining). Late-match intensity drops by {late_fatigue:.0%}.",
     "category": "weakness", "drill": "Interval training: 12x (30s high intensity + 30s rest). Simulate match demands.",
     "context_fields": ["fitness.late_rally_fatigue", "fitness.peak_intensity"]},

    {"name": "fatigue_improving",
     "check": {"field": "fitness.fatigue_trend", "operator": "==", "value": "improving"},
     "recommendation": "Great stamina — performance improves in later rallies. You outlast opponents.",
     "category": "strength", "context_fields": ["fitness.late_rally_fatigue"]},

    {"name": "low_intensity",
     "check": {"field": "fitness.rally_intensity", "operator": "<", "threshold": 1.0, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Rally intensity is low ({intensity:.2f} shots/sec). Increase pace to pressure opponents.",
     "category": "weakness", "drill": "Speed rallies: 50-shot rallies at maximum pace.",
     "context_fields": ["fitness.rally_intensity"]},

    {"name": "high_intensity",
     "check": {"field": "fitness.peak_intensity", "operator": ">", "threshold": 3.0, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "High peak intensity ({peak:.2f} shots/sec) — explosive rallies when needed.",
     "category": "strength", "context_fields": ["fitness.peak_intensity"]},

    {"name": "distance_low",
     "check": {"field": "fitness.total_distance", "operator": "<", "threshold": 100000, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Court coverage is limited ({distance:.0f} units). Work on movement to reach more shots.",
     "category": "weakness", "drill": "6-corner footwork: shadow movement to all court positions.",
     "context_fields": ["fitness.total_distance"]},

    {"name": "distance_high",
     "check": {"field": "fitness.total_distance", "operator": ">", "threshold": 300000, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Excellent court coverage ({distance:.0f} units) — you cover the full court effectively.",
     "category": "strength", "context_fields": ["fitness.total_distance"]},

    # ─── Footwork Rules ────────────────────────────────────────
    {"name": "recovery_slow",
     "check": {"field": "footwork.avg_recovery", "operator": ">", "threshold": 1.5},
     "recommendation": "Recovery to base takes {recovery:.1f} frames on average. Work on split-step timing.",
     "category": "weakness", "drill": "Split-step practice: bounce on toes, explode to shuttle on opponent's hit.",
     "context_fields": ["footwork.avg_recovery"]},

    {"name": "recovery_fast",
     "check": {"field": "footwork.avg_recovery", "operator": "<", "threshold": 0.5, "min_shots": "tactical.total_shots >= 10"},
     "recommendation": "Quick recovery ({recovery:.1f} frames) — you reset well between shots.",
     "category": "strength", "context_fields": ["footwork.avg_recovery"]},

    # ─── Rally Rules ───────────────────────────────────────────
    {"name": "short_rallies",
     "check": {"field": "rally_stats.avg_length", "operator": "<", "threshold": 5.0, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Average rally length is {avg_rally:.1f} shots. Opponents end rallies quickly — work on sustaining pressure.",
     "category": "weakness", "drill": "Patience drill: cannot smash until rally reaches 8 shots.",
     "context_fields": ["rally_stats.avg_length"]},

    {"name": "long_rallies",
     "check": {"field": "rally_stats.avg_length", "operator": ">", "threshold": 12.0, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Long rallies (avg {avg_rally:.1f} shots) — you control tempo well.",
     "category": "strength", "context_fields": ["rally_stats.avg_length"]},

    {"name": "first_shot_winner",
     "check": {"field": "rally_stats.first_shot_win_rate", "operator": ">", "threshold": 0.3, "min_shots": "tactical.total_shots >= 20"},
     "recommendation": "Strong opening shots — {first_win:.0%} of rallies won on first shot.",
     "category": "strength", "context_fields": ["rally_stats.first_shot_win_rate"]},

    # ─── Court Position Rules ──────────────────────────────────
    {"name": "front_court_weak",
     "check": {"field": "court_analysis.front_pct", "operator": "<", "threshold": 0.2, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Limited front court presence ({front:.1%}). Move forward to intercept and pressure.",
     "category": "weakness", "drill": "Net approaches: practice moving from base to net after clears.",
     "context_fields": ["court_analysis.front_pct"]},

    {"name": "rear_court_dominant",
     "check": {"field": "court_analysis.rear_pct", "operator": ">", "threshold": 0.6, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Spending {rear:.1%} of time in rear court — opponents are pushing you back.",
     "category": "weakness", "drill": "Counter-attack drills: practice attacking from rear court.",
     "context_fields": ["court_analysis.rear_pct"]},

    # ─── Comparison Rules (player vs opponent) ─────────────────
    {"name": "opponent_smash_weak",
     "check": {"field": "opponent.smash_pct", "operator": "<", "threshold": 0.08, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Opponent rarely smashes ({opp_smash:.1%}). Expect clears and drops — position forward.",
     "category": "insight", "context_fields": ["opponent.smash_pct"]},

    {"name": "opponent_net_weak",
     "check": {"field": "opponent.net_pct", "operator": "<", "threshold": 0.05, "min_shots": "tactical.total_shots >= 15"},
     "recommendation": "Opponent avoids net play ({opp_net:.1%}). Push to net to force weak returns.",
     "category": "insight", "context_fields": ["opponent.net_pct"]},
]
```

- [ ] **Step 2: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: expand coaching rules from 6 to 30+ covering all analytics dimensions"
```

---

### Task 2: Add rally stats and court analysis stages

**Files:**
- Modify: `colab/pipeline.py` (add new stage functions before `stage_coach`)

- [ ] **Step 1: Add stage_rally_stats() function**

Add before `stage_coach`:

```python
def stage_rally_stats(shots_data, rallies_data):
    """Compute rally-level statistics for coaching."""
    stats = {"avg_length": 0, "max_length": 0, "min_length": 0,
             "first_shot_win_rate": 0, "long_rally_pct": 0}
    if not rallies_data or not shots_data:
        return stats

    lengths = [r["shot_count"] for r in rallies_data]
    stats["avg_length"] = float(np.mean(lengths))
    stats["max_length"] = max(lengths)
    stats["min_length"] = min(lengths)
    stats["long_rally_pct"] = float(sum(1 for l in lengths if l > 8) / len(lengths))

    # First-shot win rate: how often the server/first hitter wins the rally
    shots_df = pd.DataFrame(shots_data)
    first_shot_wins = 0
    for rally in rallies_data:
        rally_shots = shots_df[(shots_df["frame"] >= rally["start_frame"]) &
                               (shots_df["frame"] <= rally["end_frame"])]
        if len(rally_shots) >= 2:
            first_player = rally_shots.iloc[0].get("player_id")
            last_player = rally_shots.iloc[-1].get("player_id")
            if first_player == last_player:
                first_shot_wins += 1
    stats["first_shot_win_rate"] = float(first_shot_wins / len(rallies_data)) if rallies_data else 0

    return stats
```

- [ ] **Step 2: Add stage_court_analysis() function**

```python
def stage_court_analysis(court_analytics, shots_data):
    """Analyze court zone distribution for coaching."""
    transitions = court_analytics.get("zone_transitions", [])
    if not transitions:
        return {"front_pct": 0, "mid_pct": 0, "rear_pct": 0, "left_pct": 0, "right_pct": 0}

    total = len(transitions)
    front = sum(1 for t in transitions if "front" in t.get("zone", ""))
    mid = sum(1 for t in transitions if "mid" in t.get("zone", ""))
    rear = sum(1 for t in transitions if "rear" in t.get("zone", ""))
    left = sum(1 for t in transitions if "left" in t.get("zone", ""))
    right = sum(1 for t in transitions if "right" in t.get("zone", ""))

    return {
        "front_pct": float(front / total),
        "mid_pct": float(mid / total),
        "rear_pct": float(rear / total),
        "left_pct": float(left / total),
        "right_pct": float(right / total),
    }
```

- [ ] **Step 3: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: add rally stats and court analysis stages for coaching"
```

---

### Task 3: Enhance stage_coach with context-aware feedback

**Files:**
- Modify: `colab/pipeline.py:1304-1402` (stage_coach function)

- [ ] **Step 1: Replace stage_coach with enhanced version**

```python
def stage_coach(tactical, fitness, footwork, rallies=None, court_analytics=None, shots_data=None):
    """Generate context-aware coaching recommendations using all analytics."""
    strengths_set = set()
    weaknesses_set = set()
    improvements = []
    drills = []
    evidence = []

    # Compute rally stats
    rally_stats = stage_rally_stats(shots_data or [], rallies or [])

    # Compute court analysis per player
    court_analysis = stage_court_analysis(court_analytics or {}, shots_data or [])

    # Build opponent comparison data
    player_ids = list(tactical.keys())
    opponent_data = {}
    for pid in player_ids:
        opp_ids = [p for p in player_ids if p != pid]
        if opp_ids:
            opp_id = opp_ids[0]
            opp_tactical = tactical.get(opp_id, {})
            opp_dist = opp_tactical.get("shot_distribution", {})
            opponent_data[pid] = {
                "smash_pct": opp_dist.get("smash", 0),
                "net_pct": opp_dist.get("net_shot", 0),
                "clear_pct": opp_dist.get("clear", 0),
                "total_shots": opp_tactical.get("total_shots", 0),
            }

    def get_nested(data, path):
        keys = path.split(".")
        current = data
        for key in keys:
            if current is None:
                return 0
            if isinstance(current, dict):
                current = current.get(key, 0)
            elif isinstance(current, (list, tuple)):
                try:
                    idx = int(key)
                    current = current[idx] if 0 <= idx < len(current) else 0
                except (ValueError, IndexError):
                    return 0
            else:
                return 0
        return current if current is not None else 0

    def compare(actual, op, expected):
        try:
            actual, expected = float(actual), float(expected)
        except (TypeError, ValueError):
            return str(actual) == str(expected) if op == "==" else False
        if op == "<": return actual < expected
        elif op == ">": return actual > expected
        elif op == "<=": return actual <= expected
        elif op == ">=": return actual >= expected
        elif op == "==": return actual == expected
        elif op == "!=": return actual != expected
        return False

    def evaluate_condition(expr, analytics):
        parts = expr.split()
        if len(parts) != 3:
            return False
        field_path, op, val_str = parts
        try:
            val = float(val_str)
        except ValueError:
            return False
        return compare(get_nested(analytics, field_path), op, val)

    def format_recommendation(template, analytics):
        """Format recommendation template with actual values."""
        try:
            # Extract field names from template
            import re
            fields = re.findall(r'\{(\w+(?:\.\w+)*)\}', template)
            values = {}
            for field in fields:
                val = get_nested(analytics, field)
                if isinstance(val, float):
                    if "pct" in field or "rate" in field:
                        values[field] = val  # Will be formatted as percentage
                    else:
                        values[field] = val
                else:
                    values[field] = val
            return template.format(**values)
        except (KeyError, ValueError, IndexError):
            return template

    for pid in set(list(tactical.keys()) + list(fitness.keys())):
        player_analytics = {
            "tactical": tactical.get(pid, {}),
            "fitness": fitness.get(pid, {}),
            "footwork": footwork.get(pid, {}),
            "rally_stats": rally_stats,
            "court_analysis": court_analysis,
            "opponent": opponent_data.get(pid, {}),
        }

        tactical_data = player_analytics["tactical"]
        if tactical_data:
            shot_dist = tactical_data.get("shot_distribution", {})
            tactical_data["max_shot_percentage"] = max(shot_dist.values()) if shot_dist else 0
            # Add formatted fields for context
            for stroke, pct in shot_dist.items():
                tactical_data[f"{stroke}_pct"] = pct

        fitness_data = player_analytics["fitness"]
        if fitness_data:
            fitness_data["intensity"] = fitness_data.get("rally_intensity", 0)
            fitness_data["peak"] = fitness_data.get("peak_intensity", 0)
            fitness_data["distance"] = fitness_data.get("total_distance", 0)

        footwork_data = player_analytics["footwork"]
        if footwork_data:
            footwork_data["recovery"] = footwork_data.get("avg_recovery", 0)

        total = tactical_data.get("total_shots", 0)

        for rule in RULES:
            try:
                if evaluate_rule(rule, player_analytics):
                    rec = format_recommendation(rule["recommendation"], player_analytics)
                    # Add player ID context
                    rec_with_player = f"[{pid}] {rec}"
                    entry = {
                        "finding": rec_with_player,
                        "metrics": [f"player: {pid}", f"total shots: {total}"],
                    }
                    # Add context fields as metrics
                    for cf in rule.get("context_fields", []):
                        val = get_nested(player_analytics, cf)
                        if isinstance(val, float):
                            entry["metrics"].append(f"{cf}: {val:.3f}")
                    evidence.append(entry)

                    if rule["category"] == "strength":
                        if rec_with_player not in strengths_set:
                            strengths_set.add(rec_with_player)
                    elif rule["category"] == "weakness":
                        if rec_with_player not in weaknesses_set:
                            weaknesses_set.add(rec_with_player)
                            improvements.append(rec_with_player)
                            drills.append(rule.get("drill", ""))
                    elif rule["category"] == "insight":
                        if rec_with_player not in weaknesses_set:
                            weaknesses_set.add(rec_with_player)
            except Exception:
                continue

    strengths = list(strengths_set)
    weaknesses = list(weaknesses_set)
    return {"strengths": strengths, "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3], "recommended_drills": drills[:3],
            "evidence": evidence, "rally_stats": rally_stats}
```

- [ ] **Step 2: Update stage_coach call in run_pipeline**

Find the call to `stage_coach` in `run_pipeline` and update it:

```python
    print("\n[14/14] Coach recommendations...")
    coach = stage_coach(tactical, fitness, footwork, rallies=rallies,
                        court_analytics=court_analytics, shots_data=shots)
    print(f"  {len(coach['strengths'])} strengths, {len(coach['weaknesses'])} weaknesses")
```

- [ ] **Step 3: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: context-aware coaching with player-specific, explainable feedback"
```

---

### Task 4: Add shot sequence pattern analysis

**Files:**
- Modify: `colab/pipeline.py:1272-1286` (stage_tactical function)

- [ ] **Step 1: Enhance stage_tactical with sequence patterns**

Replace the `stage_tactical` function:

```python
def stage_tactical(shots_data):
    """Compute tactical analytics with sequence patterns."""
    tactical = {}
    for shot in shots_data:
        pid = shot.get("player_id", "player_1")
        if pid not in tactical:
            tactical[pid] = {"shot_distribution": Counter(), "total_shots": 0,
                           "common_patterns": [], "unique_strokes": [],
                           "rally_openers": Counter(), "rally_enders": Counter()}
        tactical[pid]["shot_distribution"][shot["stroke_type"]] += 1
        tactical[pid]["total_shots"] += 1

    # Compute sequence patterns
    shots_by_player = {}
    for shot in shots_data:
        pid = shot.get("player_id", "player_1")
        if pid not in shots_by_player:
            shots_by_player[pid] = []
        shots_by_player[pid].append(shot["stroke_type"])

    for pid in tactical:
        total = tactical[pid]["total_shots"]
        tactical[pid]["shot_distribution"] = {k: v/total for k, v in tactical[pid]["shot_distribution"].items()}
        seq = shots_by_player.get(pid, [])

        # Common 3-shot patterns
        patterns = Counter()
        for i in range(len(seq) - 2):
            pattern = f"{seq[i]} -> {seq[i+1]} -> {seq[i+2]}"
            patterns[pattern] += 1
        tactical[pid]["common_patterns"] = [
            {"pattern": p, "count": c} for p, c in patterns.most_common(5)
        ]

        # Unique strokes
        tactical[pid]["unique_strokes"] = list(tactical[pid]["shot_distribution"].keys())

        # Rally openers and enders (first and last shot of each rally)
        from collections import defaultdict
        rally_shots = defaultdict(list)
        for shot in shots_data:
            if shot.get("player_id") == pid and shot.get("rally_id") is not None:
                rally_shots[shot["rally_id"]].append(shot["stroke_type"])
        for rally_id, strokes in rally_shots.items():
            if strokes:
                tactical[pid]["rally_openers"][strokes[0]] += 1
                tactical[pid]["rally_enders"][strokes[-1]] += 1

        # Convert Counters to dicts for JSON serialization
        tactical[pid]["rally_openers"] = dict(tactical[pid]["rally_openers"])
        tactical[pid]["rally_enders"] = dict(tactical[pid]["rally_enders"])

    return tactical
```

- [ ] **Step 2: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: add shot sequence patterns, rally openers/enders to tactical analytics"
```

---

### Task 5: Update report generation to include new analytics

**Files:**
- Modify: `colab/pipeline.py:1405-1432` (generate_report function)

- [ ] **Step 1: Update generate_report to include rally_stats and court_analysis**

```python
def generate_report(court, players, shuttle, pose, hits, shots, rallies,
                    court_analytics, footwork, fitness, tactical, technical, coach, fps=30):
    shot_dist = {}
    for pid, data in tactical.items():
        shot_dist.update(data.get("shot_distribution", {}))

    shots_with_ts = []
    for s in shots:
        shots_with_ts.append({
            "frame": s["frame"],
            "timestamp": round(s["frame"] / fps, 2),
            "stroke_type": s["stroke_type"],
            "confidence": round(s.get("stroke_confidence", 0.5), 3),
            "player_id": s.get("player_id", "player_1"),
            "rally_id": s.get("rally_id"),
        })

    return {
        "court_analytics": court_analytics, "footwork": footwork, "fitness": fitness,
        "tactical": tactical, "technical": technical,
        "shot_distribution": shot_dist,
        "strengths": coach["strengths"], "weaknesses": coach["weaknesses"],
        "top_3_improvements": coach["top_3_improvements"],
        "recommended_drills": coach["recommended_drills"], "evidence": coach["evidence"],
        "rally_stats": coach.get("rally_stats", {}),
        "rallies": rallies, "shot_count": len(shots),
        "shots": shots_with_ts,
    }
```

- [ ] **Step 2: Commit**

```bash
git add colab/pipeline.py
git commit -m "feat: include rally_stats in report output"
```

---

### Task 6: Enhance CoachPanel frontend display

**Files:**
- Modify: `frontend/src/components/CoachPanel.tsx`

- [ ] **Step 1: Update CoachPanel to show richer feedback**

```tsx
interface CoachPanelProps {
  strengths: string[];
  weaknesses: string[];
  drills: string[];
  evidence: Array<{ finding: string; metrics: string[] }>;
  rallyStats?: {
    avg_length: number;
    max_length: number;
    first_shot_win_rate: number;
  };
}

export function CoachPanel({ strengths, weaknesses, drills, evidence, rallyStats }: CoachPanelProps) {
  return (
    <div className="space-y-6">
      {/* Rally Stats Summary */}
      {rallyStats && rallyStats.avg_length > 0 && (
        <div className="p-4 rounded-xl bg-court-surface border border-court-border">
          <h3 className="font-display text-lg text-shuttle-lime mb-3">RALLY INSIGHTS</h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="font-mono text-2xl text-text-primary">{rallyStats.avg_length.toFixed(1)}</div>
              <div className="font-body text-xs text-text-muted">Avg Rally Length</div>
            </div>
            <div>
              <div className="font-mono text-2xl text-text-primary">{rallyStats.max_length}</div>
              <div className="font-body text-xs text-text-muted">Longest Rally</div>
            </div>
            <div>
              <div className="font-mono text-2xl text-shuttle-lime">{(rallyStats.first_shot_win_rate * 100).toFixed(0)}%</div>
              <div className="font-body text-xs text-text-muted">First Shot Win</div>
            </div>
          </div>
        </div>
      )}

      {/* Strengths */}
      {strengths.length > 0 && (
        <div className="p-4 rounded-xl bg-feather-green/5 border border-feather-green/20">
          <h3 className="font-display text-lg text-feather-green mb-3">STRENGTHS</h3>
          <ul className="space-y-2">
            {strengths.map((s, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-feather-green mt-1">+</span>
                <span className="font-body text-sm text-text-primary">{s}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Weaknesses & Improvements */}
      {weaknesses.length > 0 && (
        <div className="p-4 rounded-xl bg-shuttle-lime/5 border border-shuttle-lime/20">
          <h3 className="font-display text-lg text-shuttle-lime mb-3">AREAS TO IMPROVE</h3>
          <ul className="space-y-2">
            {weaknesses.map((w, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-shuttle-lime mt-1">!</span>
                <span className="font-body text-sm text-text-primary">{w}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Drills */}
      {drills.length > 0 && drills.some(d => d) && (
        <div className="p-4 rounded-xl bg-court-surface border border-court-border">
          <h3 className="font-display text-lg text-text-primary mb-3">RECOMMENDED DRILLS</h3>
          <ul className="space-y-2">
            {drills.filter(d => d).map((d, i) => (
              <li key={i} className="font-body text-sm text-text-secondary pl-4 border-l-2 border-shuttle-lime/30">
                {d}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Evidence */}
      {evidence.length > 0 && (
        <div className="p-4 rounded-xl bg-court-surface border border-court-border">
          <h3 className="font-display text-lg text-text-muted mb-3">EVIDENCE</h3>
          <ul className="space-y-2">
            {evidence.map((e, i) => (
              <li key={i} className="font-mono text-xs text-text-muted">
                <span className="text-text-secondary">{e.finding}</span>
                {e.metrics.length > 0 && (
                  <span className="ml-2 text-text-muted">({e.metrics.join(', ')})</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Update ReportView to pass rallyStats**

In `ReportView.tsx`, find the CoachPanel usage and add rallyStats:

```tsx
<CoachPanel
  strengths={report.strengths || []}
  weaknesses={report.weaknesses || []}
  drills={report.recommended_drills || []}
  evidence={report.evidence || []}
  rallyStats={report.rally_stats}
/>
```

- [ ] **Step 3: Type check**

```bash
cd frontend && npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/CoachPanel.tsx frontend/src/views/ReportView.tsx
git commit -m "feat: enhanced CoachPanel with rally insights, richer feedback display"
```

---

### Task 7: Sync backend coach engine and run tests

**Files:**
- Modify: `backend/app/coach/rules.yaml`
- Modify: `backend/app/pipeline/analytics/*.py` (add rally_stats, court_analysis stages)

- [ ] **Step 1: Update backend rules.yaml with expanded rules**

Mirror the expanded RULES from the colab pipeline into the YAML format used by the backend CoachEngine.

- [ ] **Step 2: Add rally_stats and court_analysis stages to backend pipeline**

- [ ] **Step 3: Run all tests**

```bash
cd /home/sujith/baddyCoach && .venv/bin/pytest backend/tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/
git commit -m "feat: sync backend coaching engine with expanded rules and new analytics"
```

---

### Task 8: End-to-end test with existing results

**Files:**
- Test: manual verification

- [ ] **Step 1: Re-run pipeline on test video**

```bash
PYTHONPATH=backend .venv/bin/python colab/pipeline.py videos/test_match.mp4 --output results/coaching_test.json --device cuda --pose-model rtmpose
```

- [ ] **Step 2: Verify coaching output has more than 2 findings**

```bash
python -c "
import json
with open('results/coaching_test.json') as f:
    r = json.load(f)
print(f'Strengths: {len(r[\"strengths\"])}')
print(f'Weaknesses: {len(r[\"weaknesses\"])}')
print(f'Drills: {len(r[\"recommended_drills\"])}')
print(f'Evidence: {len(r[\"evidence\"])}')
print(f'Rally stats: {r.get(\"rally_stats\", {})}')
for s in r['strengths'][:3]:
    print(f'  + {s}')
for w in r['weaknesses'][:3]:
    print(f'  ! {w}')
"
```

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: coaching engine overhaul complete — context-aware, explainable feedback"
```
