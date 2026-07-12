from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.shuttle_coach.loader import load_match, capabilities
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import run_metrics
from app.shuttle_coach.metrics.base import MetricResult
from app.shuttle_coach.feedback import derive_findings, prioritize_findings
from app.shuttle_coach.feedback.rules import evaluate_yaml_rules
from app.shuttle_coach.feedback.report import render_report, render_report_json
from app.pipeline.shared.ownership_quality import confident_owner_shots


def analyze(data_dir: str) -> dict[str, Any]:
    """Run shuttle-coach analysis from stored parquet data."""
    tables = load_match(Path(data_dir))
    caps = capabilities(tables)
    model = MatchModel.from_tables(tables)

    results = run_metrics(model, set(caps))

    results_by_id: dict[str, list[MetricResult]] = defaultdict(list)
    for r in results:
        results_by_id[r.metric_id].append(r)

    # Load quality.json if available (from DataQualityStage)
    quality = None
    quality_path = Path(data_dir) / "quality.json"
    if quality_path.exists():
        import json
        try:
            quality = json.loads(quality_path.read_text())
        except Exception:
            pass

    findings = derive_findings(results_by_id, quality=quality)
    findings = prioritize_findings(findings)

    report_md = render_report(findings)
    report_json = render_report_json(findings, model.player_ids, set(caps))

    return {
        "player_ids": model.player_ids,
        "capabilities": sorted(caps),
        "metrics": [r.to_row() for r in results],
        "findings": [vars(f) for f in findings],
        "report_md": report_md,
        "report_json": report_json,
    }


def narrate(question: str, metrics: list[dict], api_key: str) -> str:
    """Answer a coaching question via Gemini."""
    from app.shuttle_coach.narration.gemini import answer as gemini_answer
    return gemini_answer(question, metrics, api_key)


def analyze_from_pipeline(
    analytics: dict[str, Any],
    shuttle_metrics: dict[str, dict],
    player_id: str,
    data_quality: dict | None = None,
) -> dict[str, Any]:
    """Unified coach evaluation for the backend pipeline.

    Merges YAML-defined rules (tactical, fitness, footwork, rally, court)
    with shuttle_coach metric findings (recovery, effectiveness, errors).

    Args:
        analytics: full pipeline analytics dict
        shuttle_metrics: per-player shuttle_coach metric values
        player_id: the player to evaluate
        data_quality: optional quality dict from DataQualityStage.
                      When present, capability_trust flags suppress
                      findings from untrusted capabilities.

    Returns:
        dict with strengths, weaknesses, improvements, drills, evidence, rally_stats
    """
    from app.shuttle_coach.feedback import derive_findings, prioritize_findings

    quality = data_quality or {}
    capability_trust = quality.get("capability_trust", {})

    # Build YAML rule findings (suppressed by capability_trust)
    player_analytics = {
        "tactical": analytics.get("tactical_analytics", {}).get(player_id, {}),
        "fitness": analytics.get("fitness_analytics", {}).get(player_id, {}),
        "footwork": analytics.get("footwork_analytics", {}).get(player_id, {}),
        "rally_stats": _compute_rally_stats(analytics, player_id),
        "court_analysis": _compute_court_analysis(analytics, player_id),
        "opponent": _compute_opponent_data(analytics, player_id),
        "technique": _compute_technique_data(analytics, player_id),
    }

    yaml_findings = evaluate_yaml_rules(player_analytics, player_id)

    # Suppress findings from untrusted capabilities
    if not capability_trust.get("tactical", True):
        yaml_findings = [f for f in yaml_findings
                         if not f.code.startswith(("smash_", "shot_", "net_", "clear_",
                                                    "drop_", "drive_", "rush_"))]
    if not capability_trust.get("movement", True):
        yaml_findings = [f for f in yaml_findings
                         if not f.code in ("recovery_slow", "recovery_fast",
                                           "distance_low", "distance_high",
                                           "front_court_weak", "rear_court_dominant",
                                           "left_bias", "right_bias", "balanced_court")]
    if not capability_trust.get("technique", True):
        yaml_findings = [f for f in yaml_findings
                         if not f.code.startswith("technique_")]

    # Build shuttle_coach metric-based findings
    sc_metrics = shuttle_metrics.get(player_id, {})
    sc_results = []
    for mid, val in sc_metrics.items():
        sc_results.append(MetricResult(
            metric_id=mid,
            player_id=player_id,
            value=val,
            unit="",
            sample_size=1,
            confidence=0.5,
            context={},
        ))
    sc_results_by_id = defaultdict(list)
    for r in sc_results:
        sc_results_by_id[r.metric_id].append(r)
    sc_findings = derive_findings(sc_results_by_id)

    # Merge all findings
    all_findings = prioritize_findings(yaml_findings + sc_findings)

    strengths = []
    weaknesses = []
    improvements = []
    drills = []
    evidence = []

    for f in all_findings:
        entry = {
            "finding": f.detail,
            "metrics": f.evidence,
        }
        evidence.append(entry)

        if f.severity >= 0.6:
            label = f.detail
            if label not in weaknesses:
                weaknesses.append(label)
                improvements.append(label)
        elif f.severity <= 0.4:
            label = f.detail
            if label not in strengths:
                strengths.append(label)
        else:
            label = f.detail
            if label not in weaknesses:
                weaknesses.append(label)

    # Use the dynamic drill matcher instead of static strings
    try:
        from app.shuttle_coach.feedback.drill_matcher import select_drills, format_drill_flat
        structured_drills = select_drills(all_findings, trends=None, quality=quality)
        drills = [format_drill_flat(d) for d in structured_drills]
    except Exception:
        drills = []

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "top_3_improvements": improvements[:3],
        "recommended_drills": drills[:3],
        "recommended_drills_detailed": structured_drills[:3],
        "evidence": evidence,
        "rally_stats": player_analytics.get("rally_stats"),
    }


def _compute_rally_stats(analytics: dict, player_id: str) -> dict:
    rallies_df = analytics.get("_rallies_df")
    shots_df = confident_owner_shots(analytics.get("_shots_df"))
    if rallies_df is None or shots_df is None:
        return {"avg_length": 0, "max_length": 0, "min_length": 0,
                "first_shot_win_rate": 0, "long_rally_pct": 0}

    if hasattr(rallies_df, 'iterrows'):
        rally_lengths = []
        first_shot_wins = 0
        import numpy as np
        for _, rally in rallies_df.iterrows():
            sc = rally.get("shot_count", 0)
            rally_lengths.append(sc)
            if sc > 0:
                start_f = int(rally.get("start_frame", 0))
                first_shots = shots_df[shots_df["frame"] == start_f]
                if len(first_shots) > 0:
                    first_pid = first_shots.iloc[0].get("player_id")
                    winner = rally.get("winner_player_id")
                    if first_pid == player_id and winner == player_id:
                        first_shot_wins += 1

        total_rallies = len(rally_lengths) or 1
        long_rallies = sum(1 for l in rally_lengths if l > 8)
        return {
            "avg_length": float(np.mean(rally_lengths)) if rally_lengths else 0,
            "max_length": int(max(rally_lengths)) if rally_lengths else 0,
            "min_length": int(min(rally_lengths)) if rally_lengths else 0,
            "first_shot_win_rate": first_shot_wins / total_rallies if total_rallies > 0 else 0,
            "long_rally_pct": long_rallies / total_rallies if total_rallies > 0 else 0,
        }
    return {"avg_length": 0, "max_length": 0, "min_length": 0,
            "first_shot_win_rate": 0, "long_rally_pct": 0}


def _compute_court_analysis(analytics: dict, player_id: str) -> dict:
    court_data = analytics.get("court_analytics", {})
    transitions = court_data.get("zone_transitions", [])
    player_zones = [t["zone"] for t in transitions if t.get("player_id") == player_id]
    total = len(player_zones)
    if total == 0:
        return {"front_pct": 0, "mid_pct": 0, "rear_pct": 0,
                "left_pct": 0, "right_pct": 0}

    front = sum(1 for z in player_zones if z.startswith("front"))
    mid = sum(1 for z in player_zones if z.startswith("mid"))
    rear = sum(1 for z in player_zones if z.startswith("rear"))
    left = sum(1 for z in player_zones if z.endswith("left"))
    right = sum(1 for z in player_zones if z.endswith("right"))

    return {
        "front_pct": front / total,
        "mid_pct": mid / total,
        "rear_pct": rear / total,
        "left_pct": left / total,
        "right_pct": right / total,
    }


def _compute_technique_data(analytics: dict, player_id: str) -> dict:
    tech = analytics.get("technical_analytics", {}).get(player_id, {})
    if not tech:
        return {"overall": 0, "shot_count": 0}
    scores = {}
    avg_scores = []
    for stype, data in tech.items():
        score = data.get("avg_score", 0)
        scores[f"{stype}_score"] = score
        avg_scores.append(score)
    scores["overall"] = float(np.mean(avg_scores)) if avg_scores else 0
    scores["shot_count"] = sum(data.get("shot_count", 0) for data in tech.values())
    return scores


def _compute_opponent_data(analytics: dict, player_id: str) -> dict:
    tactical_all = analytics.get("tactical_analytics", {})
    opponent_id = None
    for pid in tactical_all:
        if pid != player_id:
            opponent_id = pid
            break
    if opponent_id is None:
        return {"smash_pct": 0, "net_pct": 0, "clear_pct": 0, "total_shots": 0}

    opp = tactical_all[opponent_id]
    dist = opp.get("shot_distribution", {})
    return {
        "smash_pct": dist.get("smash", 0),
        "net_pct": dist.get("net_shot", 0),
        "clear_pct": dist.get("clear", 0),
        "total_shots": opp.get("total_shots", 0),
    }
