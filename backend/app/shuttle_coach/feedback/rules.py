from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

from app.shuttle_coach.metrics.base import MetricResult


@dataclass
class Finding:
    code: str
    player_id: str | None
    severity: float  # 0..1
    headline: str
    detail: str
    evidence: list[str] = field(default_factory=list)


def _check_slow_recovery(results: list[MetricResult]) -> list[Finding]:
    findings = []
    for r in results:
        if r.metric_id == "movement.recovery_time" and isinstance(r.value, (int, float)) and r.value > 0.8:
            severity = min(1.0, (r.value - 0.8) / 0.2) if r.value <= 1.0 else 1.0
            findings.append(Finding(
                code="slow_recovery",
                player_id=r.player_id,
                severity=round(severity, 2),
                headline="Slow Recovery Time",
                detail=f"Average recovery time is {r.value:.2f}s, which is above the 0.8s threshold.",
                evidence=["movement.recovery_time"],
            ))
    return findings


def _check_weak_shots(results: list[MetricResult]) -> list[Finding]:
    findings = []
    for r in results:
        if r.metric_id == "shots.effectiveness" and isinstance(r.value, dict):
            for shot_type, effectiveness in r.value.items():
                if effectiveness < 0.35:
                    severity = min(1.0, (0.35 - effectiveness) / 0.35)
                    findings.append(Finding(
                        code="weak_shot",
                        player_id=r.player_id,
                        severity=round(severity, 2),
                        headline=f"Weak {shot_type} Effectiveness",
                        detail=f"{shot_type} win rate is {effectiveness * 100:.1f}%, below the 35% threshold.",
                        evidence=["shots.effectiveness"],
                    ))
    return findings


def _check_high_unforced(results: list[MetricResult]) -> list[Finding]:
    findings = []
    for r in results:
        if r.metric_id == "errors.location_reason" and isinstance(r.value, dict):
            unforced_pct = r.value.get("unforced", 0)
            if unforced_pct > 30:
                severity = min(1.0, (unforced_pct - 30) / 40)
                findings.append(Finding(
                    code="high_unforced_errors",
                    player_id=r.player_id,
                    severity=round(severity, 2),
                    headline="High Unforced Error Rate",
                    detail=f"Unforced errors account for {unforced_pct:.1f}% of lost rallies, above the 30% threshold.",
                    evidence=["errors.location_reason"],
                ))
    return findings


def derive_findings(results_by_id: dict[str, list[MetricResult]],
                    quality: dict | None = None) -> list[Finding]:
    """Derive findings from metrics grouped by metric_id.

    Args:
        results_by_id: MetricResults grouped by metric_id.
        quality: Optional data-quality dict.  When present, capability_trust
                 flags suppress findings from untrusted capabilities.

    Returns:
        List of Finding objects.
    """
    findings: list[Finding] = []
    all_results = []
    for results in results_by_id.values():
        all_results.extend(results)

    findings.extend(_check_slow_recovery(all_results))
    findings.extend(_check_weak_shots(all_results))
    findings.extend(_check_high_unforced(all_results))

    # Pattern findings (requires patterns metric)
    pattern_results = results_by_id.get("patterns.conditional_outcome", [])
    pattern_results += results_by_id.get("patterns.transition_outcome", [])
    if pattern_results:
        from app.shuttle_coach.feedback.patterns import derive_pattern_findings as _dpf
        findings.extend(_dpf(pattern_results, quality=quality))

    return findings


# ─── YAML Rule Evaluation ─────────────────────────────────────

_YAML_RULES_CACHE: list[dict] | None = None


def _load_yaml_rules(rules_path: Path | None = None) -> list[dict]:
    global _YAML_RULES_CACHE
    if _YAML_RULES_CACHE is not None:
        return _YAML_RULES_CACHE
    import yaml
    if rules_path is None:
        rules_path = Path(__file__).parent / "rules.yaml"
    with open(rules_path) as f:
        _YAML_RULES_CACHE = yaml.safe_load(f)["rules"]
    return _YAML_RULES_CACHE


def _get_nested(data: dict, path: str):
    """Dot-notation field access matching coach/engine.py semantics."""
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


def _compare(actual, operator: str, expected) -> bool:
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        if operator == "==":
            return str(actual) == str(expected)
        elif operator == "!=":
            return str(actual) != str(expected)
        return False
    if operator == "<":
        return actual < expected
    elif operator == ">":
        return actual > expected
    elif operator == "<=":
        return actual <= expected
    elif operator == ">=":
        return actual >= expected
    elif operator == "==":
        return actual == expected
    elif operator == "!=":
        return actual != expected
    return False


def _format_recommendation(template: str, analytics: dict) -> str:
    def replacer(match):
        field_path = match.group(1)
        value = _get_nested(analytics, field_path)
        if isinstance(value, (int, float)):
            fmt = match.group(2) if match.group(2) else ".1f"
            try:
                return format(value, fmt)
            except (ValueError, KeyError):
                return str(value)
        return str(value)
    return re.sub(r'\{([^}:]+)(?::([^}]+))?\}', replacer, template)


def _evaluate_rule(rule: dict, analytics: dict) -> bool:
    check = rule.get("check", {})
    if not check:
        return False

    min_shots_expr = check.get("min_shots")
    if min_shots_expr:
        parts = min_shots_expr.split()
        if len(parts) == 3:
            field_path, operator, value_str = parts
            try:
                threshold = float(value_str)
            except ValueError:
                return False
            field_value = _get_nested(analytics, field_path)
            if not _compare(field_value, operator, threshold):
                return False

    field_path = check.get("field")
    operator = check.get("operator")
    threshold = check.get("threshold", check.get("value"))

    if not field_path or not operator:
        return False

    value = _get_nested(analytics, field_path)
    return _compare(value, operator, threshold)


def _category_severity(category: str) -> float:
    return {"strength": 0.3, "weakness": 0.7, "insight": 0.5}.get(category, 0.5)


def evaluate_yaml_rules(analytics: dict, player_id: str) -> list[Finding]:
    """Evaluate YAML-defined rules against analytics data.

    Args:
        analytics: dict with keys: tactical, fitness, footwork, rally_stats,
                   court_analysis, opponent
        player_id: player identifier

    Returns:
        List of Finding objects from matching rules.
    """
    rules = _load_yaml_rules()
    findings = []

    # Build the nested dict the same way coach/engine.py does
    player_analytics = {
        "tactical": analytics.get("tactical", {}),
        "fitness": analytics.get("fitness", {}),
        "footwork": analytics.get("footwork", {}),
        "rally_stats": analytics.get("rally_stats", {}),
        "court_analysis": analytics.get("court_analysis", {}),
        "opponent": analytics.get("opponent", {}),
        "technique": analytics.get("technique", {}),
    }

    for rule in rules:
        try:
            if not _evaluate_rule(rule, player_analytics):
                continue

            rec = _format_recommendation(rule["recommendation"], player_analytics)
            rec_with_player = f"[{player_id}] {rec}"

            evidence = [f"player: {player_id}"]
            for cf in rule.get("context_fields", []):
                val = _get_nested(player_analytics, cf)
                if isinstance(val, float):
                    evidence.append(f"{cf}: {val:.3f}")

            findings.append(Finding(
                code=rule["name"],
                player_id=player_id,
                severity=_category_severity(rule.get("category", "insight")),
                headline=f"{rule['category'].title()}: {rule['name']}",
                detail=rec_with_player,
                evidence=evidence,
            ))
        except Exception:
            continue

    return findings
