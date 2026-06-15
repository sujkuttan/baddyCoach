from pathlib import Path
from typing import Any

import yaml


class CoachEngine:
    """Rule-based coach engine with dot-notation field paths.
    
    Rules use dot-notation to access nested analytics fields:
        tactical.shot_distribution.smash
        fitness.fatigue_trend
        footwork.avg_recovery
    """
    
    def __init__(self, rules_path: Path | None = None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules.yaml"
        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)["rules"]

    def generate(self, analytics: dict[str, Any], player_id: str) -> dict[str, Any]:
        """Generate coaching recommendations for a player."""
        strengths = []
        weaknesses = []
        improvements = []
        drills = []
        evidence = []

        player_analytics = {
            "tactical": analytics.get("tactical_analytics", {}).get(player_id, {}),
            "fitness": analytics.get("fitness_analytics", {}).get(player_id, {}),
            "footwork": analytics.get("footwork_analytics", {}).get(player_id, {}),
        }
        
        tactical = player_analytics["tactical"]
        if tactical:
            shot_dist = tactical.get("shot_distribution", {})
            tactical["max_shot_percentage"] = max(shot_dist.values()) if shot_dist else 0

        for rule in self.rules:
            if self._evaluate_rule(rule, player_analytics):
                entry = {
                    "finding": rule["recommendation"],
                    "metrics": self._extract_metrics(rule, player_analytics),
                }
                evidence.append(entry)

                if rule["category"] == "strength":
                    strengths.append(rule["recommendation"])
                elif rule["category"] == "weakness":
                    weaknesses.append(rule["recommendation"])
                    improvements.append(rule["recommendation"])
                    drills.append(rule.get("drill", ""))

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3],
            "recommended_drills": drills[:3],
            "evidence": evidence,
        }
    
    def _evaluate_rule(self, rule: dict, analytics: dict) -> bool:
        """Evaluate a rule against analytics data."""
        check = rule.get("check", {})
        if not check:
            return False
        
        min_shots_expr = check.get("min_shots")
        if min_shots_expr:
            if not self._evaluate_condition(min_shots_expr, analytics):
                return False
        
        field_path = check.get("field")
        operator = check.get("operator")
        threshold = check.get("threshold", check.get("value"))
        
        if not field_path or not operator:
            return False
        
        value = self._get_nested(analytics, field_path)
        
        return self._compare(value, operator, threshold)
    
    def _evaluate_condition(self, expr: str, analytics: dict) -> bool:
        """Evaluate a condition expression like 'tactical.total_shots >= 10'."""
        parts = expr.split()
        if len(parts) != 3:
            return False
        
        field_path, operator, value_str = parts
        
        try:
            value = float(value_str)
        except ValueError:
            return False
        
        field_value = self._get_nested(analytics, field_path)
        return self._compare(field_value, operator, value)
    
    def _compare(self, actual, operator: str, expected) -> bool:
        """Compare actual value against expected using operator."""
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
    
    def _get_nested(self, data: dict, path: str):
        """Extract value from nested dict using dot notation."""
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
    
    def _extract_metrics(self, rule: dict, analytics: dict) -> list[str]:
        """Extract human-readable metrics for evidence."""
        metrics = []
        check = rule.get("check", {})
        
        if "field" in check:
            value = self._get_nested(analytics, check["field"])
            field_name = check["field"].split(".")[-1]
            if isinstance(value, float):
                metrics.append(f"{field_name}: {value:.2f}")
            else:
                metrics.append(f"{field_name}: {value}")
        
        total_shots = self._get_nested(analytics, "tactical.total_shots")
        if total_shots > 0:
            metrics.append(f"total shots: {total_shots}")
        
        fatigue = self._get_nested(analytics, "fitness.fatigue_trend")
        if fatigue and fatigue != "unknown":
            metrics.append(f"fatigue trend: {fatigue}")
        
        return metrics if metrics else ["data available"]
