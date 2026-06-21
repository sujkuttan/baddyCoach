import re
from pathlib import Path
from typing import Any
from collections import Counter

import yaml
import numpy as np


class CoachEngine:
    """Rule-based coach engine with dot-notation field paths."""

    def __init__(self, rules_path: Path | None = None):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules.yaml"
        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)["rules"]

    def generate(self, analytics: dict[str, Any], player_id: str) -> dict[str, Any]:
        strengths = []
        weaknesses = []
        improvements = []
        drills = []
        evidence = []

        tactical = analytics.get("tactical_analytics", {}).get(player_id, {})
        fitness = analytics.get("fitness_analytics", {}).get(player_id, {})
        footwork = analytics.get("footwork_analytics", {}).get(player_id, {})

        if tactical:
            shot_dist = tactical.get("shot_distribution", {})
            tactical["max_shot_percentage"] = max(shot_dist.values()) if shot_dist else 0

        player_analytics = {
            "tactical": tactical,
            "fitness": fitness,
            "footwork": footwork,
        }

        if fitness:
            fitness["intensity"] = fitness.get("rally_intensity", 0)
            fitness["peak"] = fitness.get("peak_intensity", 0)
            fitness["distance"] = fitness.get("total_distance", 0)
        if footwork:
            footwork["recovery"] = footwork.get("avg_recovery", 0)

        rally_stats = self._compute_rally_stats(analytics, player_id)
        player_analytics["rally_stats"] = rally_stats

        court_analysis = self._compute_court_analysis(analytics, player_id)
        player_analytics["court_analysis"] = court_analysis

        opponent_data = self._compute_opponent_data(analytics, player_id)
        player_analytics["opponent"] = opponent_data

        for rule in self.rules:
            try:
                if self._evaluate_rule(rule, player_analytics):
                    rec = self._format_recommendation(rule["recommendation"], player_analytics)
                    rec_with_player = f"[{player_id}] {rec}"

                    entry = {
                        "finding": rec_with_player,
                        "metrics": [f"player: {player_id}",
                                    f"total shots: {tactical.get('total_shots', 0)}"],
                    }
                    for cf in rule.get("context_fields", []):
                        val = self._get_nested(player_analytics, cf)
                        if isinstance(val, float):
                            entry["metrics"].append(f"{cf}: {val:.3f}")
                    evidence.append(entry)

                    if rule["category"] == "strength":
                        if rec_with_player not in strengths:
                            strengths.append(rec_with_player)
                    elif rule["category"] == "weakness":
                        if rec_with_player not in weaknesses:
                            weaknesses.append(rec_with_player)
                            improvements.append(rec_with_player)
                            drills.append(rule.get("drill", ""))
                    elif rule["category"] == "insight":
                        if rec_with_player not in weaknesses:
                            weaknesses.append(rec_with_player)
            except Exception:
                continue

        rally_stats_dict = rally_stats if rally_stats.get("avg_length", 0) > 0 else None

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "top_3_improvements": improvements[:3],
            "recommended_drills": drills[:3],
            "evidence": evidence,
            "rally_stats": rally_stats_dict,
        }

    def _compute_rally_stats(self, analytics: dict, player_id: str) -> dict:
        rallies_df = analytics.get("_rallies_df")
        shots_df = analytics.get("_shots_df")
        if rallies_df is None or shots_df is None:
            return {"avg_length": 0, "max_length": 0, "min_length": 0,
                    "first_shot_win_rate": 0, "long_rally_pct": 0}

        if hasattr(rallies_df, 'iterrows'):
            rally_lengths = []
            first_shot_wins = 0
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

            total_rallies = len(rally_lengths)
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

    def _compute_court_analysis(self, analytics: dict, player_id: str) -> dict:
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

    def _compute_opponent_data(self, analytics: dict, player_id: str) -> dict:
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
        total = opp.get("total_shots", 0)
        return {
            "smash_pct": dist.get("smash", 0),
            "net_pct": dist.get("net_shot", 0),
            "clear_pct": dist.get("clear", 0),
            "total_shots": total,
        }

    @staticmethod
    def _format_recommendation(template: str, analytics: dict) -> str:
        def replacer(match):
            field_path = match.group(1)
            value = CoachEngine._get_nested(analytics, field_path)
            if isinstance(value, (int, float)):
                fmt = match.group(2) if match.group(2) else ".1f"
                try:
                    return format(value, fmt)
                except (ValueError, KeyError):
                    return str(value)
            return str(value)

        return re.sub(r'\{([^}:]+)(?::([^}]+))?\}', replacer, template)

    def _evaluate_rule(self, rule: dict, analytics: dict) -> bool:
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

    @staticmethod
    def _get_nested(data: dict, path: str):
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
        for cf in rule.get("context_fields", []):
            val = self._get_nested(analytics, cf)
            if isinstance(val, float):
                metrics.append(f"{cf}: {val:.3f}")
        return metrics if metrics else ["data available"]
