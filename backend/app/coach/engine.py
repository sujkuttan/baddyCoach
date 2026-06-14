from pathlib import Path
from typing import Any

import yaml


class CoachEngine:
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

        fitness = analytics.get("fitness_analytics", {}).get(player_id, {})
        tactical = analytics.get("tactical_analytics", {}).get(player_id, {})
        footwork = analytics.get("footwork_analytics", {}).get(player_id, {})

        shot_dist = tactical.get("shot_distribution", {})
        total_shots = tactical.get("total_shots", 0)
        avg_recovery = footwork.get("avg_recovery", 0)
        fatigue_trend = fitness.get("fatigue_trend", "unknown")

        max_shot_pct = max(shot_dist.values()) if shot_dist else 0

        for rule in self.rules:
            triggered = False

            if rule["name"] == "smash_efficiency":
                smash_pct = shot_dist.get("smash", 0)
                if total_shots >= rule.get("min_shots", 0) and smash_pct < 0.3:
                    triggered = True

            elif rule["name"] == "recovery_speed":
                if avg_recovery > float(rule["condition"].split("> ")[1]):
                    triggered = True

            elif rule["name"] == "shot_variety":
                if total_shots >= rule.get("min_shots", 0) and max_shot_pct > 0.5:
                    triggered = True

            elif rule["name"] == "fatigue_management":
                if fatigue_trend == "declining":
                    triggered = True

            elif rule["name"] == "net_play_strength":
                net_pct = shot_dist.get("net_shot", 0)
                if total_shots >= rule.get("min_shots", 0) and net_pct > 0.2:
                    triggered = True

            elif rule["name"] == "clear_usage":
                clear_pct = shot_dist.get("clear", 0)
                if total_shots >= rule.get("min_shots", 0) and clear_pct > 0.35:
                    triggered = True

            if triggered:
                metrics_list = []
                if avg_recovery > 0:
                    metrics_list.append(f"avg recovery: {avg_recovery:.1f}s")
                if total_shots > 0:
                    metrics_list.append(f"total shots: {total_shots}")
                if fatigue_trend != "unknown":
                    metrics_list.append(f"fatigue trend: {fatigue_trend}")

                evidence_item = {
                    "finding": rule["recommendation"],
                    "metrics": metrics_list if metrics_list else ["data available"],
                }
                evidence.append(evidence_item)

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
