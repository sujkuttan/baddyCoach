from dataclasses import dataclass, field

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


def derive_findings(results_by_id: dict[str, list[MetricResult]]) -> list[Finding]:
    findings: list[Finding] = []
    for player_id, results in results_by_id.items():
        findings.extend(_check_slow_recovery(results))
        findings.extend(_check_weak_shots(results))
        findings.extend(_check_high_unforced(results))
    return findings
