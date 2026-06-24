"""Findings derived from causal pattern metrics.

``derive_pattern_findings`` reads ``patterns.conditional_outcome`` and
``patterns.transition_outcome`` MetricResults and generates ``Finding``
objects with severity, headline, and detail.
"""

from app.config.settings import settings
from app.shuttle_coach.feedback.rules import Finding
from app.shuttle_coach.metrics.base import MetricResult


def _pretty_zone(zone: str) -> str:
    """Convert zone ID like 'rear_left' to readable 'rear left'."""
    return zone.replace("_", " ")


def _phrase(stroke: str, zone: str, pressed: bool) -> str:
    """Generate a human-readable headline for a pattern finding."""
    zone_pretty = _pretty_zone(zone)
    if pressed:
        return f"You lose points when playing {stroke} from {zone_pretty} under pressure."
    return f"Your {stroke} from {zone_pretty} is leaking points."


def derive_pattern_findings(results: list[MetricResult],
                            quality: dict | None = None) -> list[Finding]:
    """Convert pattern metric results into coachable Findings.

    Args:
        results: MetricResult list (should be filtered to include only
                 patterns.conditional_outcome and patterns.transition_outcome).
        quality: Optional data-quality dict.  Patterns are suppressed when
                 ``capability_trust.patterns`` is False.

    Returns:
        List of Finding objects (empty when untrusted or no patterns found).
    """
    quality = quality or {}
    capability_trust = quality.get("capability_trust", {})
    if not capability_trust.get("patterns", True):
        return []

    findings = []

    for r in results:
        if r.metric_id not in ("patterns.conditional_outcome",
                                "patterns.transition_outcome"):
            continue

        v = r.value
        if not isinstance(v, dict):
            continue

        n = v.get("n", 0)
        loss_rate = v.get("loss_rate", 0.0)
        baseline_loss = v.get("baseline_loss", 0.0)
        wilson_lb = v.get("wilson_loss_lb", 0.0)
        excess = loss_rate - baseline_loss

        if wilson_lb <= settings.pattern_loss_floor:
            continue
        if excess <= settings.pattern_excess_loss:
            continue

        stroke = v.get("stroke", "unknown")
        zone = v.get("zone", "unknown")
        pressed = v.get("pressed", False)

        if r.metric_id == "patterns.conditional_outcome":
            code = f"pattern::{stroke}::{zone}::{'pressed' if pressed else 'free'}"
            headline = _phrase(stroke, zone, pressed)
            detail = (
                f"When you play a {stroke} from the {_pretty_zone(zone)}"
                f"{' under pressure' if pressed else ''}, you lose the point "
                f"{loss_rate:.0%} of the time vs {baseline_loss:.0%} overall "
                f"(n={n})."
            )
        else:
            prev_stroke = v.get("prev_stroke", "unknown")
            code = f"pattern::transition::{prev_stroke}->{stroke}"
            headline = f"The {prev_stroke}→{stroke} transition loses points."
            detail = (
                f"When you follow a {prev_stroke} with a {stroke}, you lose the point "
                f"{loss_rate:.0%} of the time vs {baseline_loss:.0%} overall "
                f"(n={n})."
            )

        severity = round(min(1.0, excess * 2 * (r.confidence or 0.5)), 2)

        findings.append(Finding(
            code=code,
            player_id=r.player_id,
            severity=severity,
            headline=headline,
            detail=detail,
            evidence=[r.metric_id],
        ))

    return findings
