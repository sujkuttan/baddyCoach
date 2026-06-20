from dataclasses import asdict

from app.shuttle_coach.feedback.rules import Finding


def render_report(findings: list[Finding], top_k: int = 5) -> str:
    lines = ["# Coaching Report", ""]

    if not findings:
        lines.append("No findings.")
        return "\n".join(lines)

    priorities = findings[:top_k]
    lines.append("## Priorities")
    for i, f in enumerate(priorities, 1):
        evidence_str = ", ".join(f.evidence) if f.evidence else "n/a"
        lines.append(f"{i}. **{f.headline}** — {f.detail} _(evidence: {evidence_str})_")
    lines.append("")

    lines.append("## All findings")
    for f in findings:
        lines.append(f"- [{f.severity:.2f}] {f.headline}: {f.detail}")

    return "\n".join(lines)


def render_report_json(findings: list[Finding], player_ids: list[str], capabilities: set[str]) -> dict:
    return {
        "findings": [asdict(f) for f in findings],
        "player_ids": player_ids,
        "capabilities": sorted(capabilities),
    }
