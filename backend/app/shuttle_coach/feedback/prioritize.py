from app.shuttle_coach.feedback.rules import Finding


def prioritize_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: f.severity, reverse=True)
