"""Dynamic drill selector — maps findings + trends to prescriptive drills.

Reads ``drills.yaml`` and uses glob-style target matching against
finding codes.  Drill difficulty level is determined by finding
severity and trend direction (declining → foundational, improving → advance).
"""

import re
from pathlib import Path
from typing import Any

import yaml

from app.shuttle_coach.feedback.rules import Finding


_DRILL_CACHE: list[dict] | None = None


def _load_drills(drills_path: Path | None = None) -> list[dict]:
    global _DRILL_CACHE
    if _DRILL_CACHE is not None:
        return _DRILL_CACHE
    if drills_path is None:
        drills_path = Path(__file__).parent / "drills.yaml"
    with open(drills_path) as f:
        _DRILL_CACHE = yaml.safe_load(f)["drills"]
    return _DRILL_CACHE


def _glob_match(code: str, pattern: str) -> bool:
    """Simple glob match supporting * and ? wildcards."""
    regex = "^" + re.escape(pattern).replace("\\*", ".*").replace("\\?", ".") + "$"
    return bool(re.match(regex, code))


def _catalog_matching(code: str, drills: list[dict]) -> list[dict]:
    """Find all drills whose targets match a finding code."""
    matches = []
    for drill in drills:
        for target in drill.get("targets", []):
            if _glob_match(code, target):
                matches.append(drill)
                break
    return matches


def _pick_level(severity: float, trend: dict | None) -> str:
    """Pick drill difficulty level based on severity + trend.

    Args:
        severity: 0..1 finding severity.
        trend: Optional trend dict from progress tracking.

    Returns:
        One of 'foundational', 'intermediate', 'advanced'.
    """
    if trend is None:
        trend = {}
    direction = trend.get("direction", "stable")

    # Declining or high severity → remediate at foundational level
    if direction == "declining" or severity > 0.7:
        return "foundational"
    # Improving or low severity → advance
    if direction == "improving" or severity < 0.4:
        return "advanced"
    return "intermediate"


def _dedup_by_drill(drills: list[dict]) -> list[dict]:
    """Remove duplicate drill_ids, keeping the highest severity entry."""
    seen = {}
    for d in drills:
        did = d.get("drill_id", "")
        if did not in seen or d.get("_raw_severity", 0) > seen.get("_raw_severity", 0):
            seen[did] = d
    return list(seen.values())


def select_drills(findings: list[Finding],
                  trends: dict | None = None,
                  quality: dict | None = None,
                  top_n: int = 3) -> list[dict]:
    """Select the most relevant drills for a set of findings.

    Args:
        findings: Prioritised list of Finding objects.
        trends: Optional dict of trend results (keyed by finding code or metric).
        quality: Optional data-quality dict (not yet used for drill selection).
        top_n: Maximum number of drills to return.

    Returns:
        List of structured drill dicts with drill_id, name, focus, level,
        dosage, success_criteria, rationale, linked_finding, trend.
    """
    quality = quality or {}
    drills = _load_drills()
    chosen = []

    for finding in sorted(findings, key=lambda f: f.severity, reverse=True):
        matched = _catalog_matching(finding.code, drills)
        if not matched:
            continue

        trend_data = (trends or {}).get(finding.code)
        level = _pick_level(finding.severity, trend_data)
        drill = matched[0]  # Take first match per finding
        level_data = drill.get("levels", {}).get(level, {})

        chosen.append({
            "drill_id": drill["id"],
            "name": drill["id"].replace("_", " ").title(),
            "focus": drill.get("focus", ""),
            "level": level,
            "dosage": level_data.get("dosage", ""),
            "success_criteria": level_data.get("success", ""),
            "rationale": finding.detail,
            "linked_finding": finding.code,
            "trend": trend_data,
            "_raw_severity": finding.severity,
        })

    chosen = _dedup_by_drill(chosen)
    chosen.sort(key=lambda d: d.get("_raw_severity", 0), reverse=True)
    chosen = [{k: v for k, v in d.items() if not k.startswith("_")} for d in chosen]

    return chosen[:top_n]


def format_drill_flat(drill: dict) -> str:
    """Format a structured drill as a flat string for back-compat display."""
    parts = [drill.get("name", "Drill")]
    if drill.get("dosage"):
        parts.append(f"({drill['dosage']})")
    if drill.get("rationale"):
        parts.append(f"— {drill['rationale']}")
    return " ".join(parts)
