from collections import defaultdict
from pathlib import Path
from typing import Any

from app.shuttle_coach.loader import load_match, capabilities
from app.shuttle_coach.events import MatchModel
from app.shuttle_coach.metrics import run_metrics
from app.shuttle_coach.metrics.base import MetricResult
from app.shuttle_coach.feedback import derive_findings, prioritize_findings
from app.shuttle_coach.feedback.report import render_report, render_report_json


def analyze(data_dir: str) -> dict[str, Any]:
    tables = load_match(Path(data_dir))
    caps = capabilities(tables)
    model = MatchModel.from_tables(tables)

    results = run_metrics(model, set(caps))

    results_by_id: dict[str, list[MetricResult]] = defaultdict(list)
    for r in results:
        results_by_id[r.metric_id].append(r)

    findings = derive_findings(results_by_id)
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
    raise NotImplementedError(
        "Narration module (T19) not yet implemented. "
        "app.shuttle_coach.narration.gemini will provide answer()."
    )
