import re
from typing import Any

SYSTEM_PROMPT = """You are a badminton coaching assistant. You may ONLY use the metrics
provided in the context. Every claim must cite the metric_id(s) it relies
on in square brackets, e.g. [movement.recovery_time]. If the metrics do
not support an answer, say so. Do not invent numbers."""


def format_metrics_for_rag(metrics: list[dict[str, Any]], question: str) -> str:
    from app.shuttle_coach.narration.rag import retrieve_relevant_metrics

    relevant = retrieve_relevant_metrics(question, metrics)
    lines = []
    for m in relevant:
        mid = m.get("metric_id", "?")
        val = m.get("value", "?")
        unit = m.get("unit", "")
        pid = m.get("player_id", "")
        lines.append(f"- {mid} (player={pid}): {val} {unit}")
    return "\n".join(lines)


def answer(question: str, metrics: list[dict[str, Any]], api_key: str) -> str:
    """Generate a grounded answer to a coaching question."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    context = format_metrics_for_rag(metrics, question)

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\nMETRICS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )

    text = response.text
    enforce_citations(text, metrics)
    return text


def enforce_citations(text: str, metrics: list[dict[str, Any]]) -> None:
    """Validate that every sentence has a valid citation."""
    valid = {m["metric_id"] for m in metrics}

    cited = set(re.findall(r"\[([a-z_]+\.[a-z_]+)\]", text))
    unknown = cited - valid
    if unknown:
        raise ValueError(f"Narration cited unknown metrics: {unknown}")

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 6]
    uncited = [
        s for s in sentences if not re.search(r"\[[a-z_]+\.[a-z_]+\]", s)
    ]
    if uncited:
        raise ValueError(f"Ungrounded sentences: {uncited}")