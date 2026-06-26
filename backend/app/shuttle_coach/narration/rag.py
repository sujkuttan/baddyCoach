from typing import Any


def retrieve_relevant_metrics(
    question: str,
    metrics: list[dict[str, Any]],
    k: int = 12,
) -> list[dict[str, Any]]:
    """Retrieve metrics relevant to a question using keyword matching."""
    question_tokens = set(question.lower().split())

    scored: list[tuple[float, dict[str, Any]]] = []
    for m in metrics:
        score = 0.0
        metric_id = m.get("metric_id", "")
        for part in metric_id.split("."):
            for token in part.split("_"):
                if token and token in question_tokens:
                    score += 2.0
        context = m.get("context", {})
        if isinstance(context, dict):
            for val in context.values():
                if isinstance(val, str):
                    for token in val.lower().split():
                        if token in question_tokens:
                            score += 1.0
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]