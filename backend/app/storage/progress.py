"""Cross-session progress tracking for player history.

Stores per-player analytics snapshots keyed by a player identifier.
Tracks trends across multiple video uploads/analyses.
"""

import json
from pathlib import Path
from typing import Any

HISTORY_DIR = Path("data/player_history")


def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_player_session(player_id: str, job_id: str, analytics: dict[str, Any]):
    """Save a session snapshot for a player."""
    _ensure_dir()
    player_file = HISTORY_DIR / f"{player_id}.json"
    history = []
    if player_file.exists():
        history = json.loads(player_file.read_text())
    history.append({
        "job_id": job_id,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "analytics": analytics,
    })
    # Keep last 20 sessions
    history = history[-20:]
    player_file.write_text(json.dumps(history, indent=2, default=str))


def get_player_history(player_id: str) -> list[dict]:
    """Get session history for a player."""
    player_file = HISTORY_DIR / f"{player_id}.json"
    if player_file.exists():
        return json.loads(player_file.read_text())
    return []


def compute_trends(player_id: str, metric_key: str) -> dict[str, Any]:
    """Compute trend direction for a specific metric across sessions."""
    history = get_player_history(player_id)
    if len(history) < 2:
        return {"direction": "insufficient_data", "values": []}

    values = []
    for session in history:
        val = session.get("analytics", {}).get(metric_key)
        if val is not None:
            values.append(val)

    if len(values) < 2:
        return {"direction": "insufficient_data", "values": values}

    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]
    avg_first = sum(first_half) / len(first_half) if first_half else 0
    avg_second = sum(second_half) / len(second_half) if second_half else 0

    if avg_second > avg_first * 1.1:
        direction = "improving"
    elif avg_second < avg_first * 0.9:
        direction = "declining"
    else:
        direction = "stable"

    return {"direction": direction, "values": values, "avg_first": avg_first, "avg_second": avg_second}
