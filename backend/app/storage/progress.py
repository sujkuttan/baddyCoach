"""Cross-session progress tracking for player history.

Stores per-player analytics snapshots keyed by a ``player_key`` (or job-scoped
fallback).  Supports trend computation, headline detection, and sparklines
for the progress API and UI.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

HISTORY_DIR = Path("data/player_history")
MAX_SESSIONS = 50


def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _player_path(player_key: str) -> Path:
    return HISTORY_DIR / f"{player_key}.json"


# ═══════════════════════════════════════════════════════════════════════
# Session snapshot schema
# ═══════════════════════════════════════════════════════════════════════

def make_snapshot(job_id: str, analytics: dict,
                  data_quality: dict | None = None) -> dict:
    """Build a structured SessionSnapshot from pipeline analytics.

    Args:
        job_id: Pipeline job identifier.
        analytics: Raw analytics dict (tactical, fitness, footwork, etc.).
        data_quality: Optional quality dict from DataQualityStage.

    Returns:
        A dict that conforms to the SessionSnapshot schema.
    """
    snapshot = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "tactical": {},
        "fitness": {},
        "footwork": {},
        "court": {},
        "technique": {},
        "data_quality": data_quality or {"tier": "unknown", "quality_score": 0.0},
    }

    # Extract structured fields
    for pid, data in (analytics.get("tactical_analytics") or {}).items():
        snapshot["tactical"][pid] = {
            "shot_distribution": data.get("shot_distribution", {}),
            "total_shots": data.get("total_shots", 0),
            "common_patterns": data.get("common_patterns", []),
            "unique_strokes": data.get("unique_strokes", []),
        }

    for pid, data in (analytics.get("fitness_analytics") or {}).items():
        snapshot["fitness"][pid] = {
            "rally_intensity": data.get("rally_intensity", 0),
            "peak_intensity": data.get("peak_intensity", 0),
            "fatigue_trend": data.get("fatigue_trend", "insufficient_data"),
            "total_distance": data.get("total_distance", 0),
            "avg_recovery": data.get("avg_recovery", 0),
            "late_rally_fatigue": data.get("late_rally_fatigue", 0),
        }

    for pid, data in (analytics.get("footwork_analytics") or {}).items():
        snapshot["footwork"][pid] = {
            "distance_covered": data.get("distance_covered", 0),
            "avg_recovery": data.get("avg_recovery", 0),
        }

    for pid, data in (analytics.get("technical_analytics") or {}).items():
        tech = {}
        for stroke, sd in data.items():
            if isinstance(sd, dict):
                tech[stroke] = {
                    "avg_score": sd.get("avg_score", 0),
                    "shot_count": sd.get("shot_count", 0),
                }
        snapshot["technique"][pid] = tech

    return snapshot


def save_player_session(player_key: str, job_id: str,
                        analytics: dict,
                        data_quality: dict | None = None):
    """Save a session snapshot for a player."""
    _ensure_dir()
    path = _player_path(player_key)
    history = []
    if path.exists():
        history = json.loads(path.read_text())

    snapshot = make_snapshot(job_id, analytics, data_quality)
    history.append(snapshot)
    history = history[-MAX_SESSIONS:]
    path.write_text(json.dumps(history, indent=2, default=str))


def get_player_history(player_key: str) -> list[dict]:
    """Get session history for a player."""
    path = _player_path(player_key)
    if path.exists():
        return json.loads(path.read_text())
    return []


# ═══════════════════════════════════════════════════════════════════════
# Trend engine
# ═══════════════════════════════════════════════════════════════════════

_KEY_PATH_MAP = {
    "rally_intensity":        ("fitness", "rally_intensity"),
    "peak_intensity":         ("fitness", "peak_intensity"),
    "total_distance":         ("fitness", "total_distance"),
    "avg_recovery":           ("footwork", "avg_recovery"),
    "late_rally_fatigue":     ("fitness", "late_rally_fatigue"),
    "overall_technique":      ("technique", "overall"),
    "shot_count":             ("tactical", "total_shots"),
}


def _resolve(snapshot: dict, key_path: str, player_id: str = "player_1") -> float | None:
    """Resolve a dot-path key like ``fitness.rally_intensity``.

    Tries each player_id in the snapshot section, falling back to player_1.
    Returns None when not found.
    """
    parts = key_path.split(".")
    section = parts[0]
    key = parts[1] if len(parts) > 1 else parts[0]

    data = snapshot.get(section, {})
    # Try the specified player_id first, then fall through
    for pid in (player_id, "player_1"):
        if pid in data and isinstance(data[pid], dict):
            val = data[pid].get(key)
            if val is not None:
                return float(val)
    return None


def compute_metric_trend(player_key: str, key_path: str,
                         window: int = 5,
                         player_id: str = "player_1",
                         min_sessions: int = 2) -> dict:
    """Compute trend direction and sparkline for a metric.

    Args:
        player_key: Player identifier.
        key_path: Dot-path into the snapshot (e.g. ``fitness.rally_intensity``).
        window: Number of recent sessions to consider.
        player_id: Player sub-key within each session.
        min_sessions: Minimum sessions required for a trend.

    Returns:
        dict with keys: direction, slope, pct_change, values, n_sessions,
        first_value, last_value, sparkline, key_path.
    """
    history = get_player_history(player_key)
    if len(history) < min_sessions:
        return {"direction": "insufficient_data", "values": [],
                "n_sessions": len(history), "key_path": key_path}

    recent = history[-window:]
    # Filter out low-quality sessions
    valid = [s for s in recent
             if s.get("data_quality", {}).get("tier") != "low"]
    if len(valid) < min_sessions:
        valid = recent[-min_sessions:]  # fall back to raw

    values = []
    for s in valid:
        v = _resolve(s, key_path, player_id)
        if v is not None:
            values.append(v)

    if len(values) < min_sessions:
        return {"direction": "insufficient_data", "values": values,
                "n_sessions": len(values), "key_path": key_path}

    import numpy as np
    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0

    first_val = values[0]
    last_val = values[-1]
    avg_val = float(np.mean(values))

    pct_change = ((last_val - first_val) / max(abs(first_val), 1e-6)
                  if first_val != 0 else 0.0)

    if abs(pct_change) < 0.05:
        direction = "stable"
    elif pct_change > 0:
        direction = "improving" if _is_positive_metric(key_path) else "declining"
    else:
        direction = "declining" if _is_positive_metric(key_path) else "improving"

    return {
        "direction": direction,
        "slope": round(slope, 4),
        "pct_change": round(pct_change, 4),
        "values": [round(v, 4) for v in values],
        "n_sessions": len(values),
        "first_value": round(first_val, 4),
        "last_value": round(last_val, 4),
        "sparkline": [round(v, 4) for v in values],
        "key_path": key_path,
    }


def _is_positive_metric(key_path: str) -> bool:
    """Whether a higher value for this metric is considered 'improving'."""
    higher_is_better = {
        "rally_intensity", "peak_intensity", "total_distance",
        "shot_count", "overall_technique", "unique_strokes_count",
    }
    key = key_path.split(".")[-1]
    return key in higher_is_better


def compare_last_n(player_key: str, n: int = 5,
                   player_id: str = "player_1") -> list[dict]:
    """Compare the last n sessions and return headline movements.

    Returns a list of dicts sorted by absolute pct_change descending,
    each with metric, pct_change, direction, detail.
    """
    tracked_metrics = [
        "fitness.rally_intensity",
        "fitness.peak_intensity",
        "fitness.total_distance",
        "footwork.avg_recovery",
        "fitness.late_rally_fatigue",
        "tactical.total_shots",
    ]

    headlines = []
    for kp in tracked_metrics:
        trend = compute_metric_trend(player_key, kp, window=n,
                                     player_id=player_id)
        if trend["direction"] == "insufficient_data":
            continue
        label = kp.split(".")[-1].replace("_", " ").title()
        pct = trend["pct_change"]
        headlines.append({
            "metric": kp,
            "label": label,
            "pct_change": pct,
            "direction": trend["direction"],
            "detail": f"{label} {'up' if pct > 0 else 'down'} {abs(pct):.0%} over last {trend['n_sessions']} sessions",
            "sparkline": trend.get("sparkline", []),
        })

    headlines.sort(key=lambda h: abs(h["pct_change"]), reverse=True)
    return headlines
