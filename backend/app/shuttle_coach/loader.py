from pathlib import Path

import pandas as pd

COLUMN_ALIASES = {
    "shots": {
        "stroke_type": "shot_type",
        "stroke_confidence": "shot_conf",
        "frame": "hit_frame",
    },
    "rallies": {
        "shot_count": None,
    },
    "player_detections": {
        "side": "player_id",  # Colab uses 'side' (near/far), backend uses 'player_id'
    },
}

REQUIRED = {
    "rallies": ["rally_id"],
    "shots": ["rally_id", "player_id"],
    "hits": ["frame"],  # rally_id optional (Colab doesn't always have it)
    "shuttle": ["frame"],
    "pose": ["frame", "player_id"],
}

OPTIONAL_TABLES = {"pose"}


def _convert_backend_players_json(base: Path) -> pd.DataFrame | None:
    """Convert backend's players.json to the expected player_detections format."""
    players_json = base / "players.json"
    if not players_json.exists():
        return None
    import json
    with open(players_json) as f:
        data = json.load(f)
    rows = []
    for player in data.get("players", []):
        pid = player["id"]
        for det in player.get("detections", []):
            rows.append({
                "frame": det["frame"],
                "player_id": pid,
                "bbox": det["bbox"],
                "confidence": det.get("confidence", 0.5),
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_match(data_dir: Path) -> dict[str, pd.DataFrame]:
    if (data_dir / "debug").is_dir():
        base = data_dir / "debug"
    else:
        base = data_dir

    tables: dict[str, pd.DataFrame] = {}
    for pq in base.glob("*.parquet"):
        name = pq.stem
        tables[name] = pd.read_parquet(pq)

    # Try to convert backend players.json to player_detections format
    if "player_detections" not in tables:
        pdf = _convert_backend_players_json(base)
        if pdf is not None:
            tables["player_detections"] = pdf

    for table_name, aliases in COLUMN_ALIASES.items():
        if table_name not in tables:
            continue
        df = tables[table_name]
        rename_map = {}
        for old, new in aliases.items():
            if new is None:
                if old in df.columns:
                    df = df.drop(columns=[old])
                continue
            if old in df.columns:
                rename_map[old] = new
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Convert side labels to player_id format
        if table_name == "player_detections" and "player_id" in df.columns:
            side_map = {"near": "player_1", "far": "player_2"}
            df["player_id"] = df["player_id"].map(side_map).fillna(df["player_id"])
        
        tables[table_name] = df

    missing_tables = set(REQUIRED) - set(tables) - OPTIONAL_TABLES
    if missing_tables:
        raise ValueError(f"Missing required tables: {sorted(missing_tables)}")

    for table_name, cols in REQUIRED.items():
        if table_name not in tables:
            continue
        df = tables[table_name]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Table '{table_name}' missing required columns: {sorted(missing)}")

    return tables


def capabilities(tables: dict[str, pd.DataFrame]) -> list[str]:
    caps = ["shots", "errors"]

    if "player_detections" in tables:
        df = tables["player_detections"]
        if "court_x" in df.columns and "court_y" in df.columns:
            caps.append("movement")

    if "shuttle" in tables:
        df = tables["shuttle"]
        if "court_x" in df.columns and "court_y" in df.columns:
            caps.append("tactical")
    elif "hits" in tables:
        df = tables["hits"]
        if "court_x" in df.columns and "court_y" in df.columns:
            caps.append("tactical")

    if "pose" in tables:
        caps.append("technique")

    return caps
