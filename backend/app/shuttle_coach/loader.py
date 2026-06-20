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
}

REQUIRED = {
    "rallies": ["rally_id"],
    "shots": ["rally_id", "player_id"],
    "hits": ["rally_id", "frame"],
    "shuttle": ["frame"],
    "player_detections": ["frame", "player_id"],
    "pose": ["frame", "player_id"],
}

OPTIONAL_TABLES = {"pose"}


def load_match(data_dir: Path) -> dict[str, pd.DataFrame]:
    if (data_dir / "debug").is_dir():
        base = data_dir / "debug"
    else:
        base = data_dir

    tables: dict[str, pd.DataFrame] = {}
    for pq in base.glob("*.parquet"):
        name = pq.stem
        tables[name] = pd.read_parquet(pq)

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
            tables[table_name] = df.rename(columns=rename_map)

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
