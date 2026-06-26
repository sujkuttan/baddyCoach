import json
from pathlib import Path

import pandas as pd


class ArtifactStore:
    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def set(self, key: str, data: dict) -> Path:
        path = self.job_dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def get(self, key: str) -> dict | None:
        path = self.job_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set_parquet(self, key: str, df: pd.DataFrame) -> Path:
        path = self.job_dir / f"{key}.parquet"
        df.to_parquet(path, index=False)
        return path

    def get_parquet(self, key: str) -> pd.DataFrame | None:
        path = self.job_dir / f"{key}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, key: str) -> bool:
        json_path = self.job_dir / f"{key}.json"
        parquet_path = self.job_dir / f"{key}.parquet"
        return json_path.exists() or parquet_path.exists()

    def path(self, key: str) -> Path:
        json_path = self.job_dir / f"{key}.json"
        if json_path.exists():
            return json_path
        return self.job_dir / f"{key}.parquet"
