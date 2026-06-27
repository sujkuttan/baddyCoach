#!/usr/bin/env python3
"""Clean NaN/Infinity from a report.json so the frontend can parse it.

Usage:
    python clean_report.py path/to/report.json
"""

import json
import sys
from pathlib import Path
import math


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_report.py <path/to/report.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text())
    cleaned = clean(data)
    path.write_text(json.dumps(cleaned, indent=2))
    print(f"Cleaned {path}")


if __name__ == "__main__":
    main()
