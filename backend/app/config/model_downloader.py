"""Model downloader — thin CLI over shared/models.py ensure_model().

Usage:
    python -m app.config.model_downloader          # verify + download all
    python -m app.config.model_downloader --force   # re-download all
    python -m app.config.model_downloader bst       # single model
"""

import sys
from pathlib import Path

from app.pipeline.shared.models import MODEL_REGISTRY, ensure_model

# Models essential for the backend pipeline (excludes colab-only variants)
BACKEND_MODELS = [
    "tracknet", "inpaintnet", "bst", "rtmpose", "hrnet", "court_kprcnn", "yolov8s",
]


def verify_all_models() -> dict[str, bool]:
    """Check which models exist locally."""
    status = {}
    for name in BACKEND_MODELS:
        entry = MODEL_REGISTRY.get(name)
        if entry:
            status[name] = entry[0].exists()
    return status


def download_all_models(force: bool = False) -> dict[str, Path | None]:
    """Ensure all backend models are available locally."""
    results = {}
    for name in BACKEND_MODELS:
        path = ensure_model(name, force=force)
        results[name] = path
    return results


def main():
    force = "--force" in sys.argv
    single = [a for a in sys.argv[1:] if not a.startswith("--")]

    if single:
        for name in single:
            if name not in MODEL_REGISTRY:
                print(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
                continue
            path = ensure_model(name, force=force)
            if path:
                print(f"{name}: OK ({path})")
            else:
                print(f"{name}: FAILED")
        return

    print("Verifying existing models...")
    status = verify_all_models()
    for name, exists in status.items():
        print(f"  {name}: {'OK' if exists else 'MISSING'}")

    missing = [n for n, e in status.items() if not e]
    if not missing:
        print("\nAll models present.")
        if not force:
            return

    if missing:
        print(f"\nDownloading {len(missing)} missing models...")
    elif force:
        print("\nRe-downloading all models (--force)...")

    results = download_all_models(force=force)
    print("\nResults:")
    for name, path in results.items():
        print(f"  {name}: {'OK' if path else 'FAILED'} ({path or 'N/A'})")

    any_failed = any(p is None for p in results.values())
    if any_failed:
        print("\nSome models failed to download.")
        sys.exit(1)
    else:
        print("\nAll models ready.")


if __name__ == "__main__":
    main()
