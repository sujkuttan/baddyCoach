"""GPU-aware batch size configuration.

Detects available VRAM and returns optimal batch sizes for each pipeline stage.
Tiers are calibrated against T4 (15.5GB), A100 (40GB/80GB), and CPU fallback.
"""


_TIER_TABLE = [
    # (min_vram_gb, config)
    # T4 (15.5 GB) tuned: YOLO Conv2d spikes 750 MiB per sub-batch;
    # fragmentation accumulates ~6 batches → OOM with larger chunks.
    (12, {"yolo_chunk": 200,  "yolo_batch": 16, "tracknet_chunk": 16,  "rtmpose_chunk": 128, "bst_batch": 128}),
    (6,  {"yolo_chunk": 100,  "yolo_batch": 8,  "tracknet_chunk": 8,   "rtmpose_chunk": 64,  "bst_batch": 64}),
    (2,  {"yolo_chunk": 50,   "yolo_batch": 4,  "tracknet_chunk": 4,   "rtmpose_chunk": 32,  "bst_batch": 32}),
    (0,  {"yolo_chunk": 25,   "yolo_batch": 2,  "tracknet_chunk": 2,   "rtmpose_chunk": 16,  "bst_batch": 16}),
]

_CPU_CONFIG = {"yolo_chunk": 100, "yolo_batch": 8, "tracknet_chunk": 8, "rtmpose_chunk": 32, "bst_batch": 16}


def get_gpu_batch_config(device: str = "cpu") -> dict:
    """Return batch-size config tuned for the available GPU.

    Returns a dict with keys:
        yolo_chunk, yolo_batch, tracknet_chunk, rtmpose_chunk, bst_batch

    Explicit overrides from ``settings`` (env-configurable) are applied on top
    of the auto-detected tier in every branch, so users on multi-GPU hosts
    (e.g. 2×T4, where auto-detect only sees one GPU's VRAM) or wanting to push
    past the conservative tier defaults can raise batch sizes directly.
    """
    if "cuda" not in device.lower():
        cfg = dict(_CPU_CONFIG)
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                cfg = dict(_CPU_CONFIG)
            else:
                vram_bytes = torch.cuda.get_device_properties(0).total_mem
                vram_gb = vram_bytes / (1024 ** 3)
                for min_gb, tier in _TIER_TABLE:
                    if vram_gb >= min_gb:
                        cfg = dict(tier)
                        break
                else:
                    cfg = dict(_CPU_CONFIG)
        except Exception:
            cfg = dict(_CPU_CONFIG)

    # Apply explicit overrides from settings (env-configurable).
    try:
        from app.config.settings import settings
        overrides = {
            k: v
            for k, v in {
                "yolo_batch": settings.yolo_batch_size,
                "tracknet_chunk": settings.tracknet_batch_size,
                "rtmpose_chunk": settings.rtmpose_batch_size,
                "bst_batch": settings.bst_batch_size,
            }.items()
            if v is not None
        }
        if overrides:
            cfg.update(overrides)
    except Exception:
        pass

    return cfg


def get_gpu_tier(device: str = "cpu") -> str:
    """Return a human-readable tier label."""
    if "cuda" not in device.lower():
        return "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except Exception:
        return "cpu"
    if vram_gb >= 12:
        return "large"
    elif vram_gb >= 6:
        return "medium"
    elif vram_gb >= 2:
        return "small"
    return "tiny"


def print_gpu_config(device: str = "cpu") -> None:
    """Print GPU info and batch config at startup."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_mem / (1024 ** 3)
            print(f"  GPU: {props.name} ({vram_gb:.1f} GB)")
        else:
            print("  GPU: CUDA requested but not available, using CPU")
    except Exception:
        print("  GPU: detection failed, using CPU defaults")

    tier = get_gpu_tier(device)
    cfg = get_gpu_batch_config(device)
    print(f"  Tier: {tier}")
    print(f"  Batch config: YOLO chunk={cfg['yolo_chunk']} batch={cfg['yolo_batch']}, "
          f"TrackNet chunk={cfg['tracknet_chunk']}, RTMPose chunk={cfg['rtmpose_chunk']}, "
          f"BST batch={cfg['bst_batch']}")
