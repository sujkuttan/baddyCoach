#!/usr/bin/env python3
"""Export MMPose HRNet-W32 to ONNX for BMCA pipeline.

Run in Colab or a clean venv with MMPose installed:
    pip install mmpose mmdet openmim && mim install mmcv
    python colab/export_hrnet.py
"""
import sys
from pathlib import Path

CKPT_DIR = Path("ckpts/mmpose")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = CKPT_DIR / "hrnet_w32_coco_256x192.onnx"


def export():
    try:
        import torch
        from mmpose.apis import MMPoseInferencer
    except ImportError:
        print("Install: pip install mmpose mmdet openmim && mim install mmcv")
        sys.exit(1)

    print("Loading MMPose HRNet-W32 inferencer...")
    inferencer = MMPoseInferencer('human')
    pose_estimator = inferencer.pose_estimator

    dummy_input = torch.randn(1, 3, 256, 192)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        pose_estimator = pose_estimator.cuda()

    print("Exporting to ONNX...")
    torch.onnx.export(
        pose_estimator,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


def test_export():
    import numpy as np
    import onnxruntime as ort

    print(f"\nTesting exported model...")
    session = ort.InferenceSession(
        str(OUTPUT_PATH),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    dummy = np.random.randn(1, 3, 256, 192).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    print(f"  Input: {dummy.shape}")
    print(f"  Outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape={out.shape}")
    print("  Verified OK")


if __name__ == "__main__":
    if not OUTPUT_PATH.exists():
        export()
    else:
        print(f"ONNX already exists: {OUTPUT_PATH}")
    test_export()
