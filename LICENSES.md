# License Compliance Audit

This project integrates several open-source models and libraries.
Below is the license status of each component.

## Direct Dependencies

| Component | License | Notes |
|-----------|---------|-------|
| YOLOv8 (Ultralytics) | **AGPL-3.0** | Commercial use requires a separate license from Ultralytics. The AGPL requires that any network service using it distribute full source code. |
| TrackNetV3 (custom) | MIT* | Project uses a custom UNet reimplementation; the original TrackNetV3 is MIT-licensed. Verify license of any distributed weights. |
| BST (Badminton Stroke Transformer) | MIT* | The BST-CG model port is MIT; verify with original repo at https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer |
| SoloShuttlePose | MIT* | Court detection model used for homography; verify original license. |
| RTMPose | Apache-2.0 | Part of MMPose; Apache 2.0 licensed. |
| FastAPI | MIT | |
| PyTorch | BSD-style | |
| ONNX Runtime | MIT | |

## Backend Python Packages (requirements.txt)

| Package | License |
|---------|---------|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| pydantic | MIT |
| pydantic-settings | MIT |
| torch | BSD-style |
| ultralytics | AGPL-3.0 |
| onnxruntime-gpu | MIT |
| opencv-python-headless | MIT |
| google-generativeai | Apache-2.0 |

## Frontend Packages

| Package | License |
|---------|---------|
| React | MIT |
| TypeScript | Apache-2.0 |
| Vite | MIT |
| Recharts | MIT |

## Compliance Requirements

1. **AGPL-3.0 (YOLOv8)**: If distributed as a network service, the complete corresponding source code must be made available to users. Consider purchasing a commercial Ultralytics license for closed-source deployment.

2. **Model Weights**: Each model checkpoint carries its own license terms. Before distributing any `.pt` or `.onnx` files, verify the license of the originating project.

3. **Attribution**: Include the original model paper citations and copyright notices if redistributing pretrained weights.

## Recommended Actions

- [ ] Purchase commercial Ultralytics license if deploying commercially
- [ ] Verify TrackNetV3 weight origin and license
- [ ] Verify BST weight origin and license
- [ ] Verify SoloShuttlePose court model weight license
- [ ] Add full open-source license notices to NOTICE file
