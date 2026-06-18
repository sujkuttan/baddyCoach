#!/bin/bash
# Build mmcv wheel from source for upload to Google Drive.
#
# Requirements:
#   - Python 3.12, CUDA toolkit installed
#   - PyTorch with CUDA (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121)
#
# Usage:
#   conda activate <env with python 3.12 + torch+cuda>
#   bash build_mmcv.sh
#
# Output: mmcv-2.2.0-cp312-*.whl in current directory

set -e

OUT_DIR="$(pwd)"
mkdir -p "$OUT_DIR"

echo "=== Build environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python --version

echo ""
echo "=== Building mmcv 2.2.0 from source ==="
pip wheel mmcv==2.2.0 --no-binary mmcv --no-build-isolation -w "$OUT_DIR" 2>&1 | tail -10

echo ""
echo "=== Built wheel ==="
ls -lh "$OUT_DIR"/mmcv-*.whl 2>/dev/null || echo "No wheel found — check build log above"

echo ""
echo "Next steps:"
echo "  1. Upload mmcv-2.2.0-*.whl to Google Drive"
echo "  2. Set MMCV_DRIVE_FILE_ID in BMCA_Colab.ipynb cell 1"
echo "  3. Set MMCV_DRIVE_FILE_ID env var when running pipeline.py"
