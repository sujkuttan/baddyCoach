#!/bin/bash
# Build mmcv wheel from source for upload to Google Drive.
#
# Requirements:
#   - Python 3.12, CUDA toolkit installed
#   - PyTorch with CUDA (same major version as Colab, e.g. torch 2.x+cu12x)
#
# Usage:
#   conda activate <env with torch+cuda>
#   bash colab/build_mmcv.sh
#
# Output: ckpts/mmcv_wheel/mmcv-*.whl

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../ckpts/mmcv_wheel"
mkdir -p "$OUT_DIR"

echo "=== Build environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python --version

echo ""
echo "=== Building mmcv 2.1.0 from source ==="
pip wheel mmcv==2.1.0 --no-binary mmcv -w "$OUT_DIR" --no-clean 2>&1 | tail -5

echo ""
echo "=== Built wheel ==="
ls -lh "$OUT_DIR"/*.whl 2>/dev/null || echo "No wheel found — check build log above"

echo ""
echo "Next steps:"
echo "  1. Upload $OUT_DIR/*.whl to Google Drive"
echo "  2. Update BMCA_Colab.ipynb cell 1 to download from Drive instead of building"
