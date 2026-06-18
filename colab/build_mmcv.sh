#!/bin/bash
# Build mmcv for upload to Google Drive.
# Works on any machine with Python 3.11+ and CUDA torch.
#
# Usage:
#   conda activate <env with python + torch+cuda>
#   bash colab/build_mmcv.sh
#
# Output: mmcv_files.tar.gz

set -e

echo "=== Build environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || echo "No torch found (ok for CPU build)"
python --version

echo ""
echo "=== Installing mmcv from source ==="
pip install mmcv==2.2.0 --no-binary mmcv

echo ""
echo "=== Packaging installed files ==="
python -c "
import mmcv, os, tarfile
mmcv_dir = os.path.dirname(mmcv.__file__)
parent_dir = os.path.dirname(mmv_dir)
out = 'mmcv_files.tar.gz'
with tarfile.open(out, 'w:gz') as tar:
    for root, dirs, files in os.walk(mmcv_dir):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, parent_dir)
            tar.add(full, arcname=arcname)
print(f'Created {out} ({os.path.getsize(out)/1024/1024:.1f} MB)')
"

echo ""
echo "=== Done ==="
echo "Upload mmcv_files.tar.gz to Google Drive"
echo "Extract to site-packages in Colab before importing mmcv"
