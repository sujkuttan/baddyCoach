#!/bin/bash
# Build mmcv with CUDA support in Docker.
# Requires: Docker + nvidia-container-toolkit
#
# Usage: bash colab/docker_build_mmcv.sh
# Output: mmcv_files.tar.gz in project root

set -e
cd "$(dirname "$0")/.."

echo "Building mmcv with CUDA support (requires nvidia-docker)..."
echo "This will take 10-15 minutes."

docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace nvidia/cuda:12.1.1-devel-ubuntu22.04 bash -c '
set -e
apt-get update -qq
apt-get install -y -qq build-essential python3 python3-pip python3-dev libgl1 libglib2.0-0 libxcb1 > /dev/null
ln -sf /usr/bin/python3 /usr/bin/python

echo "=== Installing PyTorch with CUDA 12.1 ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "=== Building mmcv 2.2.0 from source ==="
pip install wheel setuptools -q
pip install mmcv==2.2.0 --no-binary mmcv

echo "=== Packaging ==="
python -c "
import mmcv, os, tarfile
d = os.path.dirname(mmcv.__file__)
p = os.path.dirname(d)
with tarfile.open(\"/workspace/mmcv_files.tar.gz\", \"w:gz\") as tar:
    for root, dirs, files in os.walk(d):
        for f in files:
            tar.add(os.path.join(root, f), arcname=os.path.relpath(os.path.join(root, f), p))
print(\"mmcv_files.tar.gz created\")
"
'

echo ""
echo "Done. Upload mmcv_files.tar.gz to Google Drive."
ls -lh mmcv_files.tar.gz
