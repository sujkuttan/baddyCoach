#!/bin/bash
# Build mmcv in a clean Docker container.
# Requires: Docker installed and running
#
# Usage: bash colab/docker_build_mmcv.sh
# Output: mmcv_files.tar.gz in project root

set -e
cd "$(dirname "$0")/.."

docker run --rm -v "$(pwd)":/workspace -w /workspace python:3.11-slim bash -c '
set -e
apt-get update -qq && apt-get install -y -qq build-essential > /dev/null
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install mmcv==2.2.0 --no-binary mmcv -q
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

echo "Done. Upload mmcv_files.tar.gz to Google Drive."
