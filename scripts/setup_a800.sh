#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url "${TORCH_INDEX_URL}"
python -m pip install -e ".[test]"

if [[ "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-16}" \
    python -m pip install --no-build-isolation "flash-attn==2.7.4.post1"
fi

python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
PY
