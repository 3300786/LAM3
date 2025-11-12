#!/usr/bin/env bash
set -e
python3 -V
pip install -U pip
pip install -r requirements.txt
python - <<'PY'
import torch, platform
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Python:", platform.python_version())
PY
