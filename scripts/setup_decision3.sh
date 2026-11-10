#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DECISION_ENV="${DECISION_ENV:-/home/junwue/envs/decision3}"
export HF_HOME="${HF_HOME:-/share/project/robot-rfm/cache/huggingface}"
if [[ ! -x "$DECISION_ENV/bin/python" ]]; then
  uv venv --python 3.11 "$DECISION_ENV"
fi
uv pip install --python "$DECISION_ENV/bin/python" -r "$PROJECT_ROOT/environments/decision3-requirements.txt"
uv pip check --python "$DECISION_ENV/bin/python"
"$DECISION_ENV/bin/python" - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from huggingface_hub import snapshot_download

print('Decision CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('Decision kernels:', causal_conv1d_fn.__module__, chunk_gated_delta_rule.__module__)
print(snapshot_download(
    'vllm-sr/d3-flash', revision='13dc1cc8ef23caefa765907e1d764751e5629f7d',
    allow_patterns=['*.py', '*.json', '*.jinja', '*.txt', '*.safetensors', 'README.md', 'LICENSE', 'NOTICE'],
    max_workers=4,
))
PY
