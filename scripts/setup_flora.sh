#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FLORA_ENV="${FLORA_ENV:-/home/junwue/envs/flora}"
export HF_HOME="${HF_HOME:-/share/project/robot-rfm/cache/huggingface}"
if [[ ! -x "$FLORA_ENV/bin/python" ]]; then
  uv venv --python 3.11 "$FLORA_ENV"
fi
uv pip install --python "$FLORA_ENV/bin/python" -r "$PROJECT_ROOT/environments/flora-requirements.txt"
uv pip check --python "$FLORA_ENV/bin/python"
"$FLORA_ENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    'nvidia/Flora-Rewarder', revision='031299e46e452fc0cedb4e0975eb9fa95fc37783',
    allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors', 'batch_utils/*.py', 'README.md', 'LICENSE*'],
    max_workers=4,
))
PY
