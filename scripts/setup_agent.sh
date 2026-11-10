#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_SOURCE="${PI05_SOURCE:-/home/junwue/deps/openpi-robocasa}"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.11 "$PROJECT_ROOT/.venv"
fi
uv pip install --no-sources --python "$PROJECT_ROOT/.venv/bin/python" -e "$PROJECT_ROOT[dev]" \
  -e "$PI05_SOURCE/packages/openpi-client"
uv pip check --python "$PROJECT_ROOT/.venv/bin/python"
