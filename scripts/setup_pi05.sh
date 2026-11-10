#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_SOURCE="${PI05_SOURCE:-/home/junwue/deps/openpi-robocasa}"
PI05_ENV="${PI05_ENV:-/home/junwue/envs/pi05}"
PI05_COMMIT=5a6beda9ff99da30b4e1b59320f6a32971d7c397
if [[ ! -d "$PI05_SOURCE" ]]; then
  git clone https://github.com/robocasa-benchmark/openpi.git "$PI05_SOURCE"
  git -C "$PI05_SOURCE" checkout --detach "$PI05_COMMIT"
fi
if [[ "$(git -C "$PI05_SOURCE" rev-parse HEAD)" != "$PI05_COMMIT" ]]; then
  echo "OpenPI revision differs from the validated version; use a separate checkout." >&2
  exit 1
fi
if [[ ! -x "$PI05_ENV/bin/python" ]]; then
  uv venv --python 3.11 "$PI05_ENV"
fi
# --no-sources follows the upstream pip installation route (LeRobot 0.3.3),
# avoiding the incompatible legacy uv source override and its missing LFS test artifacts.
uv pip install --no-sources --python "$PI05_ENV/bin/python" --index-url https://pypi.org/simple \
  -c "$PROJECT_ROOT/environments/pi05-constraints.txt" \
  -e "$PI05_SOURCE" -e "$PI05_SOURCE/packages/openpi-client" 'pytest>=8,<9' 'chex==0.1.89'
uv pip check --python "$PI05_ENV/bin/python"
XLA_PYTHON_CLIENT_PREALLOCATE=false "$PI05_ENV/bin/python" "$PROJECT_ROOT/scripts/verify_pi05.py"
