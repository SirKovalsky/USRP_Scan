#!/usr/bin/env bash
# Запуск SDR Scan (Linux/macOS). Все аргументы передаются в run.py.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi

exec "$PY" run.py "$@"
