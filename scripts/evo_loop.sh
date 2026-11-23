#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${EVO_ML_PYTHON:-python}" "$SCRIPT_ROOT/scripts/evo_loop.py" "$@"
