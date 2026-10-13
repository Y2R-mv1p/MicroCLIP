#!/usr/bin/env bash
# Build only the six evaluation datasets used in the paper. No training images.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${MICRO_PYTHON:-python3}"
exec "$PY" "$HERE/tools/eval.py" "$@"
