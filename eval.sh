#!/usr/bin/env bash
# Score a MicroCLIP checkpoint on the verified MicroPool paper evaluation data.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python}" "${SCRIPT_DIR}/evaluate.py" "$@"
