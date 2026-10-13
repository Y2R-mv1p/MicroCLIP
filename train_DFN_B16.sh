#!/usr/bin/env bash
# Canonical DFN2B ViT-B/16 training, 256-token context. See train.sh for settings.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE_MODEL="ViT-B-16"
export BACKBONE_PRETRAINED="${BACKBONE_PRETRAINED:-dfn2b}"
exec bash "${SCRIPT_DIR}/train.sh" "$@"
