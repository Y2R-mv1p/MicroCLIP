#!/usr/bin/env bash
# =============================================================================
# Fetch the MicroPool-2M WebDataset release into MICRO_DATA_ROOT.
#
# Usage:
#   bash scripts/download_datasets.sh /path/to/MICRO_DATA_ROOT
#
# Environment:
#   HF_ENDPOINT          Hugging Face endpoint (e.g. https://hf-mirror.com)
#   MICROPOOL_REPO       dataset repo id, org/name (overrides the default)
#   MICROPOOL_INCLUDE    comma-separated --include filters for a partial download
#
# Layout produced (this is what paths.sh expects):
#   $MICRO_DATA_ROOT/MicroPool811/webdataset/wds_train/...
#   $MICRO_DATA_ROOT/MicroPool811/webdataset/wds_eval_*/...
#   $MICRO_DATA_ROOT/MicroPool811/webdataset/wds_train_aug/...
#   $MICRO_DATA_ROOT/dataset_new/webdataset/wds_vlm/...
#   $MICRO_DATA_ROOT/data_retrieval/wds_eval_fluogen_100{,0}/...
# =============================================================================
set -euo pipefail

ROOT="${1:-}"
if [[ -z "${ROOT}" ]]; then
  echo "usage: bash scripts/download_datasets.sh /path/to/MICRO_DATA_ROOT" >&2
  exit 1
fi

MICROPOOL_REPO="${MICROPOOL_REPO:-TODO_RELEASE_ORG/MicroPool-2M}"

if [[ "${MICROPOOL_REPO}" == TODO_* ]]; then
  cat >&2 <<'EOF'
ERROR: the dataset repository id is not set yet.

This release is staged but the MicroPool-2M dataset repository has not been
published. Once it exists, either edit MICROPOOL_REPO at the top of this script
or export it:

    export MICROPOOL_REPO=<org>/MicroPool-2M
    export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
    bash scripts/download_datasets.sh /path/to/MICRO_DATA_ROOT

Until then, point MICRO_DATA_ROOT at an existing copy of the dataset; paths.sh
requires MicroPool811/, dataset_new/ and data_retrieval/ under that root.
EOF
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=huggingface-cli
  else
    echo "Neither 'hf' nor 'huggingface-cli' found. Install huggingface_hub: pip install 'huggingface_hub[cli]'" >&2
    exit 1
  fi
else
  HF_CLI=hf
fi

mkdir -p "${ROOT}"
echo "Downloading ${MICROPOOL_REPO} -> ${ROOT}"
echo "  CLI: ${HF_CLI} ${HF_ENDPOINT:+| endpoint: ${HF_ENDPOINT}}"

INCLUDE_ARGS=()
if [[ -n "${MICROPOOL_INCLUDE:-}" ]]; then
  IFS=',' read -r -a _filters <<< "${MICROPOOL_INCLUDE}"
  for f in "${_filters[@]}"; do
    INCLUDE_ARGS+=(--include "${f}")
  done
fi

"${HF_CLI}" download "${MICROPOOL_REPO}" \
  --repo-type dataset \
  --local-dir "${ROOT}" \
  "${INCLUDE_ARGS[@]}"

cat <<EOF

Done. Verify the layout, then run:

    export MICRO_DATA_ROOT=${ROOT}
    bash scripts/smoke_test.sh

Full training needs the three training corpora (base + VLM + augmentation);
evaluation alone needs the wds_eval_* splits and the retrieval pools.
EOF
