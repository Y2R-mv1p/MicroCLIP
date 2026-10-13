#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Central dataset / pretrained-weight configuration (see paths.sh). Required
# datasets must either be reachable via MICRO_DATA_ROOT or be exported individually.
source "${SCRIPT_DIR}/paths.sh"

# Backbone configuration. Override any value through an environment variable when
# changing architectures; leave an optional value empty to omit its CLI argument.
BACKBONE_MODEL="${BACKBONE_MODEL:-ViT-B-16}"
BACKBONE_FORCE_IMAGE_SIZE="${BACKBONE_FORCE_IMAGE_SIZE-}"
# Text encoder is forced to 256 context length so the semantic-caption-replaced
# titles (and the full long captions) fit under the token budget.
BACKBONE_FORCE_CONTEXT_LENGTH="${BACKBONE_FORCE_CONTEXT_LENGTH-256}"
BACKBONE_FORCE_QUICK_GELU="${BACKBONE_FORCE_QUICK_GELU:-0}"
BACKBONE_PRETRAINED_IMAGE="${BACKBONE_PRETRAINED_IMAGE:-0}"
BACKBONE_CACHE_DIR="${BACKBONE_CACHE_DIR-}"

# =============================================================================
# METHOD MODULES — each feature is an independent on/off switch.
# This repo's default method: three-way text selection and MetaCLIP balance ON,
# VLM mixing and data augmentation ON.
#   ENABLE_THREE_WAY_TEXT=1 -> TRAIN_LONG_CAPTION_MODE=three-way (randomly pick
#       long caption / short title / semantic-rewritten title per sample);
#       =0 -> "none" (plain short title only).
#   ENABLE_META_BALANCE=1   -> MetaCLIP online class balancing (METACLIP_BALANCE).
#   ENABLE_VLM_MIX=1        -> append the wds_vlm corpus to the training data.
#   ENABLE_AUG_MIX=1        -> append the wds_train_aug rare-class corpus.
# =============================================================================
ENABLE_THREE_WAY_TEXT="${ENABLE_THREE_WAY_TEXT:-1}"
ENABLE_META_BALANCE="${ENABLE_META_BALANCE:-1}"
ENABLE_VLM_MIX="${ENABLE_VLM_MIX:-1}"
ENABLE_AUG_MIX="${ENABLE_AUG_MIX:-1}"

# --- Derived: training data composition. wds_train is always the base; VLM and
# augmentation corpora are appended as WebDataset "::" components only when their
# module is enabled. The mix ratio follows shard counts (250 train + 61 vlm shards
# -> ~19.6% VLM when enabled). The braces must reach the training code's
# braceexpand verbatim, so they are never expanded here.
TRAIN_DATA="${WDS_TRAIN:-}"
if [[ "${ENABLE_VLM_MIX}" == "1" ]]; then
  TRAIN_DATA="${TRAIN_DATA}::${WDS_VLM:-}"
fi
if [[ "${ENABLE_AUG_MIX}" == "1" ]]; then
  TRAIN_DATA="${TRAIN_DATA}::${WDS_AUG:-}"
fi

# --- Derived: per-epoch sample count (drives the LR warmup scheduler). Sum the
# component counts that are actually mixed in; export TRAIN_NUM_SAMPLES to pin it.
if [[ -z "${TRAIN_NUM_SAMPLES+x}" ]]; then
  TRAIN_NUM_SAMPLES="${TRAIN_NUM_BASE}"
  if [[ "${ENABLE_VLM_MIX}" == "1" ]]; then
    TRAIN_NUM_SAMPLES=$((TRAIN_NUM_SAMPLES + TRAIN_NUM_VLM))
  fi
  if [[ "${ENABLE_AUG_MIX}" == "1" ]]; then
    TRAIN_NUM_SAMPLES=$((TRAIN_NUM_SAMPLES + TRAIN_NUM_AUG))
  fi
fi

# --- Derived: training text mode (three-way vs plain title). ---
if [[ -z "${TRAIN_LONG_CAPTION_MODE+x}" ]]; then
  if [[ "${ENABLE_THREE_WAY_TEXT}" == "1" ]]; then
    TRAIN_LONG_CAPTION_MODE="three-way"
  else
    TRAIN_LONG_CAPTION_MODE="none"
  fi
fi

# --- Derived: MetaCLIP balance toggle. ---
if [[ -z "${METACLIP_BALANCE+x}" ]]; then
  METACLIP_BALANCE="${ENABLE_META_BALANCE}"
fi

# --- Required dataset validation. Every path used by the enabled modules must
# resolve; if any is empty, point MICRO_DATA_ROOT at your dataset root (see
# paths.sh) or export the individual variable.
missing_reqs=()
for req_var in WDS_TRAIN; do
  if [[ -z "${!req_var:-}" ]]; then
    missing_reqs+=("${req_var}")
  fi
done
if [[ "${ENABLE_VLM_MIX}" == "1" && -z "${WDS_VLM:-}" ]]; then
  missing_reqs+=("WDS_VLM")
fi
if [[ "${ENABLE_AUG_MIX}" == "1" && -z "${WDS_AUG:-}" ]]; then
  missing_reqs+=("WDS_AUG")
fi
if [[ "${#missing_reqs[@]}" -gt 0 ]]; then
  echo "ERROR: the following required dataset variables are empty: ${missing_reqs[*]}" >&2
  echo "  Set MICRO_DATA_ROOT to your dataset root (see paths.sh), or export each variable individually." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/microclip_dfn_b16}"

# Runtime and selection configuration.
NUM_GPUS="${NUM_GPUS:-8}"
EPOCHS="${EPOCHS:-30}"
# Per-GPU batch size raised 4x (256 -> 1024, global 2048 -> 8192 on 8 GPUs) to speed up
# training; LEARNING_RATE follows the linear scaling rule (x4 -> 4e-5) and WARMUP is scaled
# down to keep the same warmup sample budget (and stay under the ~9k total steps at this batch).
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
WORKERS="${WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRECISION="${PRECISION:-amp_bf16}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
WARMUP="${WARMUP:-2500}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}"
# Persist a lightweight model-state snapshot per epoch at
# ${OUTPUT_DIR}/checkpoints/epochs/epoch_XXX.pt (alongside last.pt with optimizer state).
# On by default for independent checkpoint evaluation.
SAVE_EVERY_EPOCH_WEIGHTS="${SAVE_EVERY_EPOCH_WEIGHTS:-1}"

# Semantic title replacement. The training title "a microscopy photo of {structure}_of_{cell}"
# is rewritten by substituting the structure/cell names with their biological semantic captions
# from SEMANTIC_STRUCTURE_TSV / SEMANTIC_CELL_TSV. Compound structures (A_and_B) are decomposed
# and each component's caption is concatenated. The text encoder is forced to 256 tokens.
#
# SEMANTIC_REPLACE_MODE choices:
#   both           replace structure AND cell with their semantic captions
#   structure-only replace only the structure name
#   cell-only      replace only the cell name
#   none           keep the original title (pure ablation)
#   random         per-sample independent replacement using
#                  SEMANTIC_REPLACE_STRUCTURE_PROB / SEMANTIC_REPLACE_CELL_PROB
SEMANTIC_STRUCTURE_TSV="${SEMANTIC_STRUCTURE_TSV:-${SCRIPT_DIR}/semantic_caption/A_structures_split.tsv}"
SEMANTIC_CELL_TSV="${SEMANTIC_CELL_TSV:-${SCRIPT_DIR}/semantic_caption/cell_semantic_recaption_manifest.tsv}"
# Default mode is "both" for the rewrite_caption branch: the structure name (before
# "_of_") and the cell name (after "_of_") are both replaced with semantic captions.
SEMANTIC_REPLACE_MODE="${SEMANTIC_REPLACE_MODE:-both}"
SEMANTIC_REPLACE_STRUCTURE_PROB="${SEMANTIC_REPLACE_STRUCTURE_PROB:-1.0}"
SEMANTIC_REPLACE_CELL_PROB="${SEMANTIC_REPLACE_CELL_PROB:-1.0}"

# Text-side mixing. TRAIN_LONG_CAPTION_MODE is derived from ENABLE_THREE_WAY_TEXT
# above (three-way | none). In three-way mode each sample's training text is
# randomly picked between:
#   LONG_CAPTION_SAMPLE_PROB    -> full long caption (truncated to context)
#   SHORT_CAPTION_SAMPLE_PROB   -> original short title, unchanged
#   REWRITE_CAPTION_SAMPLE_PROB -> title with structure/cell names replaced by semantic captions
# (these three probabilities must sum to 1; the defaults are the 25/50/25 mix).
# VLM samples carry no canonical structure/cell names, so their rewrite branch uses the
# full long caption instead; their longer captions are never randomly chunked.
# The other modes keep the two-way title-or-* behavior: title-or-full-long-caption keeps the
# full caption (no slicing); switch to title-or-random-chunk for the sliced (chunked) logic.
LONG_CAPTION_SAMPLE_PROB="${LONG_CAPTION_SAMPLE_PROB:-0.25}"
SHORT_CAPTION_SAMPLE_PROB="${SHORT_CAPTION_SAMPLE_PROB:-0.5}"
REWRITE_CAPTION_SAMPLE_PROB="${REWRITE_CAPTION_SAMPLE_PROB:-0.25}"
LONG_CAPTION_CHUNK_TOKENS="${LONG_CAPTION_CHUNK_TOKENS:-75}"
LONG_CAPTION_CHUNK_OVERLAP="${LONG_CAPTION_CHUNK_OVERLAP:-8}"

# MetaCLIP-style online metadata balancing. METACLIP_BALANCE is derived from
# ENABLE_META_BALANCE above; the rest are balancing knobs. The first run scans
# only .txt members in the tar shards and caches their global counts. Set
# METACLIP_BALANCE_SOURCE=json:<key> to balance a scalar/list JSON field instead.
METACLIP_BALANCE_SOURCE="${METACLIP_BALANCE_SOURCE:-txt}"
METACLIP_BALANCE_THRESHOLD="${METACLIP_BALANCE_THRESHOLD:-5000}"
METACLIP_COUNTS_FILE="${METACLIP_COUNTS_FILE:-${OUTPUT_DIR}/metaclip_balance_counts.json}"
# NOTE: toggling ENABLE_VLM_MIX / ENABLE_AUG_MIX changes TRAIN_DATA and makes the
# cached counts stale — run the first training with METACLIP_FORCE_RECOUNT=1 (or
# delete METACLIP_COUNTS_FILE) so the new corpus' class counts are included.
METACLIP_FORCE_RECOUNT="${METACLIP_FORCE_RECOUNT:-0}"
METACLIP_COUNT_WORKERS="${METACLIP_COUNT_WORKERS:-8}"

if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 1
fi
if [[ -z "${BACKBONE_MODEL}" || -z "${BACKBONE_PRETRAINED}" ]]; then
  echo "BACKBONE_MODEL and BACKBONE_PRETRAINED must both be non-empty." >&2
  exit 1
fi
if [[ "${TRAIN_DATA}" =~ \{([0-9]+)\.\.([0-9]+)\.([[:alnum:]]+)\} ]]; then
  ORIGINAL_TRAIN_DATA="${TRAIN_DATA}"
  MALFORMED_RANGE="${BASH_REMATCH[0]}"
  FIXED_RANGE="{${BASH_REMATCH[1]}..${BASH_REMATCH[2]}}.${BASH_REMATCH[3]}"
  TRAIN_DATA="${TRAIN_DATA/${MALFORMED_RANGE}/${FIXED_RANGE}}"
  echo "Normalized malformed TRAIN_DATA:" >&2
  echo "  from: ${ORIGINAL_TRAIN_DATA}" >&2
  echo "  to:   ${TRAIN_DATA}" >&2
fi
for toggle_name in BACKBONE_FORCE_QUICK_GELU BACKBONE_PRETRAINED_IMAGE GRAD_CHECKPOINTING METACLIP_BALANCE METACLIP_FORCE_RECOUNT SAVE_EVERY_EPOCH_WEIGHTS ENABLE_THREE_WAY_TEXT ENABLE_META_BALANCE ENABLE_VLM_MIX ENABLE_AUG_MIX; do
  toggle_value="${!toggle_name}"
  if [[ "${toggle_value}" != "0" && "${toggle_value}" != "1" ]]; then
    echo "${toggle_name} must be 0 or 1, got: ${toggle_value}" >&2
    exit 1
  fi
done
case "${SEMANTIC_REPLACE_MODE}" in
  both|structure-only|cell-only|none|random) ;;
  *)
    echo "SEMANTIC_REPLACE_MODE must be one of both|structure-only|cell-only|none|random, got: ${SEMANTIC_REPLACE_MODE}" >&2
    exit 1
    ;;
esac
for prob_name in SEMANTIC_REPLACE_STRUCTURE_PROB SEMANTIC_REPLACE_CELL_PROB LONG_CAPTION_SAMPLE_PROB SHORT_CAPTION_SAMPLE_PROB REWRITE_CAPTION_SAMPLE_PROB; do
  prob_value="${!prob_name}"
  if [[ ! "${prob_value}" =~ ^0(\.[0-9]+)?$ && "${prob_value}" != "1" && "${prob_value}" != "1.0" ]]; then
    echo "${prob_name} must be in [0, 1], got: ${prob_value}" >&2
    exit 1
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-select GPUs." >&2
  exit 1
fi

GPU_IDS=$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    sort -t, -k2,2nr |
    head -n "${NUM_GPUS}" |
    awk -F, '{gsub(/[[:space:]]/, "", $1); print $1}' |
    paste -sd, -
)

GPU_COUNT=$(printf '%s\n' "${GPU_IDS}" | awk -F, '{print NF}')
if [[ -z "${GPU_IDS}" || "${GPU_COUNT}" -lt "${NUM_GPUS}" ]]; then
  echo "Only found ${GPU_COUNT} GPU(s), but ${NUM_GPUS} are required." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

BACKBONE_ARGS=(
  --model "${BACKBONE_MODEL}"
  --pretrained "${BACKBONE_PRETRAINED}"
)
if [[ -n "${BACKBONE_FORCE_IMAGE_SIZE}" ]]; then
  read -r -a BACKBONE_IMAGE_SIZE_VALUES <<< "${BACKBONE_FORCE_IMAGE_SIZE}"
  BACKBONE_ARGS+=(--force-image-size "${BACKBONE_IMAGE_SIZE_VALUES[@]}")
fi
if [[ -n "${BACKBONE_FORCE_CONTEXT_LENGTH}" ]]; then
  BACKBONE_ARGS+=(--force-context-length "${BACKBONE_FORCE_CONTEXT_LENGTH}")
fi
if [[ -n "${BACKBONE_CACHE_DIR}" ]]; then
  BACKBONE_ARGS+=(--cache-dir "${BACKBONE_CACHE_DIR}")
fi
if [[ "${BACKBONE_FORCE_QUICK_GELU}" == "1" ]]; then
  BACKBONE_ARGS+=(--force-quick-gelu)
fi
if [[ "${BACKBONE_PRETRAINED_IMAGE}" == "1" ]]; then
  BACKBONE_ARGS+=(--pretrained-image)
fi

TRAINING_FLAGS=(--local-loss --gather-with-grad)
if [[ -n "${RESUME:-}" ]]; then
  TRAINING_FLAGS+=(--resume "${RESUME}")
fi
if [[ "${GRAD_CHECKPOINTING}" == "1" ]]; then
  TRAINING_FLAGS+=(--grad-checkpointing)
fi
if [[ "${SAVE_EVERY_EPOCH_WEIGHTS}" == "1" ]]; then
  TRAINING_FLAGS+=(--save-every-epoch-weights)
fi

METACLIP_ARGS=()
if [[ "${METACLIP_BALANCE}" == "1" ]]; then
  if ! [[ "${METACLIP_BALANCE_THRESHOLD}" =~ ^[1-9][0-9]*$ ]]; then
    echo "METACLIP_BALANCE_THRESHOLD must be a positive integer, got: ${METACLIP_BALANCE_THRESHOLD}" >&2
    exit 1
  fi
  if ! [[ "${METACLIP_COUNT_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "METACLIP_COUNT_WORKERS must be a positive integer, got: ${METACLIP_COUNT_WORKERS}" >&2
    exit 1
  fi
  echo "Effective TRAIN_DATA=${TRAIN_DATA}"
  if [[ "${METACLIP_FORCE_RECOUNT}" == "1" || ! -s "${METACLIP_COUNTS_FILE}" ]]; then
    echo "Building MetaCLIP metadata counts at ${METACLIP_COUNTS_FILE}"
    PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}" python -m open_clip_train.metaclip_balance \
      --train-data "${TRAIN_DATA}" \
      --output "${METACLIP_COUNTS_FILE}" \
      --source "${METACLIP_BALANCE_SOURCE}" \
      --threshold "${METACLIP_BALANCE_THRESHOLD}" \
      --workers "${METACLIP_COUNT_WORKERS}"
  fi
  METACLIP_ARGS=(
    --metaclip-balance-counts "${METACLIP_COUNTS_FILE}"
    --metaclip-balance-source "${METACLIP_BALANCE_SOURCE}"
    --metaclip-balance-threshold "${METACLIP_BALANCE_THRESHOLD}"
  )
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Backbone=${BACKBONE_MODEL} pretrained=${BACKBONE_PRETRAINED}"
echo "Modules: three-way-text=${ENABLE_THREE_WAY_TEXT} meta-balance=${ENABLE_META_BALANCE} vlm-mix=${ENABLE_VLM_MIX} aug-mix=${ENABLE_AUG_MIX}"
echo "Train data: ${TRAIN_DATA}"
echo "Train samples/epoch: ${TRAIN_NUM_SAMPLES}"
echo "MetaCLIP balancing=${METACLIP_BALANCE} source=${METACLIP_BALANCE_SOURCE} threshold=${METACLIP_BALANCE_THRESHOLD}"
echo "Training only; run eval.sh separately to evaluate a checkpoint."
echo "Per-epoch results: ${OUTPUT_DIR}/metrics.csv"

PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}" torchrun --nproc_per_node "${NUM_GPUS}" \
  -m open_clip_train.train \
  "${BACKBONE_ARGS[@]}" \
  --train-data "${TRAIN_DATA}" \
  --train-num-samples "${TRAIN_NUM_SAMPLES}" \
  "${METACLIP_ARGS[@]}" \
  --csv-caption-key txt \
  --train-long-caption-key long_caption \
  --train-long-caption-mode "${TRAIN_LONG_CAPTION_MODE}" \
  --long-caption-sample-prob "${LONG_CAPTION_SAMPLE_PROB}" \
  --short-caption-sample-prob "${SHORT_CAPTION_SAMPLE_PROB}" \
  --rewrite-caption-sample-prob "${REWRITE_CAPTION_SAMPLE_PROB}" \
  --long-caption-chunk-tokens "${LONG_CAPTION_CHUNK_TOKENS}" \
  --long-caption-chunk-overlap "${LONG_CAPTION_CHUNK_OVERLAP}" \
  --semantic-structure-tsv "${SEMANTIC_STRUCTURE_TSV}" \
  --semantic-cell-tsv "${SEMANTIC_CELL_TSV}" \
  --semantic-replace-mode "${SEMANTIC_REPLACE_MODE}" \
  --semantic-replace-structure-prob "${SEMANTIC_REPLACE_STRUCTURE_PROB}" \
  --semantic-replace-cell-prob "${SEMANTIC_REPLACE_CELL_PROB}" \
  --template "{name}" \
  --precision "${PRECISION}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --warmup "${WARMUP}" \
  --epochs "${EPOCHS}" \
  --workers "${WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --output-dir "${OUTPUT_DIR}" \
  "${TRAINING_FLAGS[@]}" "$@"
