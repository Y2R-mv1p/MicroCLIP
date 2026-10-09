#!/usr/bin/env bash
# ============================================================================
# MicroCLIP scale experiment — DFN2B-CLIP ViT-L/14  (~428M total params)
#   Runs the SAME method as train.sh (three-way text + MetaCLIP balance,
#   VLM/aug mixing off) with a different backbone, for parameter-scale
#   comparison. Only the backbone and the output directory differ from the
#   canonical run; data paths come from paths.sh (MICRO_DATA_ROOT) and the
#   weights are auto-downloaded from HuggingFace Hub on first run.
#
#   Source : apple/DFN2B-CLIP-ViT-L-14  (HuggingFace Hub)
#   Output : log_scale_DFN_ViT-L-14
#   Override any runtime setting via env, e.g.
#     EPOCHS=15 TRAIN_BATCH_SIZE=512 LEARNING_RATE=2e-5 NUM_GPUS=4 \
#     bash train_DFN_L14.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Backbone: hf-hub repo id (config + weights auto-download on first run).
# The SigLIP2 text towers additionally fetch their HF tokenizer on first run.
export BACKBONE_MODEL="hf-hub:apple/DFN2B-CLIP-ViT-L-14"
export BACKBONE_PRETRAINED="hf-hub"

# Per-model output dir (checkpoints / metrics / logs).
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/log_scale_DFN_ViT-L-14}"

# Everything else (datasets, method modules, hyperparameters, seven-way eval,
# launch) is identical to the canonical train.sh — see there for overrides.
exec bash "${SCRIPT_DIR}/train.sh"
