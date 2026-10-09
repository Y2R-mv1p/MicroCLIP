#!/usr/bin/env bash
# ============================================================================
# MicroCLIP scale experiment — PE-Core ViT-bigG/14-448  (~2200M total params)
#   Largest scale in the PE-Core family (giant). Runs the SAME method as
#   train.sh (three-way text + MetaCLIP balance, VLM/aug mixing off) with a
#   different backbone, for parameter-scale comparison. Only the backbone and
#   the output directory differ from the canonical run; data paths come from
#   paths.sh (MICRO_DATA_ROOT) and the weights are auto-downloaded from
#   HuggingFace Hub on first run.
#
#   Source : timm/PE-Core-bigG-14-448  (HuggingFace Hub)
#   Output : log_scale_PECore_ViT-bigG-14-448
#   Override any runtime setting via env, e.g.
#     EPOCHS=15 TRAIN_BATCH_SIZE=256 LEARNING_RATE=1e-5 NUM_GPUS=8 \
#     bash train_PECore_G14_448.sh
#   NOTE: this is a ~2.2B-param giant at 448px — expect high VRAM. Start with a
#   small TRAIN_BATCH_SIZE and keep GRAD_CHECKPOINTING=1 (default).
# ============================================================================
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Backbone: hf-hub repo id (config + weights auto-download on first run).
export BACKBONE_MODEL="hf-hub:timm/PE-Core-bigG-14-448"
export BACKBONE_PRETRAINED="hf-hub"

# Per-model output dir (checkpoints / metrics / logs).
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/log_scale_PECore_ViT-bigG-14-448}"

# Everything else (datasets, method modules, hyperparameters, seven-way eval,
# launch) is identical to the canonical train.sh — see there for overrides.
exec bash "${SCRIPT_DIR}/train.sh"
