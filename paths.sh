#!/usr/bin/env bash
# =============================================================================
# Central dataset / pretrained-weight path configuration.
#
# This repository ships CODE ONLY. The datasets live in a separate repository and
# must be downloaded by the user; the backbone weights are auto-downloaded by
# open_clip on first run (see BACKBONE_PRETRAINED below).
#
# All dataset defaults below are derived from a single root, MICRO_DATA_ROOT, set
# to the folder that directly contains training/:
#
#   MICRO_DATA_ROOT=/mnt/datasets ./train.sh
#
# Any individual path can be overridden by exporting its own variable first, e.g.
#
#   WDS_TRAIN=/data/my/wds/shard-{000000..000249}.tar ./train.sh
#
# The training-data components (WDS_TRAIN / WDS_VLM / WDS_AUG) are combined into
# the final TRAIN_DATA inside train.sh according to its method modules
# (ENABLE_VLM_MIX / ENABLE_AUG_MIX) — this file only declares where each corpus
# lives and how many samples it contains.
#
# If MICRO_DATA_ROOT is empty (and no per-variable override is given), the variable
# is left unset and train.sh refuses to start with a message naming the missing one.
# =============================================================================

MICRO_DATA_ROOT="${MICRO_DATA_ROOT:-}"

# Pretrained backbone: open_clip pretrained tag. On first run open_clip downloads
# apple/DFN2B-CLIP-ViT-B-16 from HuggingFace into the HF cache (identical weights
# to the author's local model_cache copy).
BACKBONE_PRETRAINED="${BACKBONE_PRETRAINED:-dfn2b}"

if [[ -n "${MICRO_DATA_ROOT}" ]]; then
  # --- Training data components (WebDataset shard patterns). The braces are
  # intentional: they reach the training code's braceexpand verbatim, so assign
  # directly via an unset-check (${VAR:-...} would swallow the inner "}").
  if [[ -z "${WDS_TRAIN+x}" ]]; then
    WDS_TRAIN="${MICRO_DATA_ROOT}/training/microscopy/shard-{000000..000249}.tar"
  fi
  if [[ -z "${WDS_VLM+x}" ]]; then
    WDS_VLM="${MICRO_DATA_ROOT}/training/auxiliary/shard-{000000..000060}.tar"
  fi
  if [[ -z "${WDS_AUG+x}" ]]; then
    WDS_AUG="${MICRO_DATA_ROOT}/training/augmentation/shard-{000000..000017}.tar"
  fi

fi

# --- Per-dataset sample counts. train.sh sums the enabled components into
# TRAIN_NUM_SAMPLES (drives the LR warmup scheduler). Verified from the MetaCLIP
# counts of the 311-shard (train+vlm) run: total 3,108,357 - wds_vlm 609,740
# = 2,498,617 wds_train samples; wds_train_aug holds 175,516 rare-class copies.
TRAIN_NUM_BASE="${TRAIN_NUM_BASE:-2498617}"
TRAIN_NUM_VLM="${TRAIN_NUM_VLM:-609740}"
TRAIN_NUM_AUG="${TRAIN_NUM_AUG:-175516}"
