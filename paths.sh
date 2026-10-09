#!/usr/bin/env bash
# =============================================================================
# Central dataset / pretrained-weight path configuration.
#
# This repository ships CODE ONLY. The datasets live in a separate repository and
# must be downloaded by the user; the backbone weights are auto-downloaded by
# open_clip on first run (see BACKBONE_PRETRAINED below).
#
# All dataset defaults below are derived from a single root, MICRO_DATA_ROOT, set
# to the folder that directly contains MicroPool811/, dataset_new/ and
# data_retrieval/ (i.e. the replacement for /data2/ryr on the author's machine):
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
    WDS_TRAIN="${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_train/shard-{000000..000249}.tar"
  fi
  if [[ -z "${WDS_VLM+x}" ]]; then
    WDS_VLM="${MICRO_DATA_ROOT}/dataset_new/webdataset/wds_vlm/shard-{000000..000060}.tar"
  fi
  if [[ -z "${WDS_AUG+x}" ]]; then
    WDS_AUG="${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_train_aug/shard-{000000..000017}.tar"
  fi

  # --- Evaluation data ---
  IID_DATA="${IID_DATA:-${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_eval_iid/shard-000000.tar}"
  PARTIAL_OOD_DATA="${PARTIAL_OOD_DATA:-${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_eval_partial_ood/shard-000000.tar}"
  OOD_DATA="${OOD_DATA:-${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_eval_ood/shard-000000.tar}"
  CELL_OOD_DATA="${CELL_OOD_DATA:-${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_eval_cell_ood/shard-000000.tar}"
  STRUCTURE_OOD_DATA="${STRUCTURE_OOD_DATA:-${MICRO_DATA_ROOT}/MicroPool811/webdataset/wds_eval_structure_ood/shard-000000.tar}"
  RETRIEVAL_DATA="${RETRIEVAL_DATA:-${MICRO_DATA_ROOT}/data_retrieval/wds_eval_fluogen_100/shard-000000.tar}"
  # Second retrieval pool (1000 images). On the downloaded/HuggingFace layout this
  # corpus lives under MicroPool/ (not data_retrieval/), so the default points
  # there — the author's local copy under /data2/ryr/data_retrieval/ must be
  # supplied via RETRIEVAL_DATA_1000 instead.
  RETRIEVAL_DATA_1000="${RETRIEVAL_DATA_1000:-${MICRO_DATA_ROOT}/MicroPool/wds_eval_fluogen_1000/shard-000000.tar}"
fi

# --- Per-dataset sample counts. train.sh sums the enabled components into
# TRAIN_NUM_SAMPLES (drives the LR warmup scheduler). Verified from the MetaCLIP
# counts of the 311-shard (train+vlm) run: total 3,108,357 - wds_vlm 609,740
# = 2,498,617 wds_train samples; wds_train_aug holds 175,516 rare-class copies.
TRAIN_NUM_BASE="${TRAIN_NUM_BASE:-2498617}"
TRAIN_NUM_VLM="${TRAIN_NUM_VLM:-609740}"
TRAIN_NUM_AUG="${TRAIN_NUM_AUG:-175516}"
