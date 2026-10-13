# MicroPool evaluation datasets

We provide the MicroPool evaluation-data reconstruction tools as part of the
[MicroCLIP repository](../README.md). No separate repository clone is needed.
This folder prepares data; model evaluation lives at
the repository root in `evaluate.py` / `eval.sh`.

This package reconstructs the exact datasets used in our evaluation from
original public sources. It does not redistribute any
third-party image bytes. The fixed manifests include source identifiers,
three-level text annotations, expected image SHA-256 values, and deterministic
conversion rules.

We evaluate MicroCLIP on four internal classification splits (23,600 images), the
external Light My Cells Source Shift benchmark (1,158 images), and a fixed
1,000-pair image-text retrieval pool. The older `ood` split and 100-pair
retrieval pool are excluded from our evaluation protocol and this package.

## Build

Python 3.13 is the validated release environment:

```bash
cd MicroCLIP/MicroPool  # from the parent directory of your clone
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Keep generated datasets outside the entire MicroCLIP checkout.

# Preflight only: reports every automatic and manual source without downloading.
./build.sh --root /data/micropool-eval --stage plan

# Complete build after supplying the local inputs reported by plan.json.
./build.sh --root /data/micropool-eval \
  --prepared-root /path/to/existing/FluoGen_data_JPG \
  --raw-root /path/to/official/raw/files
```

`--prepared-root` is optional and accepts a legally obtained, already converted
tree. `--raw-root` accepts official files at the source-relative paths recorded
in the manifests. Missing public files are fetched from their official hosts.
Downloads are resumable at the file level. Archive members are read with
validated HTTP Range requests when the official server supports them.

Some sources require an account, a manual official download, or explicit
permission. The preflight exits nonzero until these inputs are supplied. It
never treats missing data as success. See [docs/SOURCES.md](docs/SOURCES.md).

If a source archive was downloaded from its official page, place it under
`<root>/archives/` or pass `--archives DIR`. For CTC images, pass
`--authorized-source fluo` only after you have actually obtained the permission
required for your use; the flag grants no rights.

The build completes only after it has checked every reconstructed image against
the frozen SHA-256 value and verified every tar member, caption, metadata record,
split count, and class assignment. It writes:

```text
<root>/evaluation/iid/
<root>/evaluation/compositional_shift/
<root>/evaluation/cell_line_shift/
<root>/evaluation/organelle_set_shift/
<root>/evaluation/source_shift/
<root>/evaluation/retrieval_1000/
<root>/verification.json
<root>/micropool_eval.env
```

## Evaluate MicroCLIP

After obtaining a MicroCLIP checkpoint, install the model dependencies in
the environment used for evaluation (see the [root installation instructions](../README.md)).
From this `MicroPool/` directory:

```bash
bash ../eval.sh \
  --root /data/micropool-eval \
  --checkpoint /path/to/microclip_epoch015.pt
```

This reports Top-1 for the five classification evaluations and bidirectional
Recall@1 on the 1,000-pair retrieval pool. It uses the model preprocessing and
256-token text encoder provided in this repository.

## Text annotations

- L1 category captions use the deterministic template
  `a microscopy photo of {structure}_of_{cell}`.
- L2 image-specific detail captions were generated with
  Qwen3.5-397B-A17B, conditioned on the image and normalized category.
- L3 biological knowledge descriptions were generated with GPT-5.6 Sol for
  structures/markers and cell lines. Composite labels concatenate constituent
  structure descriptions in canonical order.

The frozen L2 captions are stored inside the compressed evaluation manifests;
the L3 lookup tables are under `semantic_caption/`. These annotations do not
include image bytes.

## Package preparation

`tools/export_release.py` and `tools/freeze_recipes.py` document how this package
was frozen from our final evaluated shards and source material. Users
do not need them. They deliberately require explicit local paths and cannot
silently resample a benchmark.
