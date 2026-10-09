# MicroCLIP

**A 150M-parameter CLIP model for fluorescence microscopy.**
MicroCLIP reaches **98.46% in-domain zero-shot classification** over 2,032 microscopy categories and **67.70% / 34.07% / 46.13%** under compositional, cell-line, and organelle-set shift — while every general-purpose vision–language model we compare against stays below 7.63% on those splits. It also supports cross-modal retrieval (**61.5% I2T / 63.3% T2I Recall@1**), where frozen general-purpose VLMs sit below 0.7%.

MicroCLIP is fine-tuned from [DFN2B-CLIP ViT-B/16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16) on **MicroPool-2M**, a 2.5M-image corpus of single-channel fluorescence micrographs paired with three levels of text (category title, per-image description, biological knowledge).

> Paper: **TBD (arXiv link pending)** · Weights: **TBD (Hugging Face)** · Dataset: **TBD (Hugging Face)**

---

## Results

Zero-shot classification Top-1 (%) and retrieval Recall@1 (%) on the 1,000-pair pool. All MicroCLIP numbers are read from a single checkpoint (epoch 15, selected by mean Top-1 over the five classification splits); baselines use released weights with their native text configuration, no microscopy adaptation and no prompt ensembling.

| Method | Params | IID | Compositional | Cell-Line | Organelle-Set | Source Shift | I2T | T2I |
|---|---|---|---|---|---|---|---|---|
| Fully supervised classifier | — | 97.74 | — | — | — | — | — | — |
| Cell-DINO | 304M | 36.06 | 0.98 | — | — | — | — | — |
| CLIP (OpenAI ViT-B/16) | 150M | 2.97 | 5.12 | 1.80 | 2.54 | 8.38 | 0.7 | 0.2 |
| SigLIP (ViT-B/16) | 203M | 3.05 | 1.88 | 1.07 | 1.26 | 9.33 | 0.2 | 0.2 |
| SigLIP 2 (ViT-B/16) | 375M | 3.08 | 2.28 | 3.59 | 1.11 | 10.79 | 0.2 | 0.3 |
| PE-Core (ViT-B/16) | 448M | 3.34 | 3.78 | 7.63 | 2.19 | 22.54 | 0.4 | 0.0 |
| DFN2B-CLIP (ViT-B/16) | 150M | 1.44 | 1.41 | 4.32 | 0.81 | 16.49 | 0.5 | 0.1 |
| **MicroCLIP** | **150M** | **98.46** | **67.70** | **34.07** | **46.13** | **36.53** | **61.5** | **63.3** |

*Source Shift* is the external [Light My Cells](https://doi.org/10.1038/s41592-026-0000-0) benchmark (1,158 images, 13 classes). Classification splits are balanced at 100 images per class; see [Datasets](#datasets) for split definitions.

## Install

```bash
git clone <REPO_URL> MicroCLIP && cd MicroCLIP
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The vendored open_clip fork (custom 256-token text context) ships in src/.
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The training environment this release was validated against is pinned in `requirements.txt` (Python 3.13, PyTorch 2.12, CUDA 13). A single CUDA GPU with ≥16 GB is enough for inference; training used 8× 72 GB GPUs.

## Inference

`inference.py` loads a released checkpoint and runs zero-shot classification and image/text embedding, with no dataset download required.

```bash
# No example images are shipped with the repository; pull a few out of a dataset
# shard to try the quickstart immediately:
bash scripts/make_examples.sh /path/to/wds_train/shard-000000.tar 3 examples

python inference.py \
  --checkpoint /path/to/microclip_epoch015.pt \
  --images examples/00000000.jpg examples/00000001.jpg \
  --classes "a microscopy photo of ACTB_of_A549" \
            "a microscopy photo of Nucleus_of_HeLa" \
            "a microscopy photo of Mitochondria_of_U2OS"
```

Class strings are the **full L1 title**, not a bare label: `a microscopy photo of {structure}_of_{cell}`. Structure and cell names use the corpus vocabulary (e.g. `Nucleus`, `Actin_filaments`, `Plasma_membrane`; `HeLa`, `A549`, `U2OS`), with composite structures joined by `_and_` (e.g. `F_actin_and_Plasma_membrane`).

As a library:

```python
import inference

model, preprocess, tokenizer = inference.load_model("microclip_epoch015.pt", device="cuda")
image_features = inference.encode_images(model, preprocess, ["my_cell.png"], device="cuda")
text_features = inference.encode_texts(model, tokenizer, ["a microscopy photo of Nucleus_of_HeLa"], device="cuda")
```

Both towers return L2-normalized features, so cosine similarity is a plain dot product — use them directly for retrieval, clustering, or as a frozen encoder for linear probes.

## Checkpoints

| Checkpoint | Init. | Size | Notes |
|---|---|---|---|
| `microclip_epoch015.safetensors` | DFN2B-CLIP ViT-B/16 | 299 MB | The released model (fp16). Every number in the table above comes from these weights. |

The artifact is produced from the training checkpoint with the bundled converter:

```bash
python tools/convert_checkpoint.py checkpoints/epochs/epoch_015.pt \
    -o microclip_epoch015.safetensors --format safetensors --dtype fp16   # 299 MB
# or, for the full-precision fp32 state_dict:
python tools/convert_checkpoint.py checkpoints/epochs/epoch_015.pt -o microclip_epoch015.pt  # 599 MB
```

**TBD (Hugging Face)** — until the weights are uploaded, the checkpoint is `checkpoints/epochs/epoch_015.pt` from the training run (599 MB; shape-compatible with `ViT-B-16` + `force_context_length=256`). `inference.py` loads either format.

## Datasets

MicroPool-2M is released separately from this repository. `paths.sh` resolves every dataset from a single root:

```bash
export MICRO_DATA_ROOT=/path/to/containing/folder   # contains MicroPool811/, dataset_new/, data_retrieval/
```

| Component | Used by | Size | Link |
|---|---|---|---|
| `MicroPool811/webdataset/wds_train` | training (base corpus) | 2,498,617 images | **TBD (Hugging Face)** |
| `dataset_new/webdataset/wds_vlm` | training (natural-image auxiliary) | 609,740 records | **TBD (Hugging Face)** |
| `MicroPool811/webdataset/wds_train_aug` | training (rare-class augmentation) | 175,516 records | **TBD (Hugging Face)** |
| `MicroPool811/webdataset/wds_eval_*` | the five evaluation splits | 30,000 images | **TBD (Hugging Face)** |
| `data_retrieval/wds_eval_fluogen_100{0}` | retrieval pools (100 / 1,000 pairs) | 1,100 images | **TBD (Hugging Face)** |

```bash
# Convenience downloader (Hugging Face Hub; set HF_ENDPOINT for mirrors)
bash scripts/download_datasets.sh /path/to/MICRO_DATA_ROOT
```

**MicroPool-2M in one paragraph.** 3,301,696 annotated single-channel fluorescence micrographs aggregated from public microscopy repositories (Human Protein Atlas, Image Data Resource, RxRx, Broad Bioimage Benchmark Collection) and curated literature, labeled by subcellular structure and cell line — 537 structure labels × 109 cell lines, of which 3,391 combinations (≈5.8% of the product space) carry images. A content-level (MD5) deduplication removes byte-identical duplicates so no image appears in two splits; 3,301,672 unique images are partitioned into a 2,498,617-image training split (1,968 classes, 489 structures, 32 cell lines), a 30,000-image evaluation ladder (five splits, 100 images per class), and a discarded surplus set. Each sample ships as a WebDataset record with a JPEG, the L1 template title, the L2 per-image long caption, and metadata. Splits are deterministic under a fixed seed and validated by an independent audit that re-derives the disjointness and per-class-count assertions from the released shards.

| Evaluation split | Classes | Structures | Cell lines | Images | What is unseen |
|---|---|---|---|---|---|
| IID | 59 | 21 | 12 | 5,900 | nothing |
| Compositional Shift | 64 | 16 | 16 | 6,400 | the (structure, cell) pairing |
| Cell-Line Shift | 59 | 22 | 8 | 5,900 | the cell line |
| Organelle-Set Shift | 54 | 29 | 6 | 5,400 | the structure label |
| Source Shift (external) | 13 | 3 | 5 | 1,158 | the acquisition source |

## Training

```bash
export MICRO_DATA_ROOT=/data2/ryr           # or wherever the datasets live
ENABLE_VLM_MIX=1 ENABLE_AUG_MIX=1 bash train.sh
```

That single command trains, evaluates all seven splits every epoch, and keeps the best checkpoint per metric. On 8× 72 GB GPUs the released checkpoint (epoch 15) is reached in ≈25–28 h; the full 30-epoch schedule takes ≈55 h.

The recipe (matching the paper): 12,000 updates, AdamW, global batch 8,192, peak LR 4e-5, 2,500 warmup steps, cosine decay, weight decay 0.2, bf16, 224×224 inputs, 256-token text context, both towers unfrozen.

### Reproducing the numbers

Every split in the table above is evaluated inside the training loop at the end of each
epoch, so the canonical reproduction is a training run:

```bash
MICRO_DATA_ROOT=/path/to/data ENABLE_VLM_MIX=1 ENABLE_AUG_MIX=1 bash train.sh
# then read the per-epoch table the run writes:
#   $OUTPUT_DIR/final_result/epoch_results.csv     (all seven splits, every epoch)
```

The released checkpoint is **epoch 15** of that run — the epoch with the best mean Top-1
over the five classification splits (49.80%). Reproduce the table by reading row 15 of
`epoch_results.csv`, or by evaluating the released weights with
`src/open_clip_train/eval_checkpoint_tsv.py` when your classification data is in TSV form.

> A standalone "load this checkpoint, score these shards" evaluator is **TODO(release)** —
> today the classification splits can only be scored through the training entry point.
> Open an issue if you need it; it is the first item on the release roadmap.

### Method modules

Four independent on/off switches live at the top of `train.sh`:

| Switch | Default | Effect |
|---|---|---|
| `ENABLE_THREE_WAY_TEXT` | `1` | per-sample choice of L1 title / L2 long caption / L3 semantic rewrite (25/50/25) |
| `ENABLE_META_BALANCE` | `1` | MetaCLIP-style online class balancing over the metadata vocabulary |
| `ENABLE_VLM_MIX` | `0` | append the `wds_vlm` natural-image corpus |
| `ENABLE_AUG_MIX` | `0` | append the `wds_train_aug` rare-class corpus |

> ⚠️ **The paper's recipe has all four ON.** The released defaults leave the two data-mixing modules off so that the code runs with the base corpus alone; run with `ENABLE_VLM_MIX=1 ENABLE_AUG_MIX=1` to reproduce the published numbers. Toggling those two switches changes the training corpus, which invalidates the cached MetaCLIP class counts — the first run must either pass `METACLIP_FORCE_RECOUNT=1` or point `METACLIP_COUNTS_FILE` at a counts file built from the same corpus.

### Backbones

The same method runs on other backbones for the scale study — `train_DFN_{B16,L14,H14}.sh`, `train_OpenAI_{B16,L14}.sh`, `train_SigLIP2_*.sh`, `train_PECore_*.sh`. Only the backbone and output directory change; data paths come from `paths.sh` and weights are fetched from Hugging Face Hub on first run.

## Repository layout

```
train.sh                  canonical training + seven-way evaluation entry point
train_<BACKBONE>.sh       per-backbone wrappers (scale study)
paths.sh                  all dataset paths, derived from MICRO_DATA_ROOT
inference.py              zero-shot classification / embedding API
scripts/                  dataset downloader, example extractor, smoke test
semantic_caption/         L3 knowledge-level captions (structures, cell lines)
src/open_clip/            vendored open_clip fork (256-token context support)
src/open_clip_train/      training loop, evaluation, balancing, retrieval utilities
tools/                    checkpoint conversion for release
```

## Citation

```bibtex
@article{microclip2026,
  title   = {MicroCLIP: Connecting Text and Microscopy Images via Contrastive Language--Image Pretraining},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

`CITATION.cff` is machine-readable for GitHub's "Cite this repository" button.

## License and terms

- **Code**: Apache-2.0 — see `LICENSE`.
- **Model weights**: TBD — see `LICENSES.md`. Weights are fine-tuned from DFN2B-CLIP ViT-B/16, so the released weight license must be compatible with the base model's terms.
- **Data**: MicroPool-2M aggregates images from several public repositories whose terms differ per source; see `LICENSES.md` for the per-source table and the redistribution position. Where redistribution is not permitted, the dataset release ships the splitter, per-class manifests, and source identifiers instead of image bytes.

## Acknowledgements

MicroCLIP is built on [OpenCLIP](https://github.com/mlfoundations/open_clip) and initialized from [DFN2B-CLIP](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16). The MicroPool-2M corpus aggregates data from the Human Protein Atlas, the Image Data Resource, RxRx, the Broad Bioimage Benchmark Collection, and the original publications those repositories cite.
