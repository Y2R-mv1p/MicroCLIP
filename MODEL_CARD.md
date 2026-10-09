---
license: other
license_name: to-be-determined
license_link: LICENSES.md
base_model: apple/DFN2B-CLIP-ViT-B-16
datasets:
  - TODO_RELEASE_ORG/MicroPool-2M
library_name: open_clip
pipeline_tag: zero-shot-image-classification
tags:
  - microscopy
  - fluorescence-microscopy
  - cell-biology
  - clip
  - image-text-retrieval
---

# MicroCLIP

MicroCLIP is a 150M-parameter vision–language model for single-channel fluorescence
microscopy. It is fine-tuned from [DFN2B-CLIP ViT-B/16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16)
on **MicroPool-2M** — 2.5M microscopy images paired with category titles, per-image
descriptions, and knowledge-level biological captions — using the CLIP contrastive
objective with three-level text supervision, composition-aware balancing, and rare-class
augmentation.

The released checkpoint is epoch 15 of the canonical run, selected by mean Top-1 accuracy
over the five classification splits.

## Results

Zero-shot classification Top-1 (%) and retrieval Recall@1 (%) on the 1,000-pair pool.

| Method | Params | IID | Compositional | Cell-Line | Organelle-Set | Source Shift | I2T | T2I |
|---|---|---|---|---|---|---|---|---|
| CLIP (OpenAI ViT-B/16) | 150M | 2.97 | 5.12 | 1.80 | 2.54 | 8.38 | 0.7 | 0.2 |
| SigLIP (ViT-B/16) | 203M | 3.05 | 1.88 | 1.07 | 1.26 | 9.33 | 0.2 | 0.2 |
| SigLIP 2 (ViT-B/16) | 375M | 3.08 | 2.28 | 3.59 | 1.11 | 10.79 | 0.2 | 0.3 |
| PE-Core (ViT-B/16) | 448M | 3.34 | 3.78 | 7.63 | 2.19 | 22.54 | 0.4 | 0.0 |
| DFN2B-CLIP (ViT-B/16) | 150M | 1.44 | 1.41 | 4.32 | 0.81 | 16.49 | 0.5 | 0.1 |
| **MicroCLIP** | **150M** | **98.46** | **67.70** | **34.07** | **46.13** | **36.53** | **61.5** | **63.3** |

## Usage

```python
# Clone the code repository first: the model needs its vendored open_clip fork
# (256-token text context) and the checkpoint loader.
import inference

model, preprocess, tokenizer = inference.load_model("microclip_epoch015.pt", device="cuda")

classes = [
    "a microscopy photo of Nucleus_of_HeLa",
    "a microscopy photo of Actin_filaments_of_HeLa",
    "a microscopy photo of Mitochondria_of_HeLa",
]
probs, top1 = inference.zero_shot_classify(
    model, preprocess, tokenizer, ["my_cell.png"], classes, device="cuda"
)
print(classes[top1[0]], float(probs[0, top1[0]]))
```

Class prompts are full titles of the form `a microscopy photo of {structure}_of_{cell}`.
Both encoders return L2-normalized features, so dot products are cosine similarities —
the model can be used as a frozen encoder for retrieval, clustering, or linear probes.

## Training

- **Initialization**: DFN2B-CLIP ViT-B/16 (both towers), all parameters fine-tuned.
- **Data**: 2,498,617 microscopy images + 609,740 natural-image records + 175,516 rare-class augmented copies.
- **Objective**: CLIP contrastive loss, three-level text sampling (25% title / 50% long caption / 25% semantic rewrite).
- **Schedule**: 12,000 updates, AdamW, global batch 8,192, peak LR 4e-5, 2,500 warmup steps, cosine decay, weight decay 0.2.
- **Precision / hardware**: bf16, 8× NVIDIA RTX PRO 5000 (72 GB), ≈55 h for the full 30-epoch schedule; epoch 15 (the released checkpoint) at ≈25–28 h.
- **Inputs**: 224×224 images, 256-token text context.

## Intended use and limitations

- **Intended for**: zero-shot classification, cross-modal retrieval, and as a frozen feature extractor for fluorescence microscopy images. The label vocabulary is the MicroPool-2M one (537 structure labels × 109 cell lines); recognizing structures or cell lines outside it requires fine-tuning.
- **Not for**: clinical or diagnostic use. This is a research model.
- **Unseen-label behavior**: the internal 100-class "OOD" split (labels never seen in training) sits near chance for MicroCLIP as well as for every baseline; it is treated as a noise metric and is deliberately excluded from the table above. Do not use it as a benchmark. Generalization claims in this release concern *recombinations* of seen structures and cell lines (Compositional / Cell-Line / Organelle-Set shift), not unseen biological entities.
- **Imaging assumptions**: single-channel fluorescence micrographs processed as RGB; images with very different modalities (brightfield, histology) are out of distribution.
- **Prompt sensitivity**: classification accuracy assumes prompts follow the training template. Rewriting prompts moves the text distribution away from training.

## Citation

See `CITATION.cff` in the code repository. Paper and author list: TBD.

## License

Code: Apache-2.0. Weights: TBD — see `LICENSES.md`; they are derived from
DFN2B-CLIP ViT-B/16, so the released license must be compatible with the base model's terms.
