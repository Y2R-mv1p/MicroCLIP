# Licensing

This repository contains code, released model weights, and pointers to a dataset
built from third-party sources. Each has its own terms. **Nothing here should be
treated as final until the items marked TODO(release) are resolved.**

## 1. Code — Apache-2.0

All source code in this repository (including the vendored `open_clip` fork under
`src/open_clip/`) is licensed under Apache-2.0. See `LICENSE`.

The vendored fork derives from [OpenCLIP](https://github.com/mlfoundations/open_clip)
(MIT) and the CLIP BPE vocabulary (`src/open_clip/bpe_simple_vocab_16e6.txt.gz`,
MIT). Upstream notices are retained.

## 2. Model weights — TODO(release)

| Item | Status |
|---|---|
| `microclip_epoch015.pt` (released checkpoint) | License undecided |
| Base model | `apple/DFN2B-CLIP-ViT-B-16` on Hugging Face Hub |

**Action required before publishing weights:** confirm the base model's license terms
and any usage restrictions on derivatives. MicroCLIP is a full fine-tune of that model
(both towers), so it is a derivative work and cannot be relicensed more permissively
than the base model allows. Once checked, set the same license in:

- `LICENSE` (if it differs from Apache-2.0, add a `LICENSE-MODEL` file and reference it here)
- `MODEL_CARD.md` frontmatter (`license:`, `license_name:`, `license_link:`)
- the Hugging Face model repository settings
- the *Model weights* bullet in `README.md`

## 3. Data — per-source terms

MicroPool-2M aggregates images and annotations from public repositories. **Redistribution
rights differ per source**, so the dataset release must be composed source by source.
The images themselves are not part of this code repository.

| Source | Contribution | Redistribution status |
|---|---|---|
| Human Protein Atlas | fluorescence images + structure/cell labels | to confirm (HPA publishes its own terms; check the image-level license) |
| Image Data Resource (IDR) | fluorescence images + labels | to confirm (study-specific; IDR aggregates studies with differing licenses) |
| RxRx | fluorescence images + labels | to confirm (research-use terms) |
| Broad Bioimage Benchmark Collection (BBBC) | fluorescence images + labels | to confirm (per-collection terms) |
| Curated literature | figures + captions | to confirm (publisher/article-specific) |

Additional items to settle for the dataset release:

- **L2 long captions** are VLM-generated descriptions of individual micrographs. Confirm
  whether the generating model's terms restrict redistribution of its outputs.
- **L3 semantic captions** (`semantic_caption/*.tsv` in this repository) are generated
  annotations of established biology authored for this project; they are released with
  the dataset.
- **Byte-identical duplicates** were removed by content hash before splitting; the
  released shards contain one copy per unique image.
- **Uniform policy**: where a source does not permit redistribution of image bytes, ship
  the splitter, per-class manifests, and the source identifiers (accession / article /
  plate / well) so users can rebuild that portion from the original repository, and
  document it in the dataset card.
- The released evaluation pools and the retrieval pools (`data_retrieval/`) are derived
  from the same sources and are covered by the same terms.

An audit script that re-derives the disjointness and per-class-count assertions from the
released shards is part of the dataset release; run it after any redistribution change.

## 4. Third-party results quoted in this repository

The comparison table quotes published or reproduced numbers for CLIP, SigLIP, SigLIP 2,
PE-Core, DFN2B-CLIP, Cell-DINO, and Light My Cells. Those models and benchmarks remain
under their own licenses; the numbers are reported for comparison only.
