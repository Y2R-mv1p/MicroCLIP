# Licensing scope

## Project code

MicroCLIP's project code is released under Apache-2.0; see [LICENSE](LICENSE).
Third-party components retain their upstream licenses and copyright notices.

## Third-party code

The vendored implementation under `src/open_clip/` and `src/open_clip_train/`
is based on [OpenCLIP](https://github.com/mlfoundations/open_clip).
OpenCLIP is MIT-licensed; see its [upstream license](https://github.com/mlfoundations/open_clip/blob/main/LICENSE).
The model and tokenizer also incorporate code originating from
[OpenAI CLIP](https://github.com/openai/CLIP), which is MIT-licensed; see its
[upstream license](https://github.com/openai/CLIP/blob/main/LICENSE).
The project's Apache-2.0 license does not replace these upstream notices.

## Model weights

MicroCLIP is initialized from
[DFN2B-CLIP ViT-B/16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16).
Model checkpoints are distributed separately from this code repository.
Refer to the license and model card accompanying a checkpoint for its usage terms;
the code license alone does not specify a checkpoint's license.

## Data and annotations

The code repository includes evaluation manifests, reconstruction tools, and
biological annotation tables. Third-party source images and the reconstructed
image datasets are obtained separately and remain subject to their providers' terms.

For the bundled MicroPool package, the project-authored annotations and evaluation
metadata are released under CC BY 4.0, as described in
[MicroPool/LICENSES.md](MicroPool/LICENSES.md). This does not cover third-party
images or upstream metadata. See the [source instructions](MicroPool/docs/SOURCES.md)
for access and attribution information.

Human Protein Atlas, IDR, RxRx, BBBC, Light My Cells, and literature-derived material
retain the terms applicable to the original source or study. This repository does
not grant additional redistribution rights for those materials.

## Comparisons and acknowledgements

Referenced models, benchmarks, and third-party results retain their original
licenses and attribution. Reporting comparison results does not relicense the
underlying software, weights, or data.
