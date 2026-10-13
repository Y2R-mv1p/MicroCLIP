# Microscopy examples and figure provenance

The cover, Abstract, original paper figures, and quantitative main table are retained. Selected examples from ../docs are presented as recorded evidence, with new layout and local-only interactions.

## Reused examples

- Retrieval: two supplied saved runs (epoch015, 1,000 candidate pairs), each displaying five ranked micrographs. Rankings and four-decimal cosine scores are preserved. Rank 1 is marked as the reference match according to supplied ground-truth metadata. The query preview contains its first two complete sentences verbatim; full queries are available in native expandable sections. This is recorded retrieval, not live inference. Cosine scores are not probabilities.
- Supervision: Actin filaments and Cytosol / U251MG and Mitochondria / A549. Each specimen has category, knowledge and detail views. Full annotations are copied verbatim. Knowledge previews use the first complete sentence; detail previews use the first two. Sample switching changes the image and all three text views together.
- Composition: Nucleus / U2OS, Nucleus / BJ, Endoplasmic reticulum / U2OS and Endoplasmic reticulum / BJ. The comparison illustrates supplied training-pair membership. It does not present model predictions for these images.
- Reused single-channel images are byte-identical to ../docs assets. No crop, recoloring, contrast adjustment, or generated imagery is applied. Cover pseudocolor remains display-only; model inputs are single-channel.

## Records

resources/microscopy-examples.json is copied from ../docs/resources/microscopy-examples.json and retains complete queries, annotations, original labels, ranks and cosine scores. assets/microscopy/manifest.json retains original image provenance. More source context is in ../docs/ASSET_SOURCES.md.

## Verification

Checked local resource references, unique element IDs, balanced HTML tags, script syntax, exact query and annotation content, image bytes and retrieval scores. Both sample selectors passed initial state, mouse click, arrow wrapping, Home, End, keyboard focus and ARIA-state checks.

Abstract and cover HTML are unchanged. Main-table values, methods and grouped headers are retained; a horizontal-scroll hint is added. Browser visual verification remains unavailable because no browser is connected in this session.

## Observation

- Source: manuscript main.tex, ?Observation: The Biological Grounding Gap?, and Figure/observation3.pdf.
- assets/paper-observation.pdf is a byte-for-byte copy of the original figure; assets/paper-observation.png is a 1,960 ? 1,060 rendering with only the empty right and bottom PDF margins removed; all four panels, labels, and values are preserved, with no recoloring.
- The section preserves the example identity, diagnostic protocol (10 categories, 30 images per split), counts and percentages, and the manuscript limitations. Diagnostic results are distinguished from the full benchmark.
