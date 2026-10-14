# MicroCLIP project page

Open index.html directly in a browser. No build, CDN, or installation is required.

The cover markup and styling come from ../docs. The rest of the page is rebuilt around the complete manuscript Abstract and five original paper figures. The page includes responsive navigation, enlarged figure dialogs, downloadable figure PDFs, links to the arXiv paper and PDF, and BibTeX.

## Content sources

- Abstract: sibling manuscript project, main.tex, active abstract environment; comments excluded. Text preserved verbatim with emphasis added.
- Figures: manuscript Figure/teaser.pdf, framework.pdf, dataset_visual5.pdf, retrieval.pdf, and 3dpca2.pdf. PNG derivatives rendered at 2,400 pixels wide, preserving full figure boundaries.
- Cover microscopy and fluorescence mosaic: ../docs/assets; provenance in ../docs/ASSET_SOURCES.md. Hero pseudocolor is for visualization; model input is single-channel.
- Metrics: ../README.md, manuscript-reported values. The main table is shown directly before qualitative results and matches main.tex Table classification_retrieval, including all eight methods, parameter counts, grouped headings, and best/second-best markings.
- Paper: https://arxiv.org/abs/2610.17866; PDF: https://arxiv.org/pdf/2610.17866. The page links to the public arXiv version. The older local manuscript PDF is retained as an asset but is no longer linked from the page.

## Verification

All local HTML resource references and section anchors resolve. Abstract text matches the active LaTeX Abstract exactly. Script syntax checked. All five rendered figures visually inspected.

Browser rendering, mobile layout, and interactive behavior still need visual review: this session had no connected browser.

## Reading layout

The full Abstract is divided into three paragraphs, with a short lead and 20 px desktop / 18 px mobile body text. Chapter labels explicitly identify Overview, Dataset, Method, Results, and Paper & Code; descriptions remain visible on mobile. Main-table numbers use 18 px desktop / 17 px mobile type, with horizontal scrolling on narrower viewports. Captions, navigation, notes, and cover metadata use enlarged type. Abstract wording and all main-table values are preserved.

## Microscopy examples

The page now includes two saved text-to-image retrieval cases, two three-level supervision samples, a four-image seen/held-out composition comparison, and a three-step method guide before the full framework. Source records and rendering meaning are documented in ASSET_SOURCES.md. Example switches work with mouse and keyboard; without JavaScript both cases remain readable. The cover and manuscript Abstract are retained, and the main table remains directly visible.

## Observation and citation

The page order is Abstract, Overview, Observation, Dataset, Method, and Results. Recorded retrieval examples are part of Results. Observation uses the original observation3.pdf and a PNG with excess blank margins removed. Its diagnostic results are separate from the main benchmark. The figure supports enlargement and a PDF download. Cite MicroCLIP displays the BibTeX by default; readers can collapse it, copy it, or download the matching .bib file.

The Abstract begins directly with the manuscript text. Overview describes recognition and retrieval in concrete terms. Results keeps qualitative retrieval and recorded examples together, followed by a dedicated image-text embedding visualization subsection.
