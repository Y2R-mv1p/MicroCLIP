# Source access and redistribution notes

This package hosts no source images. It downloads from official locations or
accepts files that the user obtained from the official provider. Public access
does not itself grant redistribution rights; users remain responsible for each
source's terms.

The exact paper benchmark still contains three Cell Tracking Challenge images
in the 1,000-pair retrieval pool. CTC forbids cloning datasets or their parts
and requires permission for public non-CTC scientific use. The builder therefore
does not fetch these images unless `--authorized-source fluo` is supplied after
permission has actually been obtained.

Sources requiring local action are reported by `build.sh --stage plan`:

| Source | Access/action | Terms recorded for this release |
|---|---|---|
| DynamicNuclearNet | Download with a DeepCell account/API key, then supply the official files locally | Modified Apache terms; non-commercial academic use and attribution |
| FMD | Use the authors' official Google Drive/download script and supply locally | Repository software is MIT; the release does not assume that this alone licenses all image data |
| BPAEC | Use the source link cited by the paper and supply the selected images locally | Source provenance must be confirmed before redistributing images |
| BioSR | Download from the official Figshare record and supply locally | CC BY 4.0 according to the source record |
| SR-CACO-2 | Use the official project download and supply locally | CC BY-NC-SA 4.0 |
| CTC Fluo | Obtain any permission required for the intended use, download from CTC, and pass `--authorized-source fluo` | Citation required; public non-CTC use requires permission; cloning is forbidden |

Automatically located sources retain their own terms. Important examples are:

- Human Protein Atlas: CC BY 4.0, with HPA's image-specific credit and citation
  requirements.
- IDR studies: study-specific licenses. In particular, IDR0088 is CC BY-NC 4.0;
  IDR0050, IDR0037, IDR0093 and IDR0133 record CC BY 4.0; IDR0080 records CC0.
- RxRx1, BBBC017, HeLa10Class and other non-commercial sources remain subject
  to their upstream restrictions.
- Light My Cells S-BIAD1047 records CC BY 4.0.
- CPJUMP1 is fetched from the public Cell Painting Gallery bucket.

Authoritative upstream pages:

- HPA licence: <https://www.proteinatlas.org/about/licence>
- IDR study metadata: <https://github.com/IDR>
- CTC conditions: <https://celltrackingchallenge.net/datasets/>
- DeepCell terms: <https://datasets.deepcell.org/terms>
- Light My Cells: <https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1047>
- FMD: <https://github.com/yinhaoz/denoising-fluorescence>
- SR-CACO-2: <https://github.com/sbelharbi/sr-caco-2>

The repository's code can be licensed independently from the third-party data.
Do not apply the code license to reconstructed images or imply that this package
relicenses upstream material.
