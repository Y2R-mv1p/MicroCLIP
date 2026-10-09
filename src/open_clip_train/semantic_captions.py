"""Load biological semantic captions and build merged captions for text-text contrastive learning.

The training samples are keyed by ``canonical_organelle`` (a biological structure or marker,
possibly a compound joined by ``_and_``) and ``canonical_cell`` (a cell line). The structure
semantic captions and the cell semantic captions live in two separate TSVs:

* ``A_structures_split.tsv``       columns: structure, count, prompt, semantic_caption
* ``cell_semantic_recaption_manifest.tsv`` columns: cell_line, prompt, semantic_caption

A compound structure such as ``Actin_filaments_and_Cytosol`` is decomposed on ``_and_`` and each
component's caption is looked up and concatenated, then the cell caption is appended. The result
is the *merged* semantic caption used as the positive target for the class name.
"""
import csv
from pathlib import Path


class SemanticCaptionLookup:
    def __init__(self, structure_tsv, cell_tsv, and_sep="_and_"):
        self.and_sep = and_sep
        self.structure_captions = self._load_tsv(structure_tsv, "structure", "semantic_caption")
        self.cell_captions = self._load_tsv(cell_tsv, "cell_line", "semantic_caption")

    @staticmethod
    def _load_tsv(path, key_col, caption_col):
        table = {}
        with Path(path).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get(key_col) or "").strip()
                caption = (row.get(caption_col) or "").strip()
                if key and caption:
                    table[key] = caption
        return table

    def structure_caption(self, structure):
        if not structure:
            return ""
        if structure in self.structure_captions:
            return self.structure_captions[structure]
        if self.and_sep in structure:
            parts = [part.strip() for part in structure.split(self.and_sep) if part.strip()]
            captions = [self.structure_captions[part] for part in parts if part in self.structure_captions]
            if captions:
                return " ".join(captions)
        return ""

    def cell_caption(self, cell):
        if not cell:
            return ""
        return self.cell_captions.get(cell, "")

    def build(self, structure, cell):
        """Merged semantic caption: structure caption(s) followed by the cell caption."""
        parts = [part for part in (self.structure_caption(structure), self.cell_caption(cell)) if part]
        return " ".join(parts) if parts else ""
