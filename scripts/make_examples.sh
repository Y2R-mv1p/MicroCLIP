#!/usr/bin/env bash
# =============================================================================
# Extract a few example micrographs from a WebDataset shard so the quickstart in
# the README can be run end to end.
#
#   bash scripts/make_examples.sh <shard.tar> [count] [outdir]
#
# Defaults: 3 images from training/microscopy/shard-000000.tar
# written to examples/, together with the matching class titles.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${SCRIPT_DIR}"

if [[ -z "${1:-}" && -z "${MICRO_DATA_ROOT:-}" ]]; then
  echo "usage: bash scripts/make_examples.sh /path/to/shard-000000.tar [count] [output-dir]" >&2
  echo "   or: MICRO_DATA_ROOT=/path/to/data bash scripts/make_examples.sh" >&2
  exit 1
fi
SHARD="${1:-${MICRO_DATA_ROOT}/training/microscopy/shard-000000.tar}"
COUNT="${2:-3}"
OUTDIR="${3:-${SCRIPT_DIR}/examples}"

[[ -f "${SHARD}" ]] || { echo "shard not found: ${SHARD}" >&2; exit 1; }
mkdir -p "${OUTDIR}"

# The shard stores <key>.jpg / <key>.txt pairs; the .txt is the L1 title.
mapfile -t KEYS < <(tar -tf "${SHARD}" | grep '\.jpg$' | head -n "${COUNT}")
[[ "${#KEYS[@]}" -gt 0 ]] || { echo "no .jpg members in ${SHARD}" >&2; exit 1; }

: > "${OUTDIR}/labels.tsv"
for key in "${KEYS[@]}"; do
  stem="${key%.jpg}"
  tar -xOf "${SHARD}" "${key}" > "${OUTDIR}/${stem}.jpg"
  title=$(tar -xOf "${SHARD}" "${stem}.txt" | tr -d '\n')
  printf '%s\t%s\n' "${stem}.jpg" "${title}" >> "${OUTDIR}/labels.tsv"
  printf '  %s  ->  %s\n' "${stem}.jpg" "${title}"
done

cat <<EOF

Wrote ${#KEYS[@]} image(s) to ${OUTDIR}
Try:

    python inference.py --checkpoint <microclip_epoch015.pt> \\
      $(for k in "${KEYS[@]}"; do printf '%s ' "${OUTDIR}/${k}"; done) \\
      --classes "$(head -1 "${OUTDIR}/labels.tsv" | cut -f2)"

NOTE: check the source-image terms (see LICENSES.md) before committing these
files to a public repository; they are generated locally, not shipped.
EOF
