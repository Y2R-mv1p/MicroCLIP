#!/usr/bin/env bash
# =============================================================================
# Fast sanity check: does this checkout import, resolve its datasets and run a
# forward pass? No GPU, no dataset download and no checkpoint required.
#
#   bash scripts/smoke_test.sh              # code paths only (runs in CI)
#   bash scripts/smoke_test.sh --with-data  # also resolve every dataset path
#
# Exits non-zero on the first failure.
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

WITH_DATA=0
[[ "${1:-}" == "--with-data" ]] && WITH_DATA=1

PYTHON="${PYTHON:-python}"
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

echo "[1/5] python + torch"
"${PYTHON}" - <<'EOF' || fail "torch import"
import torch
print(f"  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
EOF
pass "torch"

echo "[2/5] vendored open_clip (src/)"
"${PYTHON}" - <<'EOF' || fail "vendored open_clip import"
import open_clip, inspect
assert hasattr(open_clip, "create_model_and_transforms"), "vendored open_clip not on PYTHONPATH"
params = inspect.signature(open_clip.create_model_and_transforms).parameters
assert "force_context_length" in params, "this fork must support force_context_length"
print(f"  open_clip {getattr(open_clip, '__version__', 'vendored')} with force_context_length")
EOF
pass "open_clip fork"

echo "[3/5] model construction + forward pass (CPU, random init)"
"${PYTHON}" - <<'EOF' || fail "forward pass"
import torch, open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained=None, precision="fp32", device="cpu",
    force_context_length=256, output_dict=True,
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-16", context_length=256)
with torch.no_grad():
    img = model.encode_image(torch.randn(2, 3, 224, 224), normalize=True)
    txt = model.encode_text(tokenizer(["a microscopy photo of Nucleus_of_HeLa"]), normalize=True)
assert img.shape[0] == 2 and txt.shape[0] == 1, (img.shape, txt.shape)
assert img.shape[1] == txt.shape[1], (img.shape, txt.shape)  # shared embedding dim
print(f"  image {tuple(img.shape)} | text {tuple(txt.shape)}")
EOF
pass "forward pass"

echo "[4/5] shell entry points parse"
for script in train.sh paths.sh; do
  bash -n "${script}" || fail "${script} syntax"
done
bash -n scripts/download_datasets.sh || fail "download_datasets.sh syntax"
pass "shell syntax"

echo "[5/5] dataset paths"
if [[ "${WITH_DATA}" == "1" ]]; then
  # shellcheck disable=SC1091
  source ./paths.sh
  for var in WDS_TRAIN WDS_VLM WDS_AUG IID_DATA PARTIAL_OOD_DATA OOD_DATA CELL_OOD_DATA STRUCTURE_OOD_DATA RETRIEVAL_DATA RETRIEVAL_DATA_1000; do
    value="${!var:-}"
    [[ -n "${value}" ]] || fail "${var} is unset (set MICRO_DATA_ROOT)"
    count=$(eval "ls ${value}" 2>/dev/null | wc -l)
    [[ "${count}" -gt 0 ]] || fail "${var} does not resolve: ${value}"
    printf '  ok   %-20s %s file(s)\n' "${var}" "${count}"
  done
else
  echo "  skipped (pass --with-data to resolve every path from MICRO_DATA_ROOT)"
fi

echo
echo "smoke test passed."
