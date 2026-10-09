#!/usr/bin/env python3
"""Convert a MicroCLIP training checkpoint into a release-ready weight file.

Training writes ``checkpoints/{best_*.pt,epochs/epoch_XXX.pt}``. Those files
bundle optimizer-adjacent metadata alongside the parameters, so a released
checkpoint should be re-serialized to a plain state_dict (optionally fp16)
before upload.

Usage
-----
    # fp32, plain torch state_dict (~600 MB for ViT-B/16)
    python tools/convert_checkpoint.py epoch_015.pt -o microclip_epoch015.pt

    # half precision safetensors, the usual Hugging Face Hub layout (~300 MB)
    python tools/convert_checkpoint.py epoch_015.pt -o microclip_epoch015.safetensors \
        --format safetensors --dtype fp16

The output loads with ``inference.py --checkpoint <file>`` unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch


def load_state_dict(path: str, prefix: str = "module.") -> Dict[str, torch.Tensor]:
    """Read a training checkpoint and return its bare state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict) or not state_dict:
        raise SystemExit(f"{path}: no state_dict found (top-level keys: {list(ckpt)[:8]})")
    if next(iter(state_dict)).startswith(prefix):
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def cast_dtype(state_dict: Dict[str, torch.Tensor], dtype: str) -> Dict[str, torch.Tensor]:
    """Cast floating-point tensors; integer/bool buffers are kept as-is."""
    if dtype == "fp32":
        target = torch.float32
    elif dtype == "fp16":
        target = torch.float16
    elif dtype == "bf16":
        target = torch.bfloat16
    else:
        raise SystemExit(f"unsupported --dtype {dtype}")
    return {
        k: (v.to(target) if v.is_floating_point() else v)
        for k, v in state_dict.items()
    }


def detach_contiguous(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """safetensors refuses tensors that share storage; clone the repeats."""
    out: Dict[str, torch.Tensor] = {}
    seen = set()
    for key, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        if tensor.data_ptr() in seen:
            tensor = tensor.clone()
        seen.add(tensor.data_ptr())
        out[key] = tensor
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Training checkpoint, e.g. checkpoints/epochs/epoch_015.pt")
    parser.add_argument("-o", "--output", required=True, help="Output file path.")
    parser.add_argument(
        "--format",
        choices=("pt", "safetensors"),
        default=None,
        help="Output container (default: inferred from the output suffix, else pt).",
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra metadata to store (safetensors only); repeatable.")
    args = parser.parse_args()

    fmt = args.format
    if fmt is None:
        fmt = "safetensors" if str(args.output).endswith(".safetensors") else "pt"

    state_dict = load_state_dict(args.checkpoint)
    n_params = sum(v.numel() for v in state_dict.values())
    state_dict = detach_contiguous(cast_dtype(state_dict, args.dtype))
    print(f"[convert] {args.checkpoint}: {len(state_dict)} tensors, {n_params / 1e6:.1f}M params -> {args.dtype}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise SystemExit("safetensors is required for --format safetensors (pip install safetensors)")
        metadata = {"format": "pt", "model": "MicroCLIP", "dtype": args.dtype}
        for item in args.metadata:
            key, _, value = item.partition("=")
            metadata[key] = value
        save_file(state_dict, str(out_path), metadata=metadata)
    else:
        torch.save({"state_dict": state_dict}, str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"[convert] wrote {out_path} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
