#!/usr/bin/env python3
"""MicroCLIP inference: zero-shot classification and image/text embedding.

Loads a released MicroCLIP checkpoint (ViT-B/16 initialized from DFN2B-CLIP,
256-token text context) and exposes both a CLI and a small Python API. No
dataset download is required -- point it at any microscopy images you have.

CLI
---
    python inference.py --checkpoint microclip_epoch015.pt \
        --images a.png b.png \
        --classes "a microscopy photo of Nucleus_of_HeLa" \
                  "a microscopy photo of Actin_filaments_of_HeLa"

Library
-------
    import inference
    model, preprocess, tokenizer = inference.load_model("microclip_epoch015.pt")
    img = inference.encode_images(model, preprocess, ["a.png"])
    txt = inference.encode_texts(model, tokenizer, ["a microscopy photo of Nucleus_of_HeLa"])

Both encoders return L2-normalized features, so similarity is a plain dot
product (see ``normalize=True`` below).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch

# The release vendors an open_clip fork (custom text context length) under src/.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import open_clip  # noqa: E402  (import after sys.path tweak)

DEFAULT_MODEL = "ViT-B-16"
DEFAULT_CONTEXT_LENGTH = 256


def pick_device(device: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, else CPU."""
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_checkpoint(checkpoint: str):
    """Read a training checkpoint (.pt) or a released safetensors file."""
    if str(checkpoint).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise SystemExit(
                "safetensors is required to load .safetensors checkpoints "
                "(pip install safetensors), or pass a .pt checkpoint instead."
            )
        return load_file(checkpoint, device="cpu")
    # weights_only=False: training checkpoints carry non-tensor metadata (epoch, metrics).
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def load_model(
    checkpoint: str,
    model_name: str = DEFAULT_MODEL,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    device: str = "auto",
    precision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    strict: bool = False,
) -> Tuple[torch.nn.Module, object, object]:
    """Build the model and load a MicroCLIP checkpoint.

    Returns ``(model, preprocess, tokenizer)``. ``preprocess`` is the image
    transform the model was trained with (224x224, CLIP normalization).
    """
    device_obj = pick_device(device)
    if precision is None:
        precision = "amp_bf16" if device_obj.type == "cuda" else "fp32"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
        precision=precision,
        device=device_obj,
        force_context_length=context_length,
        output_dict=True,
        cache_dir=cache_dir,
    )

    ckpt = _read_checkpoint(checkpoint)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unrecognized checkpoint layout in {checkpoint}: {type(state_dict)}")
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(
            f"[microclip] loaded {checkpoint}: {len(missing)} missing / "
            f"{len(unexpected)} unexpected keys",
            file=sys.stderr,
        )

    model.eval()
    tokenizer = open_clip.get_tokenizer(
        model_name, context_length=context_length, cache_dir=cache_dir
    )
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_images(
    model: torch.nn.Module,
    preprocess,
    paths: Sequence[str],
    device: str = "auto",
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode images to L2-normalized features, shape ``(len(paths), dim)`` on CPU."""
    from PIL import Image

    device_obj = pick_device(device)
    feats = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in chunk]
        ).to(device_obj)
        feats.append(model.encode_image(batch, normalize=True).float().cpu())
    if not feats:
        return torch.zeros(0, 1)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: str = "auto",
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode texts to L2-normalized features, shape ``(len(texts), dim)`` on CPU."""
    device_obj = pick_device(device)
    feats = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start + batch_size]).to(device_obj)
        feats.append(model.encode_text(tokens, normalize=True).float().cpu())
    if not feats:
        return torch.zeros(0, 1)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def zero_shot_classify(
    model: torch.nn.Module,
    preprocess,
    tokenizer,
    image_paths: Sequence[str],
    class_names: Sequence[str],
    device: str = "auto",
    batch_size: int = 32,
) -> Tuple[torch.Tensor, list]:
    """Zero-shot classify images against a list of class strings.

    ``class_names`` are used verbatim as text prompts -- pass the full title,
    e.g. ``"a microscopy photo of Nucleus_of_HeLa"``. Returns
    ``(probabilities, top1_indices)``.
    """
    image_features = encode_images(model, preprocess, image_paths, device, batch_size)
    text_features = encode_texts(model, tokenizer, class_names, device)
    logits = model.logit_scale.exp().cpu() * image_features @ text_features.t()
    probs = logits.softmax(dim=-1)
    return probs, probs.argmax(dim=-1).tolist()


def _format_predictions(
    probs: torch.Tensor, class_names: Sequence[str], image_paths: Sequence[str], topk: int
) -> str:
    lines = []
    for path, row in zip(image_paths, probs):
        k = min(topk, row.numel())
        values, indices = row.topk(k)
        ranked = ", ".join(
            f"{class_names[i]} ({v * 100:.1f}%)" for v, i in zip(values.tolist(), indices.tolist())
        )
        lines.append(f"{path}\n    {ranked}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MicroCLIP zero-shot classification and embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # classify against three candidate classes\n"
            "  python inference.py -c m.pt -i a.png b.png \\\n"
            "      --classes 'a microscopy photo of Nucleus_of_HeLa' \\\n"
            "                'a microscopy photo of Actin_filaments_of_HeLa'\n\n"
            "  # dump embeddings for downstream use\n"
            "  python inference.py -c m.pt -i *.png --save-features feats.pt\n"
        ),
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="MicroCLIP checkpoint (.pt).")
    parser.add_argument("--images", "-i", nargs="+", required=True, help="Image file(s).")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class prompt strings, e.g. 'a microscopy photo of Nucleus_of_HeLa'.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"open_clip model name (default {DEFAULT_MODEL}).")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    parser.add_argument("--precision", default=None, help="amp_bf16 (GPU default) | fp32 | fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--save-features", default=None, help="Write image features to this .pt file.")
    parser.add_argument("--save-text-features", default=None, help="Write text features to this .pt file.")
    parser.add_argument("--non-strict", action="store_true", help="Load the checkpoint ignoring key mismatches.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)

    model, preprocess, tokenizer = load_model(
        args.checkpoint,
        model_name=args.model,
        context_length=args.context_length,
        device=str(device),
        precision=args.precision,
        cache_dir=args.cache_dir,
        strict=not args.non_strict,
    )
    print(f"[microclip] {args.model} | ctx={args.context_length} | device={device} | {args.checkpoint}")

    if args.classes:
        probs, top1 = zero_shot_classify(
            model, preprocess, tokenizer, args.images, args.classes, str(device), args.batch_size
        )
        print(_format_predictions(probs, args.classes, args.images, args.topk))
        print("\nTop-1: " + ", ".join(f"{p} -> {args.classes[i]}" for p, i in zip(args.images, top1)))
    else:
        image_features = encode_images(model, preprocess, args.images, str(device), args.batch_size)
        print(f"[microclip] image features: {tuple(image_features.shape)}")
        if args.save_features:
            torch.save({"paths": list(args.images), "features": image_features}, args.save_features)
            print(f"[microclip] wrote {args.save_features}")

    if args.save_text_features:
        text_features = encode_texts(model, tokenizer, args.classes or [], str(device))
        torch.save({"texts": list(args.classes or []), "features": text_features}, args.save_text_features)
        print(f"[microclip] wrote {args.save_text_features}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
