#!/usr/bin/env python3
"""Evaluate a MicroCLIP checkpoint on the six paper evaluation datasets."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile


CLASSIFICATION = OrderedDict([
    ("iid", "evaluation/iid/shard-000000.tar"),
    ("compositional_shift", "evaluation/compositional_shift/shard-000000.tar"),
    ("cell_line_shift", "evaluation/cell_line_shift/shard-000000.tar"),
    ("organelle_set_shift", "evaluation/organelle_set_shift/shard-000000.tar"),
    ("source_shift", "evaluation/source_shift/shard-000000.tar"),
])
RETRIEVAL = "evaluation/retrieval_1000/shard-000000.tar"


def records(path: Path):
    """Yield image bytes, .txt, and .json in key order from a verified shard."""
    grouped = OrderedDict()
    with tarfile.open(path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                raise ValueError(f"Unexpected non-file member: {member.name}")
            key, ext = member.name.rsplit(".", 1)
            item = grouped.setdefault(key, {})
            if ext in item:
                raise ValueError(f"Duplicate member: {member.name}")
            item[ext] = tf.extractfile(member).read()
    for key, item in grouped.items():
        image_exts = set(item) & {"jpg", "jpeg", "png"}
        if len(image_exts) != 1 or "txt" not in item or "json" not in item:
            raise ValueError(f"Incomplete record {key} in {path}")
        ext = image_exts.pop()
        yield item[ext], item["txt"].decode("utf-8"), json.loads(item["json"])


def encode_images(model, preprocess, samples, device, batch_size):
    import torch
    from PIL import Image

    chunks = []
    batch = []
    for image_bytes, _, _ in samples:
        batch.append(preprocess(Image.open(BytesIO(image_bytes)).convert("RGB")))
        if len(batch) == batch_size:
            chunks.append(model.encode_image(torch.stack(batch).to(device), normalize=True).float().cpu())
            batch = []
    if batch:
        chunks.append(model.encode_image(torch.stack(batch).to(device), normalize=True).float().cpu())
    return torch.cat(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="verified MicroPool eval root")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--microclip-repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if not (args.root / "verification.json").is_file():
        raise SystemExit("Run build.sh --stage verify first; verification.json is missing.")
    sys.path[:0] = [str(args.microclip_repo), str(args.microclip_repo / "src")]
    import torch
    import inference

    device = inference.pick_device(args.device)
    model, preprocess, tokenizer = inference.load_model(
        str(args.checkpoint), device=str(device), strict=True
    )
    result = {"checkpoint": str(args.checkpoint.resolve()), "classification": {}}

    with torch.inference_mode():
        for name, relative in CLASSIFICATION.items():
            samples = list(records(args.root / relative))
            labels = [text for _, text, _ in samples]
            classes = sorted(set(labels))
            class_index = {label: index for index, label in enumerate(classes)}
            image_features = encode_images(model, preprocess, samples, device, args.batch_size)
            text_features = inference.encode_texts(
                model, tokenizer, classes, str(device), args.batch_size
            )
            predictions = (image_features @ text_features.T).argmax(dim=1)
            targets = torch.tensor([class_index[label] for label in labels])
            accuracy = (predictions == targets).float().mean().item() * 100
            result["classification"][name] = {
                "images": len(samples), "classes": len(classes), "top1": accuracy
            }
            print(f"{name:24} Top-1 {accuracy:6.2f}% ({len(samples)} images)", flush=True)

        samples = list(records(args.root / RETRIEVAL))
        captions = [metadata["long_caption"] for _, _, metadata in samples]
        image_features = encode_images(model, preprocess, samples, device, args.batch_size)
        text_features = inference.encode_texts(
            model, tokenizer, captions, str(device), args.batch_size
        )
        similarity = image_features @ text_features.T
        truth = torch.arange(len(samples))
        i2t = (similarity.argmax(dim=1) == truth).float().mean().item() * 100
        t2i = (similarity.argmax(dim=0) == truth).float().mean().item() * 100
        result["retrieval1000"] = {"pairs": len(samples), "i2t_r1": i2t, "t2i_r1": t2i}
        print(f"retrieval1000            I2T R@1 {i2t:6.2f}%  T2I R@1 {t2i:6.2f}%", flush=True)

    output = args.output or args.root / "paper_evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
