"""Export the top-1-retrieval-correct samples for one checkpoint.

For a single epoch checkpoint on the keep_extracted figure set, computes the full
image-text similarity matrix and writes, into an output directory:

  <out>/epoch{epoch:03d}/i2t_top1/   images whose top-1 image-to-text match is their
                                     own paired caption (copy of the original .jpg)
  <out>/epoch{epoch:03d}/t2i_top1/   images whose top-1 text-to-image match is their
                                     own paired image
  <out>/epoch{epoch:03d}/hits_<field>.jsonl
                                     one JSON object per correct sample, with the text,
                                     the top-1 score and the rank of the correct item
                                     in both directions
  <out>/epoch{epoch:03d}/summary.txt human-readable count + ids

Ranks are computed with the same argsort convention as eval_retrieval_keep300, so the
reported counts match that script's R@1 for the same checkpoint/text field.

Example
-------
PYTHONPATH=src python -m open_clip_train.export_retrieval_top1 \
    --epoch 20 \
    --output-dir retrieval \
    --device cuda:0
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import open_clip
from open_clip import get_input_dtype
from open_clip_train.eval_checkpoint_tsv import load_model_checkpoint
from open_clip_train.eval_retrieval_keep300 import (
    DEFAULT_CKPT_DIR,
    DEFAULT_IMAGES_DIR,
    DEFAULT_JSONL,
    DEFAULT_PRETRAINED,
    discover_pairs,
    encode_retrieval_features,
    load_text_records,
    load_valid_samples,
)
from open_clip_train.precision import get_autocast

_LOGGER = logging.getLogger("open_clip_train.export_retrieval_top1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--text-field", default="normalized_caption")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint; overrides --epoch.")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Export root (default: <project>/retrieval). A per-epoch subdirectory is created inside.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix appended to the per-epoch run dir, e.g. --tag full -> epoch020_full.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", default="amp_bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-context-length", type=int, default=256)
    parser.add_argument("--force-quick-gelu", action="store_true")
    parser.add_argument("--force-image-size", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--pretrained-image", action="store_true")
    parser.add_argument("--similarity-chunk-size", type=int, default=4096)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def top1_ranks(sim, dim):
    """Tie-aware 1-based rank of the diagonal match along ``dim`` (0: rows are queries)."""
    n = sim.shape[0]
    if dim == 0:
        # row i queries all columns; correct item is column i
        diag = torch.diagonal(sim)
        return ((sim > diag.view(-1, 1)).sum(dim=1) + 1).cpu().numpy()
    # column j queries all rows; correct item is row j
    diag = torch.diagonal(sim)
    return ((sim > diag.view(1, -1)).sum(dim=0) + 1).cpu().numpy()


def top1_index(sim, dim):
    """Index of the single best match along ``dim`` (row-major for queries)."""
    if dim == 0:
        return sim.argmax(dim=1).cpu().numpy()
    return sim.argmax(dim=0).cpu().numpy()


def main():
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        epoch = args.epoch
    else:
        checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{args.epoch:03d}.pt"
        epoch = args.epoch
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[2] / "retrieval"
    )
    run_dir = output_dir / (f"epoch{epoch:03d}" + (f"_{args.tag}" if args.tag else ""))
    i2t_dir = run_dir / "i2t_top1"
    t2i_dir = run_dir / "t2i_top1"
    i2t_dir.mkdir(parents=True, exist_ok=True)
    t2i_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "export.log", encoding="utf-8"),
        ],
    )
    _LOGGER.info("Checkpoint=%s epoch=%d text_field=%s", checkpoint_path, epoch, args.text_field)

    records = load_text_records(args.jsonl, [args.text_field])
    pairs = discover_pairs(Path(args.images_dir), records)
    pairs_by_id = {p["image_id"]: p["text"] for p in pairs}
    _LOGGER.info("Matched pairs=%d", len(pairs))

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        precision=args.precision,
        device=device,
        force_quick_gelu=args.force_quick_gelu,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    tokenizer = open_clip.get_tokenizer(
        args.model, cache_dir=args.cache_dir, context_length=args.force_context_length
    )

    ckpt, missing, unexpected = load_model_checkpoint(model, checkpoint_path, strict=not args.non_strict)
    _LOGGER.info(
        "Loaded checkpoint | file_epoch=%03d ckpt_epoch=%s missing=%d unexpected=%d",
        epoch,
        ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown",
        len(missing),
        len(unexpected),
    )

    start = time.time()
    samples, token_lengths = load_valid_samples(pairs, preprocess_val, tokenizer, args.text_field)
    num_truncated = sum(1 for n in token_lengths if n >= args.force_context_length)
    images, texts = encode_retrieval_features(model, samples, args.batch_size, device, args.precision)
    _LOGGER.info(
        "Encoded | samples=%d truncated_texts=%d encode_time=%.1fs",
        len(samples), num_truncated, time.time() - start,
    )

    # Full (N, N) cosine similarity on the GPU (ranking is scale-invariant w.r.t. logit_scale).
    sim = (images.to(device) @ texts.to(device).t()).float()
    n = sim.shape[0]

    i2t_rank = top1_ranks(sim, dim=0)          # per image: rank of its own text
    t2i_rank = top1_ranks(sim, dim=1)          # per text:  rank of its own image
    i2t_best = top1_index(sim, dim=0)          # per image: index of best text
    t2i_best = top1_index(sim, dim=1)          # per text:  index of best image
    diag = torch.diagonal(sim).cpu().numpy()

    i2t_hits = [i for i in range(n) if i2t_best[i] == i]
    t2i_hits = [i for i in range(n) if t2i_best[i] == i]
    _LOGGER.info(
        "Rank-1 correct | i2t=%d/%d (%.1f%%)  t2i=%d/%d (%.1f%%)",
        len(i2t_hits), n, 100.0 * len(i2t_hits) / n,
        len(t2i_hits), n, 100.0 * len(t2i_hits) / n,
    )

    hits = []
    for i in range(n):
        hits.append({
            "image_id": samples[i]["image_id"],
            "image_path": samples[i]["image_path"],
            "text_field": args.text_field,
            "text": pairs_by_id[samples[i]["image_id"]][args.text_field],
            "i2t_rank": int(i2t_rank[i]),
            "i2t_top1_is_correct": bool(i2t_best[i] == i),
            "t2i_rank": int(t2i_rank[i]),
            "t2i_top1_is_correct": bool(t2i_best[i] == i),
            "self_similarity": float(diag[i]),
        })

    # Copy the image files of correct samples into the per-direction folders.
    for i in i2t_hits:
        shutil.copy2(samples[i]["image_path"], i2t_dir / (samples[i]["image_id"] + ".jpg"))
    for i in t2i_hits:
        shutil.copy2(samples[i]["image_path"], t2i_dir / (samples[i]["image_id"] + ".jpg"))
    _LOGGER.info("Copied images: i2t_top1=%d t2i_top1=%d", len(i2t_hits), len(t2i_hits))

    hits_path = run_dir / f"hits_{args.text_field}.jsonl"
    with hits_path.open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    _LOGGER.info("Wrote %s", hits_path)

    summary_path = run_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"epoch: {epoch}\n")
        f.write(f"text_field: {args.text_field}\n")
        f.write(f"num_samples: {n}\n")
        f.write(f"num_truncated_texts: {num_truncated}\n")
        f.write(f"i2t_top1_correct: {len(i2t_hits)} ({100.0 * len(i2t_hits) / n:.2f}%)\n")
        f.write(f"t2i_top1_correct: {len(t2i_hits)} ({100.0 * len(t2i_hits) / n:.2f}%)\n")
        f.write("i2t_top1 image_ids:\n")
        f.write("\n".join(samples[i]["image_id"] for i in i2t_hits) + "\n")
        f.write("t2i_top1 image_ids:\n")
        f.write("\n".join(samples[i]["image_id"] for i in t2i_hits) + "\n")
    _LOGGER.info("Wrote %s", summary_path)
    _LOGGER.info("Export complete -> %s", run_dir)


if __name__ == "__main__":
    main()
