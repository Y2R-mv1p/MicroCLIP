"""Image-text retrieval evaluation (I2T + T2I, R@1/R@5/R@10) for MicroCLIP checkpoints
on the keep_extracted 1000-figure set.

Data layout
-----------
  images : <images_dir>/<image_id>.jpg          (e.g. PMC12958280_fig3.jpg)
  texts  : <jsonl>                              (one JSON record per image, keyed by "image_id")

The image and its text are matched 1:1 through ``image_id``, so every query has exactly
one relevant item and standard R@k retrieval metrics apply. For each requested text field
(default ``normalized_caption``) and each epoch checkpoint, the script computes
image-to-text and text-to-image ranks over the full 296 x 296 similarity matrix and writes
R@1/R@5/R@10 (plus median/mean rank) to a summary CSV.

Example
-------
PYTHONPATH=src python -m open_clip_train.eval_retrieval_1000 \
    --checkpoint-dir log_vlm_MicroPool811/checkpoints/epochs \
    --text-fields normalized_caption \
    --output-dir log_vlm_MicroPool811/eval_retrieval_1000
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import PIL.Image

import open_clip
from open_clip import get_input_dtype
from open_clip_train.eval_checkpoint_tsv import load_model_checkpoint
from open_clip_train.precision import get_autocast

_LOGGER = logging.getLogger("open_clip_train.eval_retrieval_1000")

DEFAULT_IMAGES_DIR = "/data2/ryr/workspace/micro-idea/MicroCLIP-M811-semantic_meta_vlm0.7_mixed/log_vlm_MicroPool811/eval_retrieval_1000"
DEFAULT_JSONL = "/data2/ryr/data_retrieval/wash_data/visual_cleaned/keep_extracted/keep.jsonl"
DEFAULT_CKPT_DIR = "/data2/ryr/workspace/micro-idea/MicroCLIP-M811-semantic_meta_vlm0.7_mixed/log_vlm_MicroPool811/checkpoints/epochs"
DEFAULT_PRETRAINED = "/data2/ryr/model_cache/DFN2B-CLIP-ViT-B-16/open_clip_pytorch_model.bin"

SUMMARY_FIELDS = [
    "epoch",
    "checkpoint",
    "text_field",
    "num_samples",
    "num_truncated_texts",
    "eval_seconds",
    "i2t_r1", "i2t_r5", "i2t_r10", "i2t_median_rank", "i2t_mean_rank",
    "t2i_r1", "t2i_r5", "t2i_r10", "t2i_median_rank", "t2i_mean_rank",
    "mean_r1", "mean_r5", "mean_r10",
]

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image-text retrieval on the keep_extracted figure set. "
            "Computes I2T and T2I R@1/R@5/R@10 for every epoch checkpoint."
        )
    )
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help="Backbone weights used to initialize before loading --checkpoint-dir snapshots.",
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Folder containing the figure images.")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL, help="keep.jsonl with one text record per image.")
    parser.add_argument(
        "--text-fields",
        default="normalized_caption",
        help="Comma-separated jsonl fields to use as the retrieval text (e.g. normalized_caption,caption,title).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CKPT_DIR,
        help="Directory holding per-epoch snapshots (glob below).",
    )
    parser.add_argument("--checkpoint-glob", default="epoch_*.pt")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Evaluate a single explicit checkpoint instead of globbing --checkpoint-dir.",
    )
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=None)
    parser.add_argument(
        "--eval-pretrained-only",
        action="store_true",
        help="Evaluate the backbone initialized from --pretrained without loading any checkpoint.",
    )
    parser.add_argument("--output-dir", default=None, help="Where to write CSVs/JSONs (default: log dir/eval_retrieval_1000).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", default="amp_bf16")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:0 (default: cuda:0 if available else cpu).")
    parser.add_argument("--force-context-length", type=int, default=256)
    parser.add_argument("--force-quick-gelu", action="store_true")
    parser.add_argument("--force-image-size", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--pretrained-image", action="store_true")
    parser.add_argument("--similarity-chunk-size", type=int, default=4096)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--save-details", action="store_true", help="Also write per-query top-10 matches JSON.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def setup_logging(output_dir):
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.handlers.clear()
    _LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    _LOGGER.addHandler(stream)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "eval_retrieval_1000.log", encoding="utf-8")
        fh.setFormatter(formatter)
        _LOGGER.addHandler(fh)


def load_text_records(jsonl_path, text_fields):
    """Return {image_id: {field: str}} for every record in the jsonl."""
    records = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = rec.get("image_id")
            if not image_id:
                continue
            records[image_id] = {field: str(rec.get(field, "")).strip() for field in text_fields}
    return records


def discover_pairs(images_dir, records):
    """Align image files with their jsonl text records via image_id (= filename stem)."""
    pairs = []
    for ext in IMAGE_EXTS:
        for img_path in sorted(images_dir.glob(ext)):
            image_id = img_path.stem
            if image_id not in records:
                _LOGGER.warning("Image %s has no text record in the jsonl; skipped.", img_path)
                continue
            pairs.append({"image_path": img_path, "image_id": image_id, "text": records[image_id]})
    return pairs


def load_valid_samples(pairs, preprocess, tokenizer, text_field):
    """Preload image tensors + text token ids, dropping unreadable images."""
    valid = []
    token_lengths = []
    for p in pairs:
        try:
            with PIL.Image.open(p["image_path"]) as im:
                im = im.convert("RGB")
            image = preprocess(im)
        except Exception as exc:
            _LOGGER.warning("Skipping unreadable image %s: %s", p["image_path"], exc)
            continue
        tokens = tokenizer([p["text"][text_field]])[0]
        token_lengths.append(int((tokens != 0).sum().item()))
        valid.append(
            {"image_id": p["image_id"], "image_path": str(p["image_path"]), "image": image, "text": tokens}
        )
    return valid, token_lengths


@torch.no_grad()
def encode_retrieval_features(model, samples, batch_size, device, precision):
    model.eval()
    autocast = get_autocast(precision, device_type=device.type)
    input_dtype = get_input_dtype(precision)
    image_feats = []
    text_feats = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = torch.stack([s["image"] for s in chunk]).to(device)
        texts = torch.stack([s["text"] for s in chunk]).to(device)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        with autocast():
            image_feats.append(model.encode_image(images, normalize=True).float().cpu())
            text_feats.append(model.encode_text(texts, normalize=True).float().cpu())
    if not image_feats:
        raise RuntimeError("No valid samples were encoded.")
    return torch.cat(image_feats, dim=0), torch.cat(text_feats, dim=0)


def ranks_from_similarity(query_features, candidate_features, chunk_size):
    """Rank (1-based) of the diagonal match for each query, computed in chunks."""
    ranks = []
    num_samples = query_features.shape[0]
    for start in range(0, num_samples, chunk_size):
        scores = query_features[start : start + chunk_size] @ candidate_features.t()
        targets = torch.arange(start, min(start + chunk_size, num_samples)).view(-1, 1)
        order = scores.argsort(dim=1, descending=True)
        rank = order.eq(targets).nonzero()[:, 1] + 1
        ranks.append(rank.cpu())
    return torch.cat(ranks).numpy()


def retrieval_metrics(ranks):
    return {
        "r1": float(np.mean(ranks <= 1)),
        "r5": float(np.mean(ranks <= 5)),
        "r10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def topk_predictions(ranks, samples, k=10):
    """For each query, list the top-k matched candidate image_ids (index-aligned)."""
    return [[samples[j]["image_id"] for j in np.argsort(ranks)[i][:k].tolist()] for i in range(ranks.shape[0])]


def find_epoch_checkpoints(checkpoint_dir, checkpoint_glob, start_epoch, end_epoch):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    pattern = re.compile(r"epoch_(\d+)")
    for path in sorted(checkpoint_dir.glob(checkpoint_glob)):
        m = pattern.search(path.name)
        if m is None:
            continue
        epoch = int(m.group(1))
        if epoch < start_epoch:
            continue
        if end_epoch is not None and epoch > end_epoch:
            continue
        checkpoints.append((epoch, path))
    if not checkpoints:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {checkpoint_dir} matching {checkpoint_glob!r} "
            f"from epoch {start_epoch} to {end_epoch or 'end'}."
        )
    return checkpoints


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def plot_retrieval_curves(output_dir, rows, suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text_fields = sorted({row["text_field"] for row in rows})
    for field in text_fields:
        sub = [r for r in rows if r["text_field"] == field]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for ax, metric in zip(axes, ("i2t_r1", "t2i_r1", "mean_r1")):
            epochs = [int(r["epoch"]) for r in sub]
            values = [float(r[metric]) for r in sub]
            ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=4)
            ax.set_ylabel(metric)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axes[-1].set_xlabel("epoch")
        fig.suptitle(f"R@1 across epochs — text field: {field}{' (pretrained)' if suffix else ''}")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = output_dir / f"retrieval_r1_{field}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        _LOGGER.info("Wrote metric plot: %s", out)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(DEFAULT_CKPT_DIR).parents[1] / "eval_retrieval_1000"
    )
    setup_logging(output_dir)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    text_fields = [f.strip() for f in args.text_fields.split(",") if f.strip()]
    if not text_fields:
        raise ValueError("--text-fields must contain at least one field.")

    records = load_text_records(args.jsonl, text_fields)
    pairs = discover_pairs(Path(args.images_dir), records)
    _LOGGER.info(
        "Data | images_dir=%s jsonl=%s matched_pairs=%d text_fields=%s",
        args.images_dir,
        args.jsonl,
        len(pairs),
        text_fields,
    )
    if not pairs:
        raise RuntimeError(f"No image/text pairs found under {args.images_dir} matching {args.jsonl}.")

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

    if args.eval_pretrained_only:
        checkpoints = [(0, Path(args.pretrained) if args.pretrained else Path("pretrained"))]
    elif args.checkpoint is not None:
        checkpoints = [(0, Path(args.checkpoint))]
    else:
        checkpoints = find_epoch_checkpoints(
            Path(args.checkpoint_dir), args.checkpoint_glob, args.start_epoch, args.end_epoch
        )
    _LOGGER.info(
        "Model=%s precision=%s device=%s force_context_length=%d checkpoints=%d",
        args.model,
        args.precision,
        device,
        args.force_context_length,
        len(checkpoints),
    )

    all_rows = []
    for epoch, checkpoint_path in checkpoints:
        if args.eval_pretrained_only:
            _LOGGER.info("Using pretrained backbone weights (no checkpoint loaded): %s", checkpoint_path)
        else:
            checkpoint, missing, unexpected = load_model_checkpoint(model, checkpoint_path, strict=not args.non_strict)
            ckpt_epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
            _LOGGER.info(
                "Loaded checkpoint | file_epoch=%03d ckpt_epoch=%s missing=%d unexpected=%d path=%s",
                epoch,
                ckpt_epoch,
                len(missing),
                len(unexpected),
                checkpoint_path,
            )

        for field in text_fields:
            start = time.time()
            samples, token_lengths = load_valid_samples(pairs, preprocess_val, tokenizer, field)
            num_truncated = sum(1 for n in token_lengths if n >= args.force_context_length)
            images, texts = encode_retrieval_features(model, samples, args.batch_size, device, args.precision)
            _LOGGER.info(
                "Encoded | epoch=%03d field=%s samples=%d truncated_texts=%d max_tokens=%d",
                epoch,
                field,
                len(samples),
                num_truncated,
                max(token_lengths) if token_lengths else 0,
            )

            i2t_ranks = ranks_from_similarity(images, texts, args.similarity_chunk_size)
            t2i_ranks = ranks_from_similarity(texts, images, args.similarity_chunk_size)
            i2t = retrieval_metrics(i2t_ranks)
            t2i = retrieval_metrics(t2i_ranks)
            eval_seconds = time.time() - start

            row = {
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "text_field": field,
                "num_samples": len(samples),
                "num_truncated_texts": num_truncated,
                "eval_seconds": round(eval_seconds, 2),
                "i2t_r1": i2t["r1"], "i2t_r5": i2t["r5"], "i2t_r10": i2t["r10"],
                "i2t_median_rank": i2t["median_rank"], "i2t_mean_rank": i2t["mean_rank"],
                "t2i_r1": t2i["r1"], "t2i_r5": t2i["r5"], "t2i_r10": t2i["r10"],
                "t2i_median_rank": t2i["median_rank"], "t2i_mean_rank": t2i["mean_rank"],
                "mean_r1": (i2t["r1"] + t2i["r1"]) / 2.0,
                "mean_r5": (i2t["r5"] + t2i["r5"]) / 2.0,
                "mean_r10": (i2t["r10"] + t2i["r10"]) / 2.0,
            }
            all_rows.append(row)
            _LOGGER.info(
                "Eval | epoch=%03d field=%s i2t_r1=%.4f i2t_r5=%.4f i2t_r10=%.4f t2i_r1=%.4f t2i_r5=%.4f t2i_r10=%.4f time=%.1fs",
                epoch,
                field,
                i2t["r1"], i2t["r5"], i2t["r10"],
                t2i["r1"], t2i["r5"], t2i["r10"],
                eval_seconds,
            )

            if args.save_details:
                details = {
                    "images": [s["image_id"] for s in samples],
                    "i2t_top10": topk_predictions(i2t_ranks, samples, k=10),
                    "t2i_top10": topk_predictions(t2i_ranks, samples, k=10),
                }
                detail_path = output_dir / f"retrieval_details_epoch{epoch:03d}_{field}.json"
                with detail_path.open("w", encoding="utf-8") as f:
                    json.dump(details, f, indent=2)
                _LOGGER.info("Wrote per-query details: %s", detail_path)

    suffix = "_pretrained" if args.eval_pretrained_only else ""
    summary_csv = output_dir / f"eval_retrieval_1000_summary{suffix}.csv"
    write_summary_csv(summary_csv, all_rows)
    with (output_dir / f"eval_retrieval_1000_results{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    _LOGGER.info("Wrote summary CSV: %s", summary_csv)

    best = {}
    for field in text_fields:
        sub = [r for r in all_rows if r["text_field"] == field]
        for metric in ("mean_r1", "i2t_r1", "t2i_r1"):
            best_row = max(sub, key=lambda r: (float(r[metric]), int(r["epoch"])))
            best[f"{field}:{metric}"] = {
                "epoch": best_row["epoch"],
                "checkpoint": best_row["checkpoint"],
                "value": best_row[metric],
            }
    with (output_dir / f"best_retrieval_summary{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    _LOGGER.info("Best per metric: %s", json.dumps(best, indent=2))

    if not args.no_plot:
        try:
            plot_retrieval_curves(output_dir, all_rows, suffix=suffix)
        except Exception as exc:
            _LOGGER.warning("Could not plot metric curves: %s", exc)

    # Human-readable table for the last (highest) epoch of each field.
    for field in text_fields:
        sub = [r for r in all_rows if r["text_field"] == field]
        last = max(sub, key=lambda r: int(r["epoch"]))
        _LOGGER.info(
            "Final [%s] epoch=%d | i2t R@1/5/10=%.4f/%.4f/%.4f  t2i R@1/5/10=%.4f/%.4f/%.4f  mean R@1/5/10=%.4f/%.4f/%.4f",
            field,
            int(last["epoch"]),
            last["i2t_r1"], last["i2t_r5"], last["i2t_r10"],
            last["t2i_r1"], last["t2i_r5"], last["t2i_r10"],
            last["mean_r1"], last["mean_r5"], last["mean_r10"],
        )


if __name__ == "__main__":
    main()
