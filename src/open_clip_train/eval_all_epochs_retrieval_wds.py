import argparse
import csv
import json
import logging
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.dataloader import default_collate

import open_clip
from open_clip import get_input_dtype
from open_clip_train.data import tarfile_to_samples_nothrow
from open_clip_train.eval_all_epochs_tsv_draw import (
    DEFAULT_EVAL_SPLITS,
    find_epoch_checkpoints,
    looks_like_wds,
)
from open_clip_train.eval_checkpoint_tsv import (
    cleanup_distributed,
    load_model_checkpoint,
    setup_distributed,
)
from open_clip_train.precision import get_autocast
from open_clip_train.train_eval_csv import is_main_process, maybe_progress, save_json, unwrap_model


_LOGGER = logging.getLogger("open_clip_train.eval_all_epochs_retrieval_wds")

SUMMARY_FIELDS = [
    "epoch",
    "checkpoint",
    "split",
    "test_data",
    "num_samples",
    "eval_seconds",
    "i2t_r1",
    "i2t_r5",
    "i2t_r10",
    "i2t_median_rank",
    "i2t_mean_rank",
    "t2i_r1",
    "t2i_r5",
    "t2i_r10",
    "t2i_median_rank",
    "t2i_mean_rank",
    "mean_r1",
    "mean_r5",
    "mean_r10",
    "best_metric",
    "best_epoch_for_split",
    "best_value_for_split",
    "is_best_for_split",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate image-text retrieval for every epoch checkpoint on WebDataset eval splits."
    )
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-glob", default="epoch_*.pt")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=None)
    parser.add_argument(
        "--data",
        default=None,
        help="Evaluate one WebDataset instead of the configured IID/OOD splits.",
    )
    parser.add_argument("--split-name", default="eval")
    parser.add_argument(
        "--eval-pretrained-only",
        action="store_true",
        help="Evaluate the model initialized from --pretrained without loading epoch checkpoints.",
    )
    parser.add_argument("--iid-data", default=DEFAULT_EVAL_SPLITS["iid"])
    parser.add_argument("--partial-ood-data", default=DEFAULT_EVAL_SPLITS["partial_ood"])
    parser.add_argument("--ood-data", default=DEFAULT_EVAL_SPLITS["ood"])
    parser.add_argument(
        "--retrieval-text-source",
        choices=["txt", "json"],
        default="json",
        help="Use short .txt text or a field from .json as the text side of retrieval.",
    )
    parser.add_argument(
        "--retrieval-json-key",
        default="long_caption",
        help="JSON key used when --retrieval-text-source=json.",
    )
    parser.add_argument(
        "--fallback-to-txt",
        action="store_true",
        help="Fall back to .txt if the requested JSON field is missing or empty.",
    )
    parser.add_argument(
        "--retrieval-direction",
        choices=["both", "i2t", "t2i"],
        default="both",
        help="Evaluate image-to-text, text-to-image, or both retrieval directions.",
    )
    parser.add_argument("--precision", default="amp_bf16")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--dataloader-timeout", type=float, default=0.0)
    parser.add_argument("--dist-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--force-quick-gelu", action="store_true")
    parser.add_argument("--force-image-size", type=int, default=None)
    parser.add_argument("--force-context-length", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--pretrained-image", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--plot-metrics", nargs="+", default=["mean_r1", "mean_r5", "mean_r10"])
    parser.add_argument(
        "--best-metric",
        choices=["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_r1", "mean_r5", "mean_r10"],
        default="mean_r1",
    )
    parser.add_argument("--similarity-chunk-size", type=int, default=4096)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def setup_logging(output_dir, args):
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.handlers.clear()
    _LOGGER.propagate = False
    if not is_main_process(args):
        _LOGGER.addHandler(logging.NullHandler())
        return
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    _LOGGER.addHandler(stream_handler)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "eval_all_epochs_retrieval.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)


def read_wds_num_samples(path):
    if not looks_like_wds(path) or "{" in str(path) or "::" in str(path):
        return None
    count = 0
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith(".txt"):
                count += 1
    return count


class SplitEvalSamples:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def __call__(self, sample):
        key = str(sample.get("__key__", "0"))
        try:
            sample_idx = int(Path(key).name)
        except ValueError:
            sample_idx = abs(hash(key))
        return sample_idx % self.world_size == self.rank


def filter_no_wds_text_or_image(sample):
    has_image = any(key in sample for key in ("jpg", "jpeg", "png", "webp"))
    return has_image and "txt" in sample


def filter_no_wds_json_text_or_image(sample):
    return filter_no_wds_text_or_image(sample) and "json" in sample


class WdsRetrievalMapper:
    def __init__(self, transform, tokenizer, text_source, json_key, fallback_to_txt):
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_source = text_source
        self.json_key = json_key
        self.fallback_to_txt = fallback_to_txt

    @staticmethod
    def _string_value(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()

    def _json_value(self, value):
        if isinstance(value, dict):
            return value
        text = self._string_value(value)
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse WebDataset JSON metadata: {exc}") from exc
        return parsed if isinstance(parsed, dict) else {}

    def _sample_text(self, sample):
        short_text = self._string_value(sample.get("txt"))
        if self.text_source == "txt":
            return short_text

        metadata = self._json_value(sample.get("json"))
        text = self._string_value(metadata.get(self.json_key))
        if text:
            return text
        if self.fallback_to_txt and short_text:
            return short_text
        raise KeyError(f"Missing non-empty JSON key {self.json_key!r} for sample {sample.get('__key__', '')!r}")

    def __call__(self, sample):
        url = str(sample.get("__url__", ""))
        key = f"{url}::{sample.get('__key__', '')}"
        text = self._sample_text(sample)
        return {
            "image": self.transform(sample["image"]),
            "text": self.tokenizer([text])[0],
            "key": key,
        }


def collate_retrieval_batch(batch):
    return default_collate([{"image": x["image"], "text": x["text"], "key": x["key"]} for x in batch])


def build_retrieval_loader(test_data, split_name, preprocess_val, tokenizer, args):
    if not looks_like_wds(test_data):
        raise ValueError(f"Retrieval eval expects WebDataset tar input, got: {test_data}")
    eval_batch_size = args.eval_batch_size or args.batch_size
    num_samples = read_wds_num_samples(test_data)
    sample_filter = filter_no_wds_json_text_or_image if args.retrieval_text_source == "json" else filter_no_wds_text_or_image
    pipeline = [
        wds.SimpleShardList(test_data),
        tarfile_to_samples_nothrow,
        wds.select(SplitEvalSamples(args.rank, args.world_size)),
        wds.split_by_worker,
        wds.select(sample_filter),
        wds.decode("pilrgb"),
        wds.rename(image="jpg;png;jpeg;webp", txt="txt", json="json", keep=False),
        wds.map(
            WdsRetrievalMapper(
                preprocess_val,
                tokenizer,
                args.retrieval_text_source,
                args.retrieval_json_key,
                args.fallback_to_txt,
            )
        ),
        wds.batched(eval_batch_size, partial=True, collation_fn=collate_retrieval_batch),
    ]
    loader = wds.WebLoader(
        wds.DataPipeline(*pipeline),
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=getattr(args, "persistent_workers", True) and args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        timeout=args.dataloader_timeout if args.workers > 0 else 0,
    )
    return {"split": split_name, "test_data": test_data, "num_samples": num_samples, "loader": loader}


def gather_object(args, value):
    if not args.distributed:
        return [value]
    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


@torch.no_grad()
def encode_retrieval_features(model, loader, args, progress):
    device = torch.device(args.device)
    model = unwrap_model(model)
    model.eval()
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    image_features = []
    text_features = []
    keys = []
    for batch in maybe_progress(loader, "retrieval", progress):
        images = batch["image"].to(device=device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        texts = batch["text"].to(device=device, non_blocking=True)
        with autocast():
            image_features.append(model.encode_image(images, normalize=True).float().cpu())
            text_features.append(model.encode_text(texts, normalize=True).float().cpu())
        keys.extend([str(key) for key in batch["key"]])

    local = {
        "image_features": torch.cat(image_features, dim=0) if image_features else torch.empty(0, 1),
        "text_features": torch.cat(text_features, dim=0) if text_features else torch.empty(0, 1),
        "keys": keys,
    }
    gathered = gather_object(args, local)
    if not is_main_process(args):
        return None, None, None

    all_items = []
    for item in gathered:
        for idx, key in enumerate(item["keys"]):
            all_items.append((key, item["image_features"][idx], item["text_features"][idx]))
    all_items.sort(key=lambda item: item[0])
    images = torch.stack([item[1] for item in all_items], dim=0)
    texts = torch.stack([item[2] for item in all_items], dim=0)
    keys = [item[0] for item in all_items]
    return images, texts, keys


def ranks_from_similarity(query_features, candidate_features, chunk_size):
    ranks = []
    num_samples = query_features.shape[0]
    for start in range(0, num_samples, chunk_size):
        scores = query_features[start : start + chunk_size] @ candidate_features.t()
        targets = torch.arange(start, min(start + chunk_size, num_samples)).view(-1, 1)
        order = scores.argsort(dim=1, descending=True)
        rank = order.eq(targets).nonzero()[:, 1] + 1
        ranks.append(rank.cpu())
    return torch.cat(ranks).numpy()


def retrieval_metrics_from_ranks(ranks, prefix):
    return {
        f"{prefix}_r1": float(np.mean(ranks <= 1)),
        f"{prefix}_r5": float(np.mean(ranks <= 5)),
        f"{prefix}_r10": float(np.mean(ranks <= 10)),
        f"{prefix}_median_rank": float(np.median(ranks)),
        f"{prefix}_mean_rank": float(np.mean(ranks)),
    }


def evaluate_retrieval(model, split_info, args):
    images, texts, keys = encode_retrieval_features(
        model,
        split_info["loader"],
        args,
        progress=is_main_process(args) and not args.no_progress,
    )
    if not is_main_process(args):
        return None
    if len(keys) == 0:
        raise RuntimeError(f"No samples were encoded for split {split_info['split']}")
    metrics = {}
    if args.retrieval_direction in ("both", "i2t"):
        i2t_ranks = ranks_from_similarity(images, texts, args.similarity_chunk_size)
        metrics.update(retrieval_metrics_from_ranks(i2t_ranks, "i2t"))
    if args.retrieval_direction in ("both", "t2i"):
        t2i_ranks = ranks_from_similarity(texts, images, args.similarity_chunk_size)
        metrics.update(retrieval_metrics_from_ranks(t2i_ranks, "t2i"))
    if args.retrieval_direction == "both":
        metrics["mean_r1"] = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2.0
        metrics["mean_r5"] = (metrics["i2t_r5"] + metrics["t2i_r5"]) / 2.0
        metrics["mean_r10"] = (metrics["i2t_r10"] + metrics["t2i_r10"]) / 2.0
    metrics["num_samples"] = len(keys)
    return metrics


def metric_row(epoch, checkpoint_path, split_info, metrics, eval_seconds, best_metric):
    return {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "split": split_info["split"],
        "test_data": split_info["test_data"],
        "num_samples": metrics["num_samples"],
        "eval_seconds": eval_seconds,
        "best_metric": best_metric,
        "best_epoch_for_split": "",
        "best_value_for_split": "",
        "is_best_for_split": "",
        **{key: metrics[key] for key in SUMMARY_FIELDS if key in metrics},
    }


def mean_row(epoch, checkpoint_path, rows, best_metric):
    metric_names = [
        name
        for name in SUMMARY_FIELDS
        if re.match(r"(i2t|t2i|mean)_", name) and all(name in item for item in rows)
    ]
    row = {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "split": "mean",
        "test_data": "mean(iid,partial_ood,ood)",
        "num_samples": sum(item["num_samples"] for item in rows),
        "eval_seconds": sum(item["eval_seconds"] for item in rows),
        "best_metric": best_metric,
        "best_epoch_for_split": "",
        "best_value_for_split": "",
        "is_best_for_split": "",
    }
    for name in metric_names:
        row[name] = float(np.mean([item[name] for item in rows]))
    return row


def mark_best_rows(rows, best_metric):
    best_summary = {}
    for split in sorted(set(row["split"] for row in rows)):
        split_rows = [row for row in rows if row["split"] == split]
        best_row = max(split_rows, key=lambda item: (float(item[best_metric]), -int(item["epoch"])))
        best_summary[split] = {
            "best_metric": best_metric,
            "best_epoch": best_row["epoch"],
            "best_value": best_row[best_metric],
            "checkpoint": best_row["checkpoint"],
        }
        for row in split_rows:
            row["best_epoch_for_split"] = best_row["epoch"]
            row["best_value_for_split"] = best_row[best_metric]
            row["is_best_for_split"] = int(row["epoch"] == best_row["epoch"])
    return best_summary


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def plot_retrieval_curves(path, rows, metric_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_metrics = [name for name in metric_names if name in SUMMARY_FIELDS]
    if not valid_metrics:
        _LOGGER.warning("No valid retrieval metrics to plot from --plot-metrics=%s", metric_names)
        return

    split_order = ["iid", "partial_ood", "ood", "mean"]
    rows_by_split = {
        split: sorted([row for row in rows if row["split"] == split], key=lambda item: int(item["epoch"]))
        for split in split_order
    }

    fig, axes = plt.subplots(
        len(valid_metrics),
        1,
        figsize=(10, max(3.0, 2.8 * len(valid_metrics))),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, metric in zip(axes, valid_metrics):
        for split in split_order:
            split_rows = rows_by_split[split]
            if not split_rows:
                continue
            epochs = [int(row["epoch"]) for row in split_rows]
            values = [float(row[metric]) for row in split_rows]
            ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=4, label=split)
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Retrieval metrics across epochs")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validate_retrieval_args(args):
    if args.retrieval_direction == "i2t" and not args.best_metric.startswith("i2t_"):
        raise ValueError("--best-metric must start with i2t_ when --retrieval-direction=i2t.")
    if args.retrieval_direction == "t2i" and not args.best_metric.startswith("t2i_"):
        raise ValueError("--best-metric must start with t2i_ when --retrieval-direction=t2i.")
    if args.retrieval_direction != "both":
        invalid_plot_metrics = [
            metric
            for metric in args.plot_metrics
            if not metric.startswith(f"{args.retrieval_direction}_")
        ]
        if invalid_plot_metrics:
            raise ValueError(
                f"--plot-metrics contains metrics outside {args.retrieval_direction.upper()}: "
                f"{invalid_plot_metrics}"
            )


@record
def main():
    args = parse_args()
    validate_retrieval_args(args)
    if not args.eval_pretrained_only and not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required unless --eval-pretrained-only is set.")
    setup_distributed(args)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_dir.parent / "eval_all_epochs_retrieval"
        if checkpoint_dir is not None
        else Path("eval_pretrained_retrieval")
    )
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "eval_all_epochs_retrieval_summary.csv"
    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "eval_all_epochs_retrieval_metrics.png"
    setup_logging(output_dir, args)

    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.eval_pretrained_only:
        checkpoints = [(0, Path(args.pretrained) if args.pretrained else Path("pretrained"))]
    else:
        checkpoints = find_epoch_checkpoints(
            checkpoint_dir, args.checkpoint_glob, args.start_epoch, args.end_epoch
        )
    _LOGGER.info(
        "Starting retrieval eval | rank=%d world_size=%d checkpoints=%d pretrained_only=%s output=%s",
        args.rank,
        args.world_size,
        len(checkpoints),
        args.eval_pretrained_only,
        summary_csv,
    )

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
    tokenizer = open_clip.get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    if args.data:
        eval_splits = [
            build_retrieval_loader(args.data, args.split_name, preprocess_val, tokenizer, args)
        ]
    else:
        eval_splits = [
            build_retrieval_loader(args.iid_data, "iid", preprocess_val, tokenizer, args),
            build_retrieval_loader(args.partial_ood_data, "partial_ood", preprocess_val, tokenizer, args),
            build_retrieval_loader(args.ood_data, "ood", preprocess_val, tokenizer, args),
        ]

    all_rows = []
    for epoch, checkpoint_path in checkpoints:
        if args.eval_pretrained_only:
            if is_main_process(args):
                _LOGGER.info("Using pretrained weights without loading epoch checkpoint: %s", checkpoint_path)
        else:
            checkpoint, missing, unexpected = load_model_checkpoint(model, checkpoint_path, strict=not args.non_strict)
            if is_main_process(args):
                _LOGGER.info(
                    "Loaded checkpoint | epoch_file=%03d checkpoint_epoch=%s missing_keys=%d unexpected_keys=%d path=%s",
                    epoch,
                    checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
                    len(missing),
                    len(unexpected),
                    checkpoint_path,
                )

        epoch_rows = []
        for split_info in eval_splits:
            start = time.time()
            metrics = evaluate_retrieval(model, split_info, args)
            eval_seconds = time.time() - start
            if is_main_process(args):
                row = metric_row(epoch, checkpoint_path, split_info, metrics, eval_seconds, args.best_metric)
                epoch_rows.append(row)
                _LOGGER.info(
                    "Retrieval finished | epoch=%03d split=%s direction=%s best_metric=%s best_value=%.6f time=%.1fs",
                    epoch,
                    split_info["split"],
                    args.retrieval_direction,
                    args.best_metric,
                    row[args.best_metric],
                    eval_seconds,
                )

        if is_main_process(args):
            all_rows.extend(epoch_rows)
            all_rows.append(mean_row(epoch, checkpoint_path, epoch_rows, args.best_metric))

    if is_main_process(args):
        best_summary = mark_best_rows(all_rows, args.best_metric)
        write_summary_csv(summary_csv, all_rows)
        try:
            plot_retrieval_curves(plot_path, all_rows, args.plot_metrics)
            _LOGGER.info("Wrote metric plot: %s", plot_path)
        except ImportError as exc:
            _LOGGER.warning("Could not plot metric curves because matplotlib is unavailable: %s", exc)
        save_json(output_dir / "best_retrieval_summary.json", best_summary)
        _LOGGER.info("Wrote retrieval summary CSV: %s", summary_csv)

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
