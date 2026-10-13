import argparse
import csv
import logging
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

import open_clip
from open_clip_train.eval_checkpoint_tsv import (
    cleanup_distributed,
    load_model_checkpoint,
    setup_distributed,
)
from open_clip_train.train_eval_csv import (
    CsvImageLabelDataset,
    decode_separator,
    evaluate_zero_shot,
    is_main_process,
    read_class_names,
    save_json,
)
from open_clip_train.data import tarfile_to_samples_nothrow


_LOGGER = logging.getLogger(__name__)

DEFAULT_EVAL_SPLITS = {
    "iid": "evaluation/iid/shard-000000.tar",
    "partial_ood": "evaluation/compositional_shift/shard-000000.tar",
    "ood": "evaluation/ood/shard-000000.tar",
    "cell_ood": "evaluation/cell_line_shift/shard-000000.tar",
    "structure_ood": "evaluation/organelle_set_shift/shard-000000.tar",
}

CLASSIFICATION_SPLITS = ["iid", "partial_ood", "ood", "cell_ood", "structure_ood"]

SUMMARY_FIELDS = [
    "epoch",
    "checkpoint",
    "split",
    "test_data",
    "num_samples",
    "num_classes",
    "eval_seconds",
    "loss",
    "top1",
    "top5",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "best_metric",
    "best_epoch_for_split",
    "best_value_for_split",
    "is_best_for_split",
]

DEFAULT_PLOT_METRICS = ["top1", "top5"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every epoch_*.pt checkpoint in a directory on iid/partial_ood/ood TSV test sets "
            "and write one CSV summary with the best epoch marked."
        )
    )
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional base OpenCLIP pretrained weights used only to initialize before each checkpoint load.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing epoch_001.pt, epoch_002.pt, ... checkpoint files.",
    )
    parser.add_argument("--checkpoint-glob", default="epoch_*.pt")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=None)
    parser.add_argument("--iid-data", default=DEFAULT_EVAL_SPLITS["iid"])
    parser.add_argument("--partial-ood-data", default=DEFAULT_EVAL_SPLITS["partial_ood"])
    parser.add_argument("--ood-data", default=DEFAULT_EVAL_SPLITS["ood"])
    parser.add_argument("--cell-ood-data", default=DEFAULT_EVAL_SPLITS["cell_ood"])
    parser.add_argument("--structure-ood-data", default=DEFAULT_EVAL_SPLITS["structure_ood"])
    parser.add_argument("--csv-img-key", default="filepath")
    parser.add_argument("--csv-caption-key", default="title")
    parser.add_argument("--csv-separator", default="\t")
    parser.add_argument("--template", default="{name}")
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp_bf16",
    )
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
    parser.add_argument("--plot-metrics", nargs="+", default=DEFAULT_PLOT_METRICS)
    parser.add_argument("--best-metric", choices=["top1", "top5", "macro_f1", "weighted_f1"], default="top1")
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected checkpoint keys.")
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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "eval_all_epochs.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        _LOGGER.addHandler(file_handler)


def checkpoint_epoch(path):
    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def find_epoch_checkpoints(checkpoint_dir, checkpoint_glob, start_epoch, end_epoch):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    for path in checkpoint_dir.glob(checkpoint_glob):
        epoch = checkpoint_epoch(path)
        if epoch is None:
            continue
        if epoch < start_epoch:
            continue
        if end_epoch is not None and epoch > end_epoch:
            continue
        checkpoints.append((epoch, path))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {checkpoint_dir} matching {checkpoint_glob!r} "
            f"from epoch {start_epoch} to {end_epoch or 'end'}."
        )
    return checkpoints


def looks_like_wds(path):
    return any(part in str(path) for part in (".tar", "{", "::"))


def read_wds_class_names(path):
    class_names = []
    seen = set()
    num_samples = 0
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".txt"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            label = extracted.read().decode("utf-8", errors="replace").strip()
            if not label:
                continue
            num_samples += 1
            if label not in seen:
                seen.add(label)
                class_names.append(label)
    if not class_names:
        raise RuntimeError(f"No non-empty .txt labels found in WebDataset tar: {path}")
    return class_names, num_samples


def filter_no_wds_label_or_image(sample):
    has_image = any(key in sample for key in ("jpg", "jpeg", "png", "webp"))
    return has_image and "txt" in sample


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


class WdsEvalMapper:
    def __init__(self, transform, class_to_idx):
        self.transform = transform
        self.class_to_idx = class_to_idx

    @staticmethod
    def _label_value(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()

    def __call__(self, sample):
        label = self._label_value(sample["txt"])
        if label not in self.class_to_idx:
            raise KeyError(f"Label not found in class_to_idx: {label}")
        return {
            "image": self.transform(sample["image"]),
            "target": torch.tensor(self.class_to_idx[label], dtype=torch.long),
            "label": label,
        }


def collate_eval_batch(batch):
    return default_collate(
        [
            {
                "image": sample["image"],
                "target": sample["target"],
                "label": sample["label"],
            }
            for sample in batch
        ]
    )


def build_eval_loader(test_data, split_name, preprocess_val, args, device):
    if looks_like_wds(test_data):
        class_names, num_samples = read_wds_class_names(test_data)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        eval_batch_size = args.eval_batch_size or args.batch_size
        pipeline = [
            wds.SimpleShardList(test_data),
            tarfile_to_samples_nothrow,
            wds.select(SplitEvalSamples(args.rank, args.world_size)),
            wds.split_by_worker,
            wds.select(filter_no_wds_label_or_image),
            wds.decode("pilrgb"),
            wds.rename(image="jpg;png;jpeg;webp", txt="txt", keep=False),
            wds.map(WdsEvalMapper(preprocess_val, class_to_idx)),
            wds.batched(eval_batch_size, partial=True, collation_fn=collate_eval_batch),
        ]
        dataset = wds.DataPipeline(*pipeline)
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=getattr(args, "persistent_workers", True) and args.workers > 0,
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
            timeout=args.dataloader_timeout if args.workers > 0 else 0,
        )
        batches_per_rank = int(np.ceil((num_samples / max(args.world_size, 1)) / eval_batch_size))
        _LOGGER.info(
            "Loaded WebDataset eval split | split=%s samples=%d classes=%d batches_per_rank~=%d test_data=%s",
            split_name,
            num_samples,
            len(class_names),
            batches_per_rank,
            test_data,
        )
        return {
            "split": split_name,
            "test_data": test_data,
            "class_names": class_names,
            "num_samples": num_samples,
            "num_classes": len(class_names),
            "loader": loader,
        }

    class_names = read_class_names(test_data, args.csv_caption_key, args.csv_separator)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    dataset = CsvImageLabelDataset(
        test_data,
        args.csv_img_key,
        args.csv_caption_key,
        preprocess_val,
        class_to_idx,
        sep=args.csv_separator,
    )
    dataset_for_loader = (
        Subset(dataset, list(range(args.rank, len(dataset), args.world_size)))
        if args.distributed
        else dataset
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    loader = DataLoader(
        dataset_for_loader,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda" and not args.disable_pin_memory,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        timeout=args.dataloader_timeout if args.workers > 0 else 0,
        drop_last=False,
    )
    _LOGGER.info(
        "Loaded eval split | split=%s samples=%d classes=%d batches_per_rank=%d test_data=%s",
        split_name,
        len(dataset),
        len(class_names),
        len(loader),
        test_data,
    )
    return {
        "split": split_name,
        "test_data": test_data,
        "class_names": class_names,
        "num_samples": len(dataset),
        "num_classes": len(class_names),
        "loader": loader,
    }


def metric_row(epoch, checkpoint_path, split_info, metrics, eval_seconds, best_metric):
    return {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "split": split_info["split"],
        "test_data": split_info["test_data"],
        "num_samples": split_info["num_samples"],
        "num_classes": split_info["num_classes"],
        "eval_seconds": eval_seconds,
        "loss": metrics["loss"],
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "best_metric": best_metric,
        "best_epoch_for_split": "",
        "best_value_for_split": "",
        "is_best_for_split": "",
    }


def mean_row(epoch, checkpoint_path, rows, best_metric):
    metric_names = [
        "loss",
        "top1",
        "top5",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    row = {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "split": "mean",
        "test_data": "mean(iid,partial_ood,ood,cell_ood,structure_ood)",
        "num_samples": sum(item["num_samples"] for item in rows),
        "num_classes": "",
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
    rows_by_split = {}
    for row in rows:
        rows_by_split.setdefault(row["split"], []).append(row)

    best_summary = {}
    for split, split_rows in rows_by_split.items():
        best_row = max(split_rows, key=lambda item: (float(item[best_metric]), -int(item["epoch"])))
        best_epoch = best_row["epoch"]
        best_value = best_row[best_metric]
        best_summary[split] = {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_value": best_value,
            "checkpoint": best_row["checkpoint"],
        }
        for row in split_rows:
            row["best_epoch_for_split"] = best_epoch
            row["best_value_for_split"] = best_value
            row["is_best_for_split"] = int(row["epoch"] == best_epoch)
    return best_summary


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in SUMMARY_FIELDS})


def plot_metric_curves(path, rows, metric_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_metrics = [
        name
        for name in metric_names
        if name in SUMMARY_FIELDS and name not in {"epoch", "eval_seconds", "num_samples", "num_classes"}
    ]
    if not valid_metrics:
        _LOGGER.warning("No valid metrics to plot from --plot-metrics=%s", metric_names)
        return

    split_order = ["iid", "partial_ood", "ood", "cell_ood", "structure_ood", "mean"]
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
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        if metric != "loss":
            ax.set_ylim(bottom=0.0)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Evaluation metrics across epochs")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


@record
def main():
    args = parse_args()
    args.csv_separator = decode_separator(args.csv_separator)
    setup_distributed(args)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent / "eval_all_epochs"
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "eval_all_epochs_summary.csv"
    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "eval_all_epochs_metrics.png"
    setup_logging(output_dir, args)

    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    checkpoints = find_epoch_checkpoints(
        checkpoint_dir,
        args.checkpoint_glob,
        args.start_epoch,
        args.end_epoch,
    )
    _LOGGER.info(
        "Starting all-epoch eval | rank=%d world_size=%d device=%s checkpoints=%d first=%s last=%s output=%s",
        args.rank,
        args.world_size,
        args.device,
        len(checkpoints),
        checkpoints[0][1],
        checkpoints[-1][1],
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
    tokenizer = open_clip.get_tokenizer(
        args.model,
        cache_dir=args.cache_dir,
        context_length=args.force_context_length,
    )

    eval_splits = [
        build_eval_loader(getattr(args, f"{split}_data"), split, preprocess_val, args, device)
        for split in CLASSIFICATION_SPLITS
    ]

    all_rows = []
    for epoch, checkpoint_path in checkpoints:
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
            metrics, _ = evaluate_zero_shot(
                model,
                split_info["loader"],
                tokenizer,
                split_info["class_names"],
                epoch=epoch,
                args=args,
                progress=is_main_process(args) and not args.no_progress,
            )
            eval_seconds = time.time() - start
            row = metric_row(epoch, checkpoint_path, split_info, metrics, eval_seconds, args.best_metric)
            epoch_rows.append(row)
            if is_main_process(args):
                _LOGGER.info(
                    "Eval finished | epoch=%03d split=%s top1=%.6f top5=%.6f macro_f1=%.6f time=%.1fs",
                    epoch,
                    split_info["split"],
                    row["top1"],
                    row["top5"],
                    row["macro_f1"],
                    eval_seconds,
                )

        if is_main_process(args):
            all_rows.extend(epoch_rows)
            all_rows.append(mean_row(epoch, checkpoint_path, epoch_rows, args.best_metric))

    if is_main_process(args):
        best_summary = mark_best_rows(all_rows, args.best_metric)
        write_summary_csv(summary_csv, all_rows)
        try:
            plot_metric_curves(plot_path, all_rows, args.plot_metrics)
            _LOGGER.info("Wrote metric plot: %s", plot_path)
        except ImportError as exc:
            _LOGGER.warning("Could not plot metric curves because matplotlib is unavailable: %s", exc)
        save_json(output_dir / "best_summary.json", best_summary)
        for split in CLASSIFICATION_SPLITS + ["mean"]:
            best = best_summary[split]
            _LOGGER.info(
                "Best %s | split=%s epoch=%s value=%.6f checkpoint=%s",
                args.best_metric,
                split,
                best["best_epoch"],
                best["best_value"],
                best["checkpoint"],
            )
        _LOGGER.info("Wrote summary CSV: %s", summary_csv)

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
