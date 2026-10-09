import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import open_clip
from open_clip.model import resize_pos_embed, resize_text_pos_embed
from open_clip_train.train_eval_csv import (
    CsvImageLabelDataset,
    classification_metrics,
    decode_separator,
    evaluate_zero_shot,
    get_autocast,
    get_input_dtype,
    is_main_process,
    maybe_progress,
    read_class_names,
    save_json,
    save_per_class_report,
    topk_correct,
    unwrap_model,
)


_LOGGER = logging.getLogger("open_clip_train.eval_checkpoint_tsv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OpenCLIP checkpoint on a TSV test set with top1/top5 metrics."
    )
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint path, e.g. checkpoints/best.pt.")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional base OpenCLIP pretrained weights used only to initialize before loading --checkpoint.",
    )
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--csv-img-key", default="filepath")
    parser.add_argument("--csv-caption-key", default="title")
    parser.add_argument(
        "--csv-long-caption-key",
        default=None,
        help=(
            "Column name for long text candidates. When set, images are scored against this column, "
            "then candidates are mapped back to --csv-caption-key labels for classification metrics."
        ),
    )
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
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected checkpoint keys.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def setup_distributed(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.distributed = False
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return

    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    else:
        args.device = "cpu"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=args.dist_timeout_minutes))


def cleanup_distributed(args):
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


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
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "eval.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        _LOGGER.addHandler(file_handler)


def load_model_checkpoint(model, checkpoint_path, strict=False):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model_to_load = unwrap_model(model)
    resize_pos_embed(state_dict, model_to_load)
    resize_text_pos_embed(state_dict, model_to_load)
    missing, unexpected = model_to_load.load_state_dict(state_dict, strict=strict)
    return checkpoint, missing, unexpected


def read_long_caption_candidates(tsv_path, label_key, long_caption_key, class_to_idx, sep):
    df = pd.read_csv(tsv_path, sep=sep, usecols=[label_key, long_caption_key])
    if df[[label_key, long_caption_key]].isna().any().any():
        raise ValueError(f"{tsv_path} contains empty values in {label_key!r} or {long_caption_key!r}.")

    candidates = []
    candidate_class_indices = []
    long_caption_to_label = {}
    seen_pairs = set()
    for _, row in df.iterrows():
        label = str(row[label_key])
        long_caption = str(row[long_caption_key])
        if label not in class_to_idx:
            raise ValueError(f"Label {label!r} from {label_key!r} is not in class_to_idx.")
        previous_label = long_caption_to_label.get(long_caption)
        if previous_label is not None and previous_label != label:
            raise ValueError(
                f"Long caption maps to multiple labels: {previous_label!r} and {label!r}. "
                "Cannot compute unambiguous class metrics."
            )
        long_caption_to_label[long_caption] = label

        pair = (long_caption, label)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidates.append(long_caption)
        candidate_class_indices.append(class_to_idx[label])

    if not candidates:
        raise ValueError(f"No long caption candidates found in {tsv_path}.")
    return candidates, candidate_class_indices


@torch.no_grad()
def build_raw_text_features(model, tokenizer, texts, batch_size, device, args):
    model.eval()
    autocast = get_autocast(args.precision, device_type=device.type)
    features = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(batch_texts).to(device=device, non_blocking=True)
        with autocast():
            text_features = model.encode_text(tokens, normalize=True)
        features.append(text_features)
    return torch.cat(features, dim=0)


def candidate_logits_to_class_logits(candidate_logits, class_candidate_indices, num_classes):
    class_logits = torch.full(
        (candidate_logits.shape[0], num_classes),
        float("-inf"),
        device=candidate_logits.device,
        dtype=torch.float32,
    )
    for class_idx, candidate_indices in enumerate(class_candidate_indices):
        if candidate_indices.numel() == 0:
            continue
        class_logits[:, class_idx] = candidate_logits[:, candidate_indices].max(dim=1).values
    return class_logits


@torch.no_grad()
def evaluate_long_caption_zero_shot(
    model,
    loader,
    tokenizer,
    long_captions,
    candidate_class_indices,
    class_names,
    epoch,
    args,
    progress,
):
    device = torch.device(args.device)
    unwrapped = unwrap_model(model)
    unwrapped.eval()
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    eval_batch_size = args.eval_batch_size or args.batch_size
    text_features = build_raw_text_features(unwrapped, tokenizer, long_captions, eval_batch_size, device, args)
    logit_scale = unwrapped.logit_scale.exp()
    logit_bias = getattr(unwrapped, "logit_bias", None)

    num_classes = len(class_names)
    candidate_class_indices = torch.tensor(candidate_class_indices, dtype=torch.long)
    class_candidate_indices = [
        torch.where(candidate_class_indices == class_idx)[0].to(device=device)
        for class_idx in range(num_classes)
    ]

    total_loss = 0.0
    total_seen = 0
    total_top1 = 0
    total_top5 = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in maybe_progress(loader, f"long-caption test epoch {epoch}", progress):
        images = batch["image"].to(device=device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        targets = batch["target"].to(device=device, non_blocking=True)

        with autocast():
            image_features = unwrapped.encode_image(images, normalize=True)
            candidate_logits = logit_scale * image_features @ text_features.t()
            if logit_bias is not None:
                candidate_logits = candidate_logits + logit_bias
        logits = candidate_logits_to_class_logits(candidate_logits.float(), class_candidate_indices, num_classes)
        loss = F.cross_entropy(logits, targets)
        preds = logits.argmax(dim=1)

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_seen += batch_size
        total_top1 += topk_correct(logits, targets, 1)
        total_top5 += topk_correct(logits, targets, 5)

        for target, pred in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            confusion[target, pred] += 1

    values = torch.tensor([total_loss, total_seen, total_top1, total_top5], dtype=torch.float64, device=device)
    if args.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        confusion_tensor = torch.from_numpy(confusion).to(device=device, dtype=torch.long)
        dist.all_reduce(confusion_tensor, op=dist.ReduceOp.SUM)
        confusion = confusion_tensor.cpu().numpy()
    total_loss, total_seen, total_top1, total_top5 = values.tolist()

    metrics = classification_metrics(confusion)
    metrics.update(
        {
            "loss": total_loss / max(total_seen, 1.0),
            "top1": total_top1 / max(total_seen, 1.0),
            "top5": total_top5 / max(total_seen, 1.0),
        }
    )
    return metrics, confusion


def main():
    args = parse_args()
    args.csv_separator = decode_separator(args.csv_separator)
    setup_distributed(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    setup_logging(output_dir, args)

    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    _LOGGER.info(
        "Starting checkpoint eval | rank=%d world_size=%d device=%s checkpoint=%s test_data=%s",
        args.rank,
        args.world_size,
        args.device,
        args.checkpoint,
        args.test_data,
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
    checkpoint, missing, unexpected = load_model_checkpoint(model, args.checkpoint, strict=not args.non_strict)
    if is_main_process(args):
        _LOGGER.info(
            "Loaded checkpoint | epoch=%s best_epoch=%s missing_keys=%d unexpected_keys=%d",
            checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
            checkpoint.get("best_epoch", "unknown") if isinstance(checkpoint, dict) else "unknown",
            len(missing),
            len(unexpected),
        )

    tokenizer = open_clip.get_tokenizer(
        args.model,
        cache_dir=args.cache_dir,
        context_length=args.force_context_length,
    )
    class_names = read_class_names(args.test_data, args.csv_caption_key, args.csv_separator)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_dataset = CsvImageLabelDataset(
        args.test_data,
        args.csv_img_key,
        args.csv_caption_key,
        preprocess_val,
        class_to_idx,
        sep=args.csv_separator,
    )
    test_dataset_for_loader = (
        Subset(test_dataset, list(range(args.rank, len(test_dataset), args.world_size)))
        if args.distributed
        else test_dataset
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    test_loader = DataLoader(
        test_dataset_for_loader,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda" and not args.disable_pin_memory,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        timeout=args.dataloader_timeout if args.workers > 0 else 0,
        drop_last=False,
    )

    use_long_caption = args.csv_long_caption_key is not None
    long_caption_count = None
    if use_long_caption:
        if args.force_context_length is None:
            _LOGGER.warning(
                "--csv-long-caption-key is set but --force-context-length is not. "
                "Long captions exceeding the model's default 77-token context length will be truncated."
            )
        long_captions, candidate_class_indices = read_long_caption_candidates(
            args.test_data,
            args.csv_caption_key,
            args.csv_long_caption_key,
            class_to_idx,
            args.csv_separator,
        )
        long_caption_count = len(long_captions)
        _LOGGER.info(
            "Long-caption eval | long_caption_candidates=%d classes=%d label_key=%s text_key=%s",
            long_caption_count,
            len(class_names),
            args.csv_caption_key,
            args.csv_long_caption_key,
        )
        metrics, confusion = evaluate_long_caption_zero_shot(
            model,
            test_loader,
            tokenizer,
            long_captions,
            candidate_class_indices,
            class_names,
            epoch=0,
            args=args,
            progress=is_main_process(args) and not args.no_progress,
        )
    else:
        metrics, confusion = evaluate_zero_shot(
            model,
            test_loader,
            tokenizer,
            class_names,
            epoch=0,
            args=args,
            progress=is_main_process(args) and not args.no_progress,
        )

    if is_main_process(args):
        eval_mode = "long_caption" if use_long_caption else "short_caption"
        result = {
            "checkpoint": str(args.checkpoint),
            "test_data": str(args.test_data),
            "eval_mode": eval_mode,
            "num_samples": len(test_dataset),
            "num_classes": len(class_names),
            "loss": metrics["loss"],
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
        }
        if long_caption_count is not None:
            result["num_long_caption_candidates"] = long_caption_count
        _LOGGER.info(
            "Eval finished [%s] | samples=%d classes=%d top1=%.6f top5=%.6f macro_f1=%.6f",
            eval_mode,
            result["num_samples"],
            result["num_classes"],
            result["top1"],
            result["top5"],
            result["macro_f1"],
        )
        if output_dir is not None:
            save_json(output_dir / "eval_topk.json", result)
            save_json(output_dir / "class_to_idx.json", class_to_idx)
            np.save(output_dir / "confusion_matrix.npy", confusion)
            save_per_class_report(output_dir / "per_class_report.csv", class_names, metrics["per_class"])
            with (output_dir / "eval_topk.csv").open("w", encoding="utf-8") as f:
                f.write(
                    "checkpoint,test_data,eval_mode,num_samples,num_classes,loss,top1,top5,"
                    "macro_precision,macro_recall,macro_f1,weighted_precision,weighted_recall,weighted_f1\n"
                )
                f.write(
                    f"{result['checkpoint']},{result['test_data']},{result['eval_mode']},"
                    f"{result['num_samples']},{result['num_classes']},{result['loss']},"
                    f"{result['top1']},{result['top5']},{result['macro_precision']},"
                    f"{result['macro_recall']},{result['macro_f1']},{result['weighted_precision']},"
                    f"{result['weighted_recall']},{result['weighted_f1']}\n"
                )

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
