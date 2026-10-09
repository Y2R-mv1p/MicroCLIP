import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

import open_clip
from open_clip import get_input_dtype
from open_clip.loss import ClipLoss
from open_clip_train.params import get_default_params
from open_clip_train.precision import get_autocast
from open_clip_train.scheduler import const_lr, const_lr_cooldown, cosine_lr

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_TRAIN_DATA = "/data2/ryr/dataset_new/ood_split_strict_check/iid/train.tsv"
DEFAULT_TEST_DATA = "/data2/ryr/dataset_new/ood_split_strict_check/iid/test.tsv"
DEFAULT_PRETRAINED = (
    "/data2/ryr/workspace/microllm/open_clip-main/logs/"
    "2026_05_19-14_53_04-model_ViT-L-14-lr_1e-05-b_2048-j_8-p_amp_bf16/"
    "checkpoints/epoch_1.pt"
)

METRICS_FIELDS = [
    "epoch",
    "train_loss",
    "train_top1",
    "train_top5",
    "test_loss",
    "test_top1",
    "test_top5",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_precision",
    "test_weighted_recall",
    "test_weighted_f1",
]

TIME_FIELDS = [
    "epoch",
    "train_seconds",
    "test_seconds",
    "epoch_seconds",
    "total_elapsed_seconds",
]

_LOGGER = logging.getLogger("open_clip_train.train_eval_csv")


class CsvImageTextDataset(Dataset):
    def __init__(
        self,
        tsv_path,
        image_key,
        caption_key,
        transform,
        tokenizer,
        sep="\t",
        long_caption_key=None,
        long_caption_mode="none",
        long_caption_chunk_tokens=75,
        long_caption_chunk_overlap=8,
        long_caption_prepend_title=False,
    ):
        self.tsv_path = Path(tsv_path)
        self.df = pd.read_csv(self.tsv_path, sep=sep)
        self.image_key = image_key
        self.caption_key = caption_key
        self.transform = transform
        self.tokenizer = tokenizer
        self.long_caption_key = long_caption_key
        self.long_caption_mode = long_caption_mode
        self.long_caption_chunk_tokens = long_caption_chunk_tokens
        self.long_caption_chunk_overlap = long_caption_chunk_overlap
        self.long_caption_prepend_title = long_caption_prepend_title
        self._check_columns()

    def _check_columns(self):
        if self.long_caption_mode not in ("none", "random-chunk"):
            raise ValueError(f"Unsupported long_caption_mode: {self.long_caption_mode}")
        if self.long_caption_mode != "none" and not self.long_caption_key:
            raise ValueError("long_caption_key is required when long_caption_mode is not 'none'.")

        required = [self.image_key, self.caption_key]
        if self.long_caption_mode != "none":
            required.append(self.long_caption_key)
        missing = [key for key in required if key not in self.df.columns]
        if missing:
            raise KeyError(f"{self.tsv_path} missing required columns: {missing}")
        if self.long_caption_mode != "none" and not hasattr(self.tokenizer, "encode"):
            raise TypeError("Long caption chunking requires a tokenizer with an encode() method.")
        if self.long_caption_chunk_tokens <= 0:
            raise ValueError("long_caption_chunk_tokens must be positive.")
        if self.long_caption_chunk_overlap < 0:
            raise ValueError("long_caption_chunk_overlap must be non-negative.")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _string_value(value):
        if pd.isna(value):
            return ""
        return str(value)

    def _decode_tokens(self, token_ids):
        if not token_ids:
            return ""
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids).strip()
        return ""

    def _random_token_chunk(self, text, token_budget):
        token_ids = self.tokenizer.encode(text)
        if len(token_ids) <= token_budget:
            return self._decode_tokens(token_ids)

        overlap = min(self.long_caption_chunk_overlap, max(token_budget - 1, 0))
        stride = max(1, token_budget - overlap)
        last_start = len(token_ids) - token_budget
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        start = random.choice(starts)
        return self._decode_tokens(token_ids[start : start + token_budget])

    def _build_training_text(self, row):
        title = self._string_value(row[self.caption_key])
        if self.long_caption_mode == "none":
            return title

        long_caption = self._string_value(row[self.long_caption_key])
        if not long_caption:
            return title

        context_length = getattr(self.tokenizer, "context_length", None) or 77
        prefix = f"{title}. " if self.long_caption_prepend_title and title else ""
        prefix_tokens = self.tokenizer.encode(prefix) if prefix else []
        token_budget = min(
            self.long_caption_chunk_tokens,
            max(context_length - 2 - len(prefix_tokens), 0),
        )
        if token_budget <= 0:
            return title

        chunk = self._random_token_chunk(long_caption, token_budget)
        return f"{prefix}{chunk}".strip()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(str(row[self.image_key])).convert("RGB")
        image = self.transform(image)
        text = self.tokenizer([self._build_training_text(row)])[0]
        return {"image": image, "text": text}


class CsvImageLabelDataset(Dataset):
    def __init__(self, tsv_path, image_key, label_key, transform, class_to_idx, sep="\t"):
        self.tsv_path = Path(tsv_path)
        self.df = pd.read_csv(self.tsv_path, sep=sep)
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.class_to_idx = class_to_idx
        self._check_columns()

    def _check_columns(self):
        missing = [key for key in (self.image_key, self.label_key) if key not in self.df.columns]
        if missing:
            raise KeyError(f"{self.tsv_path} missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = str(row[self.label_key])
        image = Image.open(str(row[self.image_key])).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "target": torch.tensor(self.class_to_idx[label], dtype=torch.long),
            "label": label,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OpenCLIP on TSV data and run zero-shot test after every epoch."
    )
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--csv-img-key", default="filepath")
    parser.add_argument("--csv-caption-key", default="title")
    parser.add_argument("--csv-separator", default="\t")
    parser.add_argument("--template", default="{name}")
    parser.add_argument(
        "--train-long-caption-key",
        default=None,
        help="Optional TSV column for long captions used only by contrastive training.",
    )
    parser.add_argument(
        "--train-long-caption-mode",
        choices=["none", "random-chunk"],
        default="none",
        help="Use random chunks from --train-long-caption-key during contrastive training.",
    )
    parser.add_argument(
        "--long-caption-chunk-tokens",
        type=int,
        default=75,
        help="Maximum CLIP BPE tokens sampled from each long caption before SOT/EOT.",
    )
    parser.add_argument(
        "--long-caption-chunk-overlap",
        type=int,
        default=8,
        help="Token overlap between candidate long-caption chunks.",
    )
    parser.add_argument(
        "--long-caption-prepend-title",
        action="store_true",
        help="Prefix sampled long-caption chunks with the short title/class caption.",
    )
    parser.add_argument(
        "--train-objective",
        choices=["contrastive", "text-prototype"],
        default="contrastive",
        help=(
            "contrastive: standard CLIP image-text pair loss. "
            "text-prototype: classify images against all class text embeddings with cross entropy."
        ),
    )
    parser.add_argument("--logs", default="./logs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size per process/GPU.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Eval batch size per process/GPU.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--wd", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--lr-scheduler", choices=["cosine", "const", "const-cooldown"], default="cosine")
    parser.add_argument("--epochs-cooldown", type=int, default=None)
    parser.add_argument("--lr-cooldown-end", type=float, default=0.0)
    parser.add_argument("--lr-cooldown-power", type=float, default=1.0)
    parser.add_argument("--skip-scheduler", action="store_true")
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp_bf16",
    )
    parser.add_argument("--local-loss", action="store_true")
    parser.add_argument("--gather-with-grad", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--force-quick-gelu", action="store_true")
    parser.add_argument("--force-image-size", type=int, default=None)
    parser.add_argument("--force-context-length", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--pretrained-image", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Optional training checkpoint to resume from.")
    parser.add_argument("--save-epoch-copy", action="store_true", help="Also keep a copy of each new best epoch.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    return parser.parse_args()


def decode_separator(value):
    return "\t" if value == "\\t" else value


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
    else:
        args.device = "cpu"
    dist.init_process_group(backend=backend)


def cleanup_distributed(args):
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return getattr(args, "rank", 0) == 0


def broadcast_value(args, value):
    if not getattr(args, "distributed", False):
        return value
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def set_seed(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir, args):
    logger = logging.getLogger("open_clip_train.train_eval_csv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if not is_main_process(args):
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def default_name(args):
    model_name = args.model.replace("/", "-")
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    return "-".join(
        [
            date_str,
            f"model_{model_name}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
            f"obj_{args.train_objective}",
            "train_eval_iid",
        ]
    )


def resolve_output_dir(args):
    if args.name is None and is_main_process(args):
        args.name = default_name(args)
    args.name = broadcast_value(args, args.name)
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.logs) / args.name


def append_csv(path, fieldnames, row):
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fieldnames})


def save_json(path, obj):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_class_names(tsv_path, label_key, sep):
    df = pd.read_csv(tsv_path, sep=sep, usecols=[label_key])
    return sorted(df[label_key].astype(str).unique().tolist())


def topk_correct(logits, labels, k):
    k = min(k, logits.shape[-1])
    preds = logits.topk(k, dim=1).indices
    return preds.eq(labels.view(-1, 1)).any(dim=1).sum().item()


def classification_metrics(confusion):
    num_classes = confusion.shape[0]
    support = confusion.sum(axis=1)
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)

    for idx in range(num_classes):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision[idx] = tp / (tp + fp) if (tp + fp) else 0.0
        recall[idx] = tp / (tp + fn) if (tp + fn) else 0.0
        f1[idx] = 2 * precision[idx] * recall[idx] / (precision[idx] + recall[idx]) if (
            precision[idx] + recall[idx]
        ) else 0.0

    weights = support / support.sum() if support.sum() else np.zeros_like(support, dtype=np.float64)
    return {
        "macro_precision": float(precision.mean()) if num_classes else 0.0,
        "macro_recall": float(recall.mean()) if num_classes else 0.0,
        "macro_f1": float(f1.mean()) if num_classes else 0.0,
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.astype(int).tolist(),
        },
    }


def save_per_class_report(path, classes, per_class):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for idx, name in enumerate(classes):
            writer.writerow(
                [
                    name,
                    per_class["precision"][idx],
                    per_class["recall"][idx],
                    per_class["f1"][idx],
                    per_class["support"][idx],
                ]
            )


def maybe_progress(iterable, desc, enabled):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def create_optimizer(model, args):
    default_params = get_default_params(args.model)
    beta1 = args.beta1 if args.beta1 is not None else default_params["beta1"]
    beta2 = args.beta2 if args.beta2 is not None else default_params["beta2"]
    eps = args.eps if args.eps is not None else default_params["eps"]

    def exclude(name, param):
        return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if not exclude(n, p) and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(beta1, beta2),
        eps=eps,
    )


def create_scheduler(optimizer, train_loader, args):
    if args.skip_scheduler:
        return None
    total_steps = len(train_loader) * args.epochs
    if args.lr_scheduler == "cosine":
        return cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    if args.lr_scheduler == "const":
        return const_lr(optimizer, args.lr, args.warmup, total_steps)
    if args.epochs_cooldown is None:
        raise ValueError("--epochs-cooldown is required for const-cooldown scheduler.")
    cooldown_steps = len(train_loader) * args.epochs_cooldown
    return const_lr_cooldown(
        optimizer,
        args.lr,
        args.warmup,
        total_steps,
        cooldown_steps,
        args.lr_cooldown_power,
        args.lr_cooldown_end,
    )


def compute_clip_loss_and_logits(loss_fn, image_features, text_features, logit_scale, logit_bias=None):
    logits_per_image, logits_per_text = loss_fn.get_logits(
        image_features,
        text_features,
        logit_scale,
        logit_bias=logit_bias,
    )
    labels = loss_fn.get_ground_truth(image_features.device, logits_per_image.shape[0])
    loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    return loss, logits_per_image, labels


def metric_rows_for_local_rank(logits_per_image, labels, batch_size, args, local_loss):
    if args.world_size <= 1 or local_loss:
        return logits_per_image, labels
    start = args.rank * batch_size
    end = start + batch_size
    return logits_per_image[start:end], labels[start:end]


def reduce_train_metrics(loss_sum, seen, top1_correct, top5_correct, device, args):
    values = torch.tensor([loss_sum, seen, top1_correct, top5_correct], dtype=torch.float64, device=device)
    if args.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    loss_sum, seen, top1_correct, top5_correct = values.tolist()
    return {
        "loss": loss_sum / max(seen, 1.0),
        "top1": top1_correct / max(seen, 1.0),
        "top5": top5_correct / max(seen, 1.0),
    }


def train_one_epoch(model, loader, optimizer, scaler, scheduler, loss_fn, epoch, args, progress):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    loss_sum = 0.0
    seen = 0
    top1_correct = 0
    top5_correct = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    h2d_time_sum = 0.0
    compute_time_sum = 0.0
    timed_steps = 0
    end = time.time()

    for step_in_epoch, batch in enumerate(maybe_progress(loader, f"train epoch {epoch}", progress)):
        data_time = time.time() - end
        step_start = time.time()
        global_step = (epoch - 1) * len(loader) + step_in_epoch
        if scheduler is not None:
            scheduler(global_step)

        h2d_start = time.time()
        images = batch["image"].to(device=device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        texts = batch["text"].to(device=device, non_blocking=True)
        h2d_time = time.time() - h2d_start

        optimizer.zero_grad(set_to_none=True)
        compute_start = time.time()
        with autocast():
            model_out = model(image=images, text=texts)
            loss, logits_per_image, labels = compute_clip_loss_and_logits(
                loss_fn,
                model_out["image_features"],
                model_out["text_features"],
                model_out["logit_scale"],
                logit_bias=model_out.get("logit_bias"),
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip_norm)
            optimizer.step()

        unwrap_model(model).logit_scale.data.clamp_(0, math.log(100))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_time = time.time() - compute_start

        metric_logits, metric_labels = metric_rows_for_local_rank(
            logits_per_image.detach(),
            labels.detach(),
            images.shape[0],
            args,
            loss_fn.local_loss,
        )
        batch_seen = metric_labels.numel()
        loss_sum += loss.item() * batch_seen
        seen += batch_seen
        top1_correct += topk_correct(metric_logits.float(), metric_labels, 1)
        top5_correct += topk_correct(metric_logits.float(), metric_labels, 5)
        step_time = time.time() - step_start
        data_time_sum += data_time
        step_time_sum += step_time
        h2d_time_sum += h2d_time
        compute_time_sum += compute_time
        timed_steps += 1

        if is_main_process(args) and (
            step_in_epoch % args.log_every_n_steps == 0 or step_in_epoch + 1 == len(loader)
        ):
            _LOGGER.info(
                "Train epoch %d step %d/%d | loss=%.5f lr=%.6g | "
                "avg_data_wait=%.2fs avg_h2d=%.2fs avg_compute=%.2fs avg_step=%.2fs",
                epoch,
                step_in_epoch + 1,
                len(loader),
                loss.item(),
                optimizer.param_groups[0]["lr"],
                data_time_sum / max(timed_steps, 1),
                h2d_time_sum / max(timed_steps, 1),
                compute_time_sum / max(timed_steps, 1),
                step_time_sum / max(timed_steps, 1),
            )
            data_time_sum = 0.0
            step_time_sum = 0.0
            h2d_time_sum = 0.0
            compute_time_sum = 0.0
            timed_steps = 0
        end = time.time()

    return reduce_train_metrics(loss_sum, seen, top1_correct, top5_correct, device, args)


def train_one_epoch_text_prototype(model, loader, optimizer, scaler, scheduler, class_tokens, epoch, args, progress):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    loss_sum = 0.0
    seen = 0
    top1_correct = 0
    top5_correct = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    h2d_time_sum = 0.0
    compute_time_sum = 0.0
    timed_steps = 0
    end = time.time()

    for step_in_epoch, batch in enumerate(maybe_progress(loader, f"train epoch {epoch}", progress)):
        data_time = time.time() - end
        step_start = time.time()
        global_step = (epoch - 1) * len(loader) + step_in_epoch
        if scheduler is not None:
            scheduler(global_step)

        h2d_start = time.time()
        images = batch["image"].to(device=device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        targets = batch["target"].to(device=device, non_blocking=True)
        tokens = class_tokens.to(device=device, non_blocking=True)
        h2d_time = time.time() - h2d_start

        optimizer.zero_grad(set_to_none=True)
        compute_start = time.time()
        with autocast():
            model_out = model(image=images, text=tokens)
            logits = model_out["logit_scale"] * model_out["image_features"] @ model_out["text_features"].t()
            if model_out.get("logit_bias") is not None:
                logits = logits + model_out["logit_bias"]
            loss = F.cross_entropy(logits.float(), targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip_norm)
            optimizer.step()

        unwrap_model(model).logit_scale.data.clamp_(0, math.log(100))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_time = time.time() - compute_start

        batch_seen = targets.numel()
        logits = logits.detach().float()
        loss_sum += loss.item() * batch_seen
        seen += batch_seen
        top1_correct += topk_correct(logits, targets, 1)
        top5_correct += topk_correct(logits, targets, 5)
        step_time = time.time() - step_start
        data_time_sum += data_time
        step_time_sum += step_time
        h2d_time_sum += h2d_time
        compute_time_sum += compute_time
        timed_steps += 1

        if is_main_process(args) and (
            step_in_epoch % args.log_every_n_steps == 0 or step_in_epoch + 1 == len(loader)
        ):
            _LOGGER.info(
                "Train epoch %d step %d/%d | loss=%.5f lr=%.6g | "
                "avg_data_wait=%.2fs avg_h2d=%.2fs avg_compute=%.2fs avg_step=%.2fs",
                epoch,
                step_in_epoch + 1,
                len(loader),
                loss.item(),
                optimizer.param_groups[0]["lr"],
                data_time_sum / max(timed_steps, 1),
                h2d_time_sum / max(timed_steps, 1),
                compute_time_sum / max(timed_steps, 1),
                step_time_sum / max(timed_steps, 1),
            )
            data_time_sum = 0.0
            step_time_sum = 0.0
            h2d_time_sum = 0.0
            compute_time_sum = 0.0
            timed_steps = 0
        end = time.time()

    return reduce_train_metrics(loss_sum, seen, top1_correct, top5_correct, device, args)


@torch.no_grad()
def build_text_features(model, tokenizer, class_names, template, batch_size, device, args):
    model.eval()
    autocast = get_autocast(args.precision, device_type=device.type)
    features = []
    for start in range(0, len(class_names), batch_size):
        names = class_names[start : start + batch_size]
        prompts = [template.format(name=name) for name in names]
        tokens = tokenizer(prompts).to(device=device, non_blocking=True)
        with autocast():
            text_features = model.encode_text(tokens, normalize=True)
        features.append(text_features)
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate_zero_shot(model, loader, tokenizer, class_names, epoch, args, progress):
    device = torch.device(args.device)
    unwrapped = unwrap_model(model)
    unwrapped.eval()
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    eval_batch_size = args.eval_batch_size or args.batch_size
    text_features = build_text_features(
        unwrapped,
        tokenizer,
        class_names,
        args.template,
        eval_batch_size,
        device,
        args,
    )
    logit_scale = unwrapped.logit_scale.exp()
    logit_bias = getattr(unwrapped, "logit_bias", None)

    total_loss = 0.0
    total_seen = 0
    total_top1 = 0
    total_top5 = 0
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for batch in maybe_progress(loader, f"test epoch {epoch}", progress):
        images = batch["image"].to(device=device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        targets = batch["target"].to(device=device, non_blocking=True)

        with autocast():
            image_features = unwrapped.encode_image(images, normalize=True)
            logits = logit_scale * image_features @ text_features.t()
            if logit_bias is not None:
                logits = logits + logit_bias
        logits = logits.float()
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


def load_resume(path, model, optimizer, scaler):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if next(iter(state_dict.keys())).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("best_epoch", 0)), float(
        checkpoint.get("best_top1", -math.inf)
    )


def save_checkpoint(path, model, optimizer, scaler, epoch, best_epoch, best_top1, args, metrics):
    checkpoint = {
        "epoch": epoch,
        "name": args.name,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "metrics": metrics,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    args.csv_separator = decode_separator(args.csv_separator)
    setup_distributed(args)
    set_seed(args.seed, args.rank)

    output_dir = resolve_output_dir(args)
    checkpoint_dir = output_dir / "checkpoints"
    if is_main_process(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed:
        dist.barrier()

    logger = setup_logging(output_dir, args)
    progress = is_main_process(args) and not args.no_progress
    if progress and tqdm is None:
        logger.warning("tqdm is not installed; progress bars are disabled.")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    logger.info(
        "Starting train/eval | rank=%d world_size=%d device=%s output_dir=%s",
        args.rank,
        args.world_size,
        args.device,
        output_dir,
    )
    logger.info("Train data: %s", args.train_data)
    logger.info("Test data: %s", args.test_data)

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        force_quick_gelu=args.force_quick_gelu,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    tokenizer = open_clip.get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    train_class_names = read_class_names(args.train_data, args.csv_caption_key, args.csv_separator)
    test_class_names = read_class_names(args.test_data, args.csv_caption_key, args.csv_separator)
    missing_test_classes = sorted(set(test_class_names) - set(train_class_names))
    if missing_test_classes:
        raise ValueError(
            "test data contains labels not present in train data: "
            f"{missing_test_classes[:20]}"
        )
    class_names = train_class_names
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_tokens = tokenizer([args.template.format(name=name) for name in class_names])
    if is_main_process(args):
        save_json(output_dir / "class_to_idx.json", class_to_idx)
        save_json(output_dir / "args.json", vars(args))

    if args.train_objective == "text-prototype":
        train_dataset = CsvImageLabelDataset(
            args.train_data,
            args.csv_img_key,
            args.csv_caption_key,
            preprocess_train,
            class_to_idx,
            sep=args.csv_separator,
        )
    else:
        train_dataset = CsvImageTextDataset(
            args.train_data,
            args.csv_img_key,
            args.csv_caption_key,
            preprocess_train,
            tokenizer,
            sep=args.csv_separator,
            long_caption_key=args.train_long_caption_key,
            long_caption_mode=args.train_long_caption_mode,
            long_caption_chunk_tokens=args.long_caption_chunk_tokens,
            long_caption_chunk_overlap=args.long_caption_chunk_overlap,
            long_caption_prepend_title=args.long_caption_prepend_title,
        )
    test_dataset = CsvImageLabelDataset(
        args.test_data,
        args.csv_img_key,
        args.csv_caption_key,
        preprocess_val,
        class_to_idx,
        sep=args.csv_separator,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
    ) if args.distributed else None
    test_dataset_for_loader = (
        Subset(test_dataset, list(range(args.rank, len(test_dataset), args.world_size)))
        if args.distributed
        else test_dataset
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=args.train_objective == "contrastive",
    )
    eval_batch_size = args.eval_batch_size or args.batch_size
    test_loader = DataLoader(
        test_dataset_for_loader,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.eval_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.eval_workers > 0,
        prefetch_factor=args.prefetch_factor if args.eval_workers > 0 else None,
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader has 0 batches. Reduce --batch-size or check train data.")

    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, train_loader, args)
    scaler = None
    if args.precision == "amp" and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_epoch = 0
    best_top1 = -math.inf
    if args.resume:
        start_epoch, best_epoch, best_top1 = load_resume(args.resume, model, optimizer, scaler)
        logger.info("Resumed checkpoint %s at epoch %d", args.resume, start_epoch)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None)

    loss_fn = None
    if args.train_objective == "contrastive":
        loss_fn = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )

    logger.info(
        "Loaded data | train_samples=%d test_samples=%d train_classes=%d test_classes=%d "
        "train_batches_per_rank=%d objective=%s template=%s",
        len(train_dataset),
        len(test_dataset),
        len(class_names),
        len(test_class_names),
        len(train_loader),
        args.train_objective,
        args.template,
    )

    total_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        logger.info("Epoch %d/%d started", epoch, args.epochs)
        epoch_start = time.time()

        train_start = time.time()
        if args.train_objective == "text-prototype":
            train_metrics = train_one_epoch_text_prototype(
                model,
                train_loader,
                optimizer,
                scaler,
                scheduler,
                class_tokens,
                epoch,
                args,
                progress,
            )
        else:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                scheduler,
                loss_fn,
                epoch,
                args,
                progress,
            )
        train_seconds = time.time() - train_start

        test_start = time.time()
        test_metrics, confusion = evaluate_zero_shot(
            model,
            test_loader,
            tokenizer,
            class_names,
            epoch,
            args,
            progress,
        )
        test_seconds = time.time() - test_start
        epoch_seconds = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "test_loss": test_metrics["loss"],
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_precision": test_metrics["weighted_precision"],
            "test_weighted_recall": test_metrics["weighted_recall"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        }
        time_row = {
            "epoch": epoch,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
            "epoch_seconds": epoch_seconds,
            "total_elapsed_seconds": total_elapsed,
        }

        if is_main_process(args):
            append_csv(output_dir / "metrics.csv", METRICS_FIELDS, row)
            append_csv(output_dir / "time.csv", TIME_FIELDS, time_row)
            save_json(output_dir / "latest_metrics.json", row)
            np.save(output_dir / "latest_confusion_matrix.npy", confusion)
            save_per_class_report(output_dir / "latest_per_class_report.csv", class_names, test_metrics["per_class"])

        logger.info(
            "Epoch %03d finished | train_loss=%.4f train_top1=%.4f train_top5=%.4f | "
            "test_loss=%.4f test_top1=%.4f test_top5=%.4f macro_f1=%.4f weighted_f1=%.4f | "
            "time train=%.1fs test=%.1fs epoch=%.1fs elapsed=%.1fs",
            epoch,
            row["train_loss"],
            row["train_top1"],
            row["train_top5"],
            row["test_loss"],
            row["test_top1"],
            row["test_top5"],
            row["test_macro_f1"],
            row["test_weighted_f1"],
            train_seconds,
            test_seconds,
            epoch_seconds,
            total_elapsed,
        )

        if row["test_top1"] > best_top1:
            best_top1 = row["test_top1"]
            best_epoch = epoch
            if is_main_process(args):
                best_path = checkpoint_dir / "best.pt"
                save_checkpoint(best_path, model, optimizer, scaler, epoch, best_epoch, best_top1, args, row)
                if args.save_epoch_copy:
                    save_checkpoint(
                        checkpoint_dir / f"best_epoch_{epoch:03d}.pt",
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        best_epoch,
                        best_top1,
                        args,
                        row,
                    )
                save_json(
                    output_dir / "best_metrics.json",
                    {
                        **row,
                        "best_epoch": best_epoch,
                        "best_top1": best_top1,
                        "selection_metric": "test_top1",
                        "checkpoint": str(best_path),
                    },
                )
            logger.info("New best checkpoint at epoch %d | test_top1=%.4f", best_epoch, best_top1)

        if is_main_process(args):
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scaler, epoch, best_epoch, best_top1, args, row)

        if args.distributed:
            dist.barrier()

    logger.info(
        "Training finished | best_epoch=%d best_test_top1=%.4f checkpoint=%s total_time=%.1fs",
        best_epoch,
        best_top1,
        checkpoint_dir / "best.pt",
        time.time() - total_start,
    )
    cleanup_distributed(args)


if __name__ == "__main__":
    main()
