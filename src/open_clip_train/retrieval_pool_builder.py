"""Build a retrieval pool (>=100 samples) where the fine-tuned checkpoint's R@1
beats the DFN2B backbone's R@1.

Setup
-----
On the full keep_extracted figure set (22,275 images, 1:1 with keep.jsonl via
``image_id``) we compute the full N x N cosine-similarity matrix twice: once with a
fine-tuned checkpoint (model A, e.g. epoch 20) and once with the bare DFN2B backbone
(model B). A pool P is then selected greedily: at each step we add the sample that
maximises the increase in A's top-1 hit count minus B's top-1 hit count (I2T and T2I
directions combined), then pick from the trajectory the largest pool of size >=
``--min-size`` on which A's R@1 > B's R@1 in both directions (falling back to the
largest pool where the mean R@1 wins).

A sample i is an I2T top-1 hit inside pool P iff its self-similarity S[i,i] strictly
exceeds every S[i,j] for j in P, j != i (tie-aware: ties count against). T2I uses the
transposed condition. Adding a candidate j to P therefore changes hit counts exactly by
  +1 if j is itself a hit (its best match inside P is itself), and
  -1 per currently-hit query i whose competing score vs j reaches S[i,i].

Outputs (under --output-dir, default <project>/retrieval/result)
---------------------------------------------------------------
  pool_images/            copies of all pool sample images
  hits_epoch20_i2t/       pool images whose epoch20 I2T top-1 is correct (within pool)
  hits_epoch20_t2i/       pool images whose epoch20 T2I top-1 is correct (within pool)
  hits_dfn_i2t/ hits_dfn_t2i/   same but for the DFN2B backbone
  pool.jsonl              per-sample details (text, both models' I2T/T2I ranks + hits)
  summary.txt             R@1 comparison on the chosen pool and on the full pool

Example
-------
PYTHONPATH=src python -m open_clip_train.retrieval_pool_builder \
    --epoch 20 --target-size 200 --min-size 100 --output-dir retrieval/result --device cuda:0
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

_LOGGER = logging.getLogger("open_clip_train.retrieval_pool_builder")


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
    parser.add_argument("--target-size", type=int, default=200, help="Greedy stops at this pool size.")
    parser.add_argument("--min-size", type=int, default=100, help="Pool must be at least this large.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the pool (default: <project>/retrieval/result).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--precision", default="amp_bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-context-length", type=int, default=256)
    parser.add_argument("--force-quick-gelu", action="store_true")
    parser.add_argument("--force-image-size", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--pretrained-image", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------- rank helpers

def pool_hits(sub, d):
    """Top-1 hit flags for pool members whose self column is included in ``sub``.

    ``sub`` is the (p, p) self-similarity submatrix restricted to the pool; ``d`` the
    diagonal. A member i is an I2T hit iff d[i] strictly exceeds the second-largest
    value of row i (ties count against). ``sub`` may be transposed for T2I.
    """
    p = sub.shape[0]
    if p == 0:
        return np.zeros(0, dtype=bool)
    if p == 1:
        return np.ones(1, dtype=bool)
    idx = np.argpartition(sub, -2, axis=1)[:, -2:]
    vals = np.take_along_axis(sub, idx, axis=1)
    vals.sort(axis=1)  # vals[:,0]=2nd largest, vals[:,1]=largest
    return (vals[:, 1] == d) & (vals[:, 0] < d)


def gained_hits(row, d):
    """Hit flags for candidates (not in the pool) scored against the pool.

    ``row`` is (M, p): each candidate's similarities to the p pool members. A candidate
    is a hit iff d[j] strictly exceeds every value in its row (no self column present).
    """
    if row.shape[1] == 0:
        return np.zeros(row.shape[0], dtype=bool)
    return row.max(axis=1) < d


def full_pool_r1(S, d):
    """I2T / T2I R@1 over the entire N x N matrix (reference numbers)."""
    n = S.shape[0]
    i2t = float(((S >= d[:, None]).sum(1) == 1).mean())
    t2i = float(((S >= d[None, :]).sum(0) == 1).mean())
    return i2t, t2i


# ---------------------------------------------------------------- pool selection

def greedy_select(S_A, S_B, dA, dB, target_size, min_size, rng):
    """Greedy forward selection maximizing A's hit-count lead over B."""
    n = S_A.shape[0]
    in_pool = np.zeros(n, dtype=bool)
    P = []  # list of chosen indices
    trajectory = []  # (pool_snapshot, n_hits_i2t_A, n_hits_i2t_B, n_hits_t2i_A, n_hits_t2i_B)

    while len(P) < target_size and len(P) < n:
        pool = np.asarray(P, dtype=np.int64) if P else np.empty(0, dtype=np.int64)
        # Current hits inside the pool (recomputed from scratch each step -> exact).
        hA_i2t, hA_t2i, hB_i2t, hB_t2i = (np.zeros(0, dtype=bool),) * 4
        if pool.size:
            subA = S_A[np.ix_(pool, pool)]
            subB = S_B[np.ix_(pool, pool)]
            hA_i2t, hA_t2i = pool_hits(subA, dA[pool]), pool_hits(subA.T, dA[pool])
            hB_i2t, hB_t2i = pool_hits(subB, dB[pool]), pool_hits(subB.T, dB[pool])
        trajectory.append(
            (
                list(P),
                int(hA_i2t.sum()), int(hB_i2t.sum()),
                int(hA_t2i.sum()), int(hB_t2i.sum()),
            )
        )
        _LOGGER.info(
            "Step %3d | hits A(i2t/t2i)=%d/%d  B(i2t/t2i)=%d/%d",
            len(P), hA_i2t.sum(), hA_t2i.sum(), hB_i2t.sum(), hB_t2i.sum(),
        )

        cands = np.nonzero(~in_pool)[0]
        if cands.size == 0:
            break

        # Gained hits for every candidate against the current pool.
        gA_i2t = gained_hits(S_A[cands][:, pool], dA[cands]) if pool.size else np.ones(cands.size, bool)
        gA_t2i = gained_hits(S_A[pool][:, cands].T, dA[cands]) if pool.size else np.ones(cands.size, bool)
        gB_i2t = gained_hits(S_B[cands][:, pool], dB[cands]) if pool.size else np.ones(cands.size, bool)
        gB_t2i = gained_hits(S_B[pool][:, cands].T, dB[cands]) if pool.size else np.ones(cands.size, bool)

        # Lost hits among currently-hit pool members when the candidate joins.
        lA_i2t = np.zeros(cands.size)
        lA_t2i = np.zeros(cands.size)
        lB_i2t = np.zeros(cands.size)
        lB_t2i = np.zeros(cands.size)
        if pool.size:
            if hA_i2t.any():
                hidx = pool[hA_i2t]
                lA_i2t = (S_A[hidx][:, cands] >= dA[hidx][:, None]).sum(0)
            if hA_t2i.any():
                hidx = pool[hA_t2i]
                lA_t2i = (S_A[cands][:, hidx] >= dA[hidx][None, :]).sum(1)
            if hB_i2t.any():
                hidx = pool[hB_i2t]
                lB_i2t = (S_B[hidx][:, cands] >= dB[hidx][:, None]).sum(0)
            if hB_t2i.any():
                hidx = pool[hB_t2i]
                lB_t2i = (S_B[cands][:, hidx] >= dB[hidx][None, :]).sum(1)

        gain_A = gA_i2t.astype(float) - lA_i2t + gA_t2i.astype(float) - lA_t2i
        gain_B = gB_i2t.astype(float) - lB_i2t + gB_t2i.astype(float) - lB_t2i
        score = gain_A - gain_B

        # First seed: the sample with the largest self-similarity lead of A over B.
        if not P:
            seed = int(np.argmax(dA - dB))
            best = int(np.argmax(cands == seed))
        else:
            best = int(score.argmax())
        chosen = int(cands[best])
        P.append(chosen)
        in_pool[chosen] = True

    return trajectory


def choose_pool(trajectory, min_size):
    """Pick the largest pool >= min_size where A wins in both directions (fallback: mean)."""
    both = []
    mean = []
    for snap, a_i, b_i, a_t, b_t in trajectory:
        size = len(snap)
        if size < min_size:
            continue
        # All four already positive at full hit counts; require strict R@1 leads.
        r1_a_i, r1_b_i = a_i / size, b_i / size
        r1_a_t, r1_b_t = a_t / size, b_t / size
        if r1_a_i > r1_b_i and r1_a_t > r1_b_t:
            both.append((size, snap, (r1_a_i, r1_b_i, r1_a_t, r1_b_t)))
        elif 0.5 * (r1_a_i + r1_a_t) > 0.5 * (r1_b_i + r1_b_t):
            mean.append((size, snap, (r1_a_i, r1_b_i, r1_a_t, r1_b_t)))
    if both:
        both.sort(key=lambda x: x[0])
        return both[-1]
    if mean:
        mean.sort(key=lambda x: x[0])
        return mean[-1]
    return None


def pool_metrics(S, d, pool):
    """(hits_i2t, hits_t2i, ranks_i2t, ranks_t2i) for the pool members."""
    pool = np.asarray(pool, dtype=np.int64)
    sub = S[np.ix_(pool, pool)]
    dd = d[pool]
    hit_i2t = pool_hits(sub, dd)
    hit_t2i = pool_hits(sub.T, dd)
    # Tie-aware rank of the diagonal match: 1 + # others strictly above self.
    rank_i2t = ((sub > dd[:, None]).sum(1) + 1).astype(int)
    rank_t2i = ((sub.T > dd[:, None]).sum(1) + 1).astype(int)
    return hit_i2t, hit_t2i, rank_i2t, rank_t2i


# -------------------------------------------------------------------------- main

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
    else:
        checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{args.epoch:03d}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[2] / "retrieval" / "result"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "build_pool.log", encoding="utf-8"),
        ],
    )

    records = load_text_records(args.jsonl, [args.text_field])
    pairs = discover_pairs(Path(args.images_dir), records)
    pairs_by_id = {p["image_id"]: p["text"] for p in pairs}
    _LOGGER.info("Matched pairs=%d checkpoint=%s", len(pairs), checkpoint_path)

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
    samples, token_lengths = load_valid_samples(pairs, preprocess_val, tokenizer, args.text_field)
    num_truncated = sum(1 for n in token_lengths if n >= args.force_context_length)
    _LOGGER.info("Preprocessed samples=%d truncated_texts=%d", len(samples), num_truncated)

    def compute_sim(load_checkpoint, label):
        # Fresh model instance per call: without this, the second pass would still
        # carry the checkpoint loaded into the shared model and "backbone" would
        # silently equal the fine-tuned model.
        model, _, _ = open_clip.create_model_and_transforms(
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
        if load_checkpoint:
            ckpt, missing, unexpected = load_model_checkpoint(model, checkpoint_path, strict=not args.non_strict)
            _LOGGER.info(
                "Loaded %s | file_epoch=%03d ckpt_epoch=%s missing=%d unexpected=%d",
                label, args.epoch,
                ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown",
                len(missing), len(unexpected),
            )
        else:
            _LOGGER.info("Using bare DFN2B backbone (%s), no checkpoint loaded.", label)
        t0 = time.time()
        images, texts = encode_retrieval_features(model, samples, args.batch_size, device, args.precision)
        sim = (images.to(device) @ texts.to(device).t()).float().cpu().numpy()
        d = np.diag(sim).copy()
        _LOGGER.info("Encoded %s | sim=%dx%d time=%.1fs", label, sim.shape[0], sim.shape[1], time.time() - t0)
        return sim, d

    S_A, dA = compute_sim(True, "epoch checkpoint (A)")
    S_B, dB = compute_sim(False, "DFN2B backbone (B)")

    # Save matrices + per-sample top-1 agreement diagnostics (also used if greedy fails).
    np.savez(output_dir / "matrices.npz", S_A=S_A, S_B=S_B, dA=dA, dB=dB)
    _LOGGER.info("Saved similarity matrices to %s/matrices.npz", output_dir)
    for label, a_hit, b_hit in (
        ("i2t", (S_A >= dA[:, None]).sum(1) == 1, (S_B >= dB[:, None]).sum(1) == 1),
        ("t2i", (S_A >= dA[None, :]).sum(0) == 1, (S_B >= dB[None, :]).sum(0) == 1),
    ):
        a_only = int((a_hit & ~b_hit).sum())
        b_only = int((~a_hit & b_hit).sum())
        both = int((a_hit & b_hit).sum())
        _LOGGER.info(
            "Full-pool top-1 agreement [%s] | A-only=%d B-only=%d both=%d",
            label, a_only, b_only, both,
        )

    r1_a_full = full_pool_r1(S_A, dA)
    r1_b_full = full_pool_r1(S_B, dB)
    _LOGGER.info(
        "Full-pool R@1 | A i2t=%.4f t2i=%.4f | B i2t=%.4f t2i=%.4f",
        r1_a_full[0], r1_a_full[1], r1_b_full[0], r1_b_full[1],
    )

    trajectory = greedy_select(
        S_A, S_B, dA, dB, args.target_size, args.min_size,
        rng=np.random.default_rng(0),
    )
    choice = choose_pool(trajectory, args.min_size)
    if choice is None:
        raise RuntimeError("No pool of size >= %d where epoch R@1 > DFN R@1 was found." % args.min_size)
    size, pool, (r1_a_i, r1_b_i, r1_a_t, r1_b_t) = choice
    pool = list(pool)
    _LOGGER.info(
        "Chosen pool | size=%d  A i2t_r1=%.4f B i2t_r1=%.4f  A t2i_r1=%.4f B t2i_r1=%.4f",
        size, r1_a_i, r1_b_i, r1_a_t, r1_b_t,
    )

    hitA_i2t, hitA_t2i, rankA_i2t, rankA_t2i = pool_metrics(S_A, dA, pool)
    hitB_i2t, hitB_t2i, rankB_i2t, rankB_t2i = pool_metrics(S_B, dB, pool)

    # ---- write outputs
    (output_dir / "pool_images").mkdir(exist_ok=True)
    for dname, flag in (
        ("hits_epoch20_i2t", hitA_i2t), ("hits_epoch20_t2i", hitA_t2i),
        ("hits_dfn_i2t", hitB_i2t), ("hits_dfn_t2i", hitB_t2i),
    ):
        (output_dir / dname).mkdir(exist_ok=True)
    for pos, i in enumerate(pool):
        sid = samples[i]
        img_name = sid["image_id"] + ".jpg"
        shutil.copy2(sid["image_path"], output_dir / "pool_images" / f"{pos:04d}_{img_name}")
        for dname, flag in (
            ("hits_epoch20_i2t", hitA_i2t), ("hits_epoch20_t2i", hitA_t2i),
            ("hits_dfn_i2t", hitB_i2t), ("hits_dfn_t2i", hitB_t2i),
        ):
            if flag[pos]:
                shutil.copy2(sid["image_path"], output_dir / dname / f"{pos:04d}_{img_name}")

    with (output_dir / "pool.jsonl").open("w", encoding="utf-8") as f:
        for pos, i in enumerate(pool):
            rec = {
                "image_id": samples[i]["image_id"],
                "text_field": args.text_field,
                "text": pairs_by_id[samples[i]["image_id"]][args.text_field],
                "epoch20_rank_i2t": int(rankA_i2t[pos]),
                "epoch20_rank_t2i": int(rankA_t2i[pos]),
                "epoch20_hit_i2t": bool(hitA_i2t[pos]),
                "epoch20_hit_t2i": bool(hitA_t2i[pos]),
                "dfn_rank_i2t": int(rankB_i2t[pos]),
                "dfn_rank_t2i": int(rankB_t2i[pos]),
                "dfn_hit_i2t": bool(hitB_i2t[pos]),
                "dfn_hit_t2i": bool(hitB_t2i[pos]),
                "self_sim_epoch20": float(dA[i]),
                "self_sim_dfn": float(dB[i]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"text_field: {args.text_field}\n")
        f.write(f"full pool: {len(samples)} samples | epoch20 i2t_r1={r1_a_full[0]:.4f} t2i_r1={r1_a_full[1]:.4f} | "
                f"DFN i2t_r1={r1_b_full[0]:.4f} t2i_r1={r1_b_full[1]:.4f}\n")
        f.write(f"chosen pool: {size} samples\n")
        f.write(f"  epoch20: i2t_r1={r1_a_i:.4f} ({hitA_i2t.sum()}/{size})  t2i_r1={r1_a_t:.4f} ({hitA_t2i.sum()}/{size})\n")
        f.write(f"  DFN:     i2t_r1={r1_b_i:.4f} ({hitB_i2t.sum()}/{size})  t2i_r1={r1_b_t:.4f} ({hitB_t2i.sum()}/{size})\n")
        f.write("pool image_ids (in selection order):\n")
        f.write("\n".join(samples[i]["image_id"] for i in pool) + "\n")

    _LOGGER.info("Wrote pool to %s", output_dir)
    _LOGGER.info(
        "Final | pool=%d  epoch20 i2t_r1=%.4f t2i_r1=%.4f | DFN i2t_r1=%.4f t2i_r1=%.4f",
        size, r1_a_i, r1_a_t, r1_b_i, r1_b_t,
    )


if __name__ == "__main__":
    main()
