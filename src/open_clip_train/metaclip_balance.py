"""MetaCLIP-style metadata counting and online balancing utilities."""

import argparse
import json
import random
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from torch.utils.data import get_worker_info

from open_clip_train.data import expand_urls


COUNTS_FORMAT_VERSION = 1


def _string_value(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _nested_json_value(metadata, dotted_key):
    value = metadata
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def extract_concepts(txt=None, json_value=None, source="txt"):
    """Extract one or more metadata entries from a WebDataset sample."""
    if source == "txt":
        concept = _string_value(txt)
        return [concept] if concept else []
    if not source.startswith("json:") or not source[5:]:
        raise ValueError("MetaCLIP concept source must be 'txt' or 'json:<dotted-key>'.")

    if isinstance(json_value, bytes):
        json_value = json_value.decode("utf-8", errors="replace")
    if isinstance(json_value, str):
        try:
            json_value = json.loads(json_value)
        except json.JSONDecodeError:
            return []
    value = _nested_json_value(json_value, source[5:])
    values = value if isinstance(value, (list, tuple, set)) else [value]
    # A sample matching the same entry more than once should still count once.
    return list(dict.fromkeys(concept for item in values if (concept := _string_value(item))))


def _member_sample_key(name):
    basename = Path(name).name
    if "." not in basename:
        return basename, ""
    return basename.rsplit(".", 1)


def _count_concepts_in_shard(shard, source):
    counts = Counter()
    matched_samples = 0
    total_samples = 0
    wanted_suffix = "txt" if source == "txt" else "json"

    with tarfile.open(shard, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            _, suffix = _member_sample_key(member.name)
            if suffix.lower() != wanted_suffix:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            value = stream.read()
            concepts = extract_concepts(
                txt=value if source == "txt" else None,
                json_value=value if source != "txt" else None,
                source=source,
            )
            total_samples += 1
            if concepts:
                matched_samples += 1
                counts.update(concepts)
    return counts, total_samples, matched_samples


def count_concepts(shards, source="txt", workers=1, progress=False):
    """Count metadata matches without decoding image payloads."""
    counts = Counter()
    total_samples = 0
    matched_samples = 0
    completed = 0

    if workers == 1:
        results = (_count_concepts_in_shard(shard, source) for shard in shards)
        for shard, result in zip(shards, results):
            shard_counts, shard_total, shard_matched = result
            counts.update(shard_counts)
            total_samples += shard_total
            matched_samples += shard_matched
            completed += 1
            if progress:
                print(
                    f"MetaCLIP counting [{completed}/{len(shards)}] {shard}: "
                    f"matched={matched_samples}/{total_samples}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_count_concepts_in_shard, shard, source): shard
                for shard in shards
            }
            for future in as_completed(futures):
                shard = futures[future]
                shard_counts, shard_total, shard_matched = future.result()
                counts.update(shard_counts)
                total_samples += shard_total
                matched_samples += shard_matched
                completed += 1
                if progress:
                    print(
                        f"MetaCLIP counting [{completed}/{len(shards)}] {shard}: "
                        f"matched={matched_samples}/{total_samples}",
                        flush=True,
                    )
    return counts, total_samples, matched_samples


def save_counts(path, counts, source, shards, total_samples, matched_samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": COUNTS_FORMAT_VERSION,
        "source": source,
        "num_shards": len(shards),
        "total_samples": total_samples,
        "matched_samples": matched_samples,
        "counts": dict(sorted(counts.items())),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_counts(path, expected_source=None):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != COUNTS_FORMAT_VERSION:
        raise ValueError(f"Unsupported MetaCLIP counts format in {path!s}.")
    if expected_source is not None and payload.get("source") != expected_source:
        raise ValueError(
            f"MetaCLIP counts source mismatch: file has {payload.get('source')!r}, "
            f"training requested {expected_source!r}."
        )
    counts = payload.get("counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError(f"MetaCLIP counts file contains no concepts: {path!s}")
    return {str(key): int(value) for key, value in counts.items()}, payload


class MetaClipBalanceFilter:
    """Independent sampler from MetaCLIP Algorithm 1.

    For every matched entry e, p(e)=min(1, threshold/count(e)). A pair is
    retained when at least one of its entries independently selects it.
    """

    def __init__(self, counts, threshold, source="txt", seed=0):
        if threshold <= 0:
            raise ValueError("MetaCLIP balance threshold must be positive.")
        self.source = source
        self.probabilities = {
            concept: min(1.0, float(threshold) / count)
            for concept, count in counts.items()
            if count > 0
        }
        self.seed = seed
        self._rng = None

    def _random(self):
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = worker.seed if worker is not None else 0
            self._rng = random.Random(self.seed + worker_seed)
        return self._rng.random()

    def __call__(self, sample):
        concepts = extract_concepts(sample.get("txt"), sample.get("json"), self.source)
        for concept in concepts:
            probability = self.probabilities.get(concept)
            if probability is not None and self._random() < probability:
                return True
        return False


def expected_single_concept_keep_rate(counts, threshold):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    kept = sum(min(count, threshold) for count in counts.values())
    return kept / total


def main():
    parser = argparse.ArgumentParser(description="Count metadata entries for MetaCLIP-style balancing.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="txt")
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}.")

    shards, _ = expand_urls(args.train_data)
    if not shards:
        raise ValueError(f"--train-data resolved to no shards: {args.train_data}")
    missing = [shard for shard in shards if not Path(shard).is_file()]
    if missing:
        raise FileNotFoundError(f"MetaCLIP counting requires local tar shards; missing: {missing[:3]}")

    print(
        f"MetaCLIP counting started: shards={len(shards)} source={args.source} workers={args.workers}",
        flush=True,
    )
    counts, total_samples, matched_samples = count_concepts(
        shards,
        args.source,
        workers=min(args.workers, len(shards)),
        progress=True,
    )
    if not counts:
        raise RuntimeError(f"No concepts found using source {args.source!r}.")
    save_counts(args.output, counts, args.source, shards, total_samples, matched_samples)
    print(
        f"MetaCLIP counts written to {args.output}: concepts={len(counts)} "
        f"matched_samples={matched_samples}/{total_samples}"
    )
    if args.threshold is not None:
        rate = expected_single_concept_keep_rate(counts, args.threshold)
        print(f"Estimated keep rate for single-concept samples at t={args.threshold}: {rate:.6f}")


if __name__ == "__main__":
    main()
