"""Build the RedPajama continuation dataset used for QAD.

Each document is converted into a long unsupervised prefix followed by a short
supervised continuation. All QAD losses are masked to continuation tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from faquant.qad_windowing import MIN_TARGET_TOKENS, window

DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"
SOURCE_REPO = "liang2kl/RedPajama-Data-1T-Sample-Backup"
SHARD_COUNT = 11

# togethercomputer/RedPajama-Data-1T-Sample is the canonical 1B-token sample but
# now 404s; this is a parquet mirror of it, which also sidesteps the deprecated
# dataset loading script that the full 1T repo still ships.


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def classify(meta: str) -> str:
    """Recover the RedPajama subset, which the sample exposes only via meta.

    Quoting is not consistent across subsets -- CommonCrawl rows are JSON with
    double quotes, GitHub rows are Python reprs with single quotes -- so match
    with the quotes stripped rather than against either style.
    """
    flat = meta.replace("'", "").replace('"', "")
    if "arxiv_id" in flat:
        return "arxiv"
    # wiki_prob is a CommonCrawl quality-classifier field, not a Wikipedia mark.
    if "wiki_prob" in flat or "source: cc/" in flat:
        return "common_crawl"
    if "source: c4" in flat:
        return "c4"
    if "source: github" in flat or "content_hash" in flat:
        return "github"
    if "source: stackexchange" in flat or "question_score" in flat:
        return "stackexchange"
    if "wikipedia.org" in flat:
        return "wikipedia"
    if "short_book_title" in flat:
        return "book"
    return "other"


def process_shard(job: dict[str, Any]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    index = job["index"]
    tokenizer = AutoTokenizer.from_pretrained(job["tokenizer"])
    path = hf_hub_download(
        SOURCE_REPO,
        f"data/train-{index:05d}-of-{SHARD_COUNT:05d}.parquet",
        repo_type="dataset",
    )
    rng = random.Random(job["seed"] + index)
    quota = job["quota"]
    # Reservoir rather than "first quota rows": the shards are not homogeneous
    # and their row order is arbitrary, so truncating in order lets that order
    # decide the mixture. Shard 10 interleaves Wikipedia, StackExchange and
    # GitHub, and taking a prefix admitted 5,477 of its 29,834 Wikipedia
    # documents for no reason other than where the cut happened to land.
    records: list[dict[str, Any]] = []
    kept = 0
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        rows = batch.to_pylist()
        for row in rows:
            seen += 1
            text = row.get("text") or ""
            cut = window(
                text,
                tokenizer,
                rng,
                max_tokens=job["max_tokens"],
                target_tokens=job["target_tokens"],
                target_fraction=job["target_fraction"],
                min_tokens=job["min_tokens"],
                char_budget=job["char_budget"],
            )
            if cut is None:
                continue
            head, tail = cut
            record = {
                "id": f"redpajama-{index:02d}-{kept:07d}",
                "source": f"redpajama/{classify(str(row.get('meta') or ''))}",
                "messages": [
                    {"role": "user", "content": head},
                    {"role": "assistant", "content": tail},
                ],
            }
            kept += 1
            if len(records) < quota:
                records.append(record)
            else:
                slot = rng.randrange(kept)
                if slot < quota:
                    records[slot] = record
    log(f"  shard {index}: sampled {len(records)} from {kept} usable of {seen}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/qad_redpajama"))
    parser.add_argument("--target-records", type=int, default=560_000)
    parser.add_argument("--validation-records", type=int, default=2_000)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--min-tokens", type=int, default=160)
    parser.add_argument("--char-budget", type=int, default=2_400)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if args.max_tokens > 1024:
        raise ValueError("max-tokens must fit the collator's --max-length of 1024")
    if args.min_tokens <= MIN_TARGET_TOKENS:
        raise ValueError(
            f"min-tokens must exceed the {MIN_TARGET_TOKENS}-token target floor "
            "so every window keeps some unsupervised context"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shards are homogeneous per subset, so an equal quota each preserves the
    # sample's own mixture only if the shards are equally sized -- they are, at
    # 84,593 rows apiece.
    per_shard = -(-args.target_records // SHARD_COUNT)
    jobs = [
        {
            "index": index,
            "quota": per_shard,
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "target_tokens": args.target_tokens,
            "target_fraction": args.target_fraction,
            "min_tokens": args.min_tokens,
            "char_budget": args.char_budget,
        }
        for index in range(SHARD_COUNT)
    ]
    log(f"building {args.target_records} records, {per_shard} per shard")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        batches = pool.map(process_shard, jobs)

    records = [record for batch in batches for record in batch]
    random.Random(args.seed).shuffle(records)
    records = records[: args.target_records]
    validation = records[: args.validation_records]
    train = records[args.validation_records :]

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    train_sha = write_jsonl(train_path, train)
    validation_sha = write_jsonl(validation_path, validation)

    manifest = {
        "source_repo": SOURCE_REPO,
        "candidate_count": args.target_records,
        "window": {
            "max_tokens": args.max_tokens,
            "target_tokens": args.target_tokens,
            "target_fraction": args.target_fraction,
            "min_tokens": args.min_tokens,
            "char_budget": args.char_budget,
            "offset": "random per document",
        },
        "notes": {
            "geometry": (
                "Each document becomes a user/assistant pair so the unsupervised "
                "prefix and supervised continuation are explicit."
            ),
            "chat_template": (
                "Chat markers separate context from the supervised continuation."
            ),
            "mixture": (
                "Each shard contributes a uniform reservoir sample, so the "
                "mixture is the 1B-token sample's own composition by document "
                "count and is not re-weighted to the published 1T token ratios. "
                "arXiv in particular is capped by availability: the sample holds "
                "only 1,510 arXiv documents in total, so its share cannot reach "
                "the 1T ratio without oversampling."
            ),
            "sample_mirror": (
                "togethercomputer/RedPajama-Data-1T-Sample now 404s; this parquet "
                "mirror also avoids the deprecated loading script of the 1T repo."
            ),
        },
        "train_count": len(train),
        "validation_count": len(validation),
        "train_by_source": dict(Counter(item["source"] for item in train)),
        "files": {
            train_path.name: {"sha256": train_sha, "records": len(train)},
            validation_path.name: {
                "sha256": validation_sha,
                "records": len(validation),
            },
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
