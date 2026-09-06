"""Run every record of a QAD dataset through the real collator before training.

A single unusable record is fatal: the collator raises inside the dataloader, the
rank dies, and torchrun tears down the whole job.  The RedPajama run lost four
hours at step 261 that way, on record ~67k of 558k, because the earlier check had
only sampled the first 1024 records.  Sampling cannot clear a corpus built from
arbitrary web text -- the failure modes live in the tail by construction.

Reports every failure with enough context to classify it, and writes the offending
ids so the dataset can be filtered without rebuilding.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def check_chunk(job: dict[str, Any]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    from faquant.qad_data import AssistantOnlyDataCollator

    tokenizer = AutoTokenizer.from_pretrained(job["tokenizer"])
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_length=job["max_length"],
        max_answer_tokens=job["max_answer_tokens"],
    )
    failures = []
    for line_number, line in job["lines"]:
        record = json.loads(line)
        try:
            # One record per batch so a failure names the record, and so padding
            # never hides a length problem behind a longer neighbour.
            collator([record])
        except Exception as error:  # noqa: BLE001 - reporting, not handling
            answer = record["messages"][-1]["content"]
            failures.append(
                {
                    "line": line_number,
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "error": type(error).__name__,
                    "message": str(error),
                    "answer_head": answer[:120],
                    "answer_repr_head": repr(answer[:60]),
                    "answer_chars": len(answer),
                }
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument(
        "--report", type=Path, help="write the failure list here as JSON"
    )
    args = parser.parse_args()

    report: dict[str, Any] = {}
    total_failures = 0
    for path in args.paths:
        # Iterate the handle rather than splitlines(), which also breaks on
        # \u2028, \u2029 and \x85.  json.dumps(ensure_ascii=False) leaves those
        # three unescaped, so splitlines() cuts records containing them in half
        # and reports a JSON error for data the trainer reads without complaint.
        with path.open(encoding="utf-8") as handle:
            lines = [
                (number, line)
                for number, line in enumerate(handle)
                if line.strip()
            ]
        jobs = [
            {
                "lines": lines[start : start + args.chunk_size],
                "tokenizer": args.tokenizer,
                "max_length": args.max_length,
                "max_answer_tokens": args.max_answer_tokens,
            }
            for start in range(0, len(lines), args.chunk_size)
        ]
        log(f"{path}: {len(lines)} records in {len(jobs)} chunks")
        with mp.get_context("spawn").Pool(args.workers) as pool:
            batches = pool.map(check_chunk, jobs)
        failures = [item for batch in batches for item in batch]
        total_failures += len(failures)
        rate = len(failures) / max(len(lines), 1)
        log(f"{path}: {len(failures)} failures ({rate:.6%})")
        if failures:
            log("  by error type: " + str(Counter(f["error"] for f in failures)))
            for item in failures[:10]:
                log(f"  line {item['line']} {item['id']}: {item['message']}")
                log(f"    answer starts {item['answer_repr_head']}")
        report[str(path)] = {
            "records": len(lines),
            "failures": failures,
            "failure_ids": sorted({f["id"] for f in failures if f["id"]}),
        }

    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        log(f"report written to {args.report}")
    raise SystemExit(1 if total_failures else 0)


if __name__ == "__main__":
    main()
