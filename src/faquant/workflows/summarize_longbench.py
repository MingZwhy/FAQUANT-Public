"""Audit and summarize official LongBench shards without publishing samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


TASKS = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "multifieldqa_zh": 200,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "dureader": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "vcsum": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "lsht": 200,
    "passage_count": 200,
    "passage_retrieval_en": 200,
    "passage_retrieval_zh": 200,
    "lcc": 500,
    "repobench-p": 500,
}

CATEGORIES = {
    "single_document_qa": (
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
    ),
    "multi_document_qa": ("hotpotqa", "2wikimqa", "musique", "dureader"),
    "summarization": ("gov_report", "qmsum", "multi_news", "vcsum"),
    "few_shot_learning": ("trec", "triviaqa", "samsum", "lsht"),
    "synthetic": (
        "passage_count",
        "passage_retrieval_en",
        "passage_retrieval_zh",
    ),
    "code": ("lcc", "repobench-p"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_score(payload: dict, task_key: str) -> float:
    result = payload["results"][task_key]
    for metric in ("score,none", "f1,none", "rougeL,none", "acc,none"):
        if metric in result:
            return float(result[metric]) * 100.0
    raise KeyError(f"{task_key}: no supported score in {sorted(result)}")


def render_markdown(summary: dict) -> str:
    lines = [
        "# Official LongBench summary",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Samples: {summary['sample_count']}",
        f"- BF16 macro: **{summary['bf16_macro']:.4f}**",
        f"- Candidate macro: **{summary['candidate_macro']:.4f}**",
        f"- Loss: **{summary['loss_pp']:.4f} pp**",
        "",
        "| Task | BF16 | Candidate | Loss |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task']} | {row['bf16']:.2f} | "
            f"{row['candidate']:.2f} | {row['loss_pp']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = []
    hash_mismatches = 0
    checkpoint_id = None
    for task, expected_count in TASKS.items():
        task_key = f"faquant_longbench_{task}"
        candidate_samples = []
        weighted_score = 0.0
        observed_count = 0
        for index in range(args.num_shards):
            path = (
                args.result_dir
                / "shards"
                / f"{task_key}_shard_{index}_of_{args.num_shards}.json"
            )
            payload = load_json(path)
            samples = payload["samples"][task_key]
            runtime = payload["faquant_qad_runtime"]
            if runtime.get("quantization_scope") != "prefill_only":
                raise ValueError(f"{path}: not a prefill-only result")
            if runtime.get("decode_reuses_quantized_prefill_kv_cache") is not True:
                raise ValueError(f"{path}: BF16 decode did not reuse quantized cache")
            checkpoint_id = checkpoint_id or Path(runtime["checkpoint"]).name
            weighted_score += result_score(payload, task_key) * len(samples)
            observed_count += len(samples)
            candidate_samples.extend(samples)
        if observed_count != expected_count:
            raise ValueError(f"{task}: expected {expected_count}, got {observed_count}")

        by_id = {int(sample["doc_id"]): sample for sample in candidate_samples}
        if sorted(by_id) != list(range(expected_count)):
            raise ValueError(f"{task}: missing or duplicate document IDs")
        bf16 = load_json(args.bf16_dir / f"{task_key}_full_bf16_native.json")
        bf16_by_id = {
            int(sample["doc_id"]): sample for sample in bf16["samples"][task_key]
        }
        for doc_id, sample in by_id.items():
            reference = bf16_by_id[doc_id]
            for key in ("doc_hash", "prompt_hash", "target_hash"):
                hash_mismatches += sample.get(key) != reference.get(key)
        candidate_score = weighted_score / observed_count
        bf16_score = result_score(bf16, task_key)
        rows.append(
            {
                "task": task,
                "bf16": bf16_score,
                "candidate": candidate_score,
                "loss_pp": bf16_score - candidate_score,
            }
        )

    if hash_mismatches:
        raise ValueError(f"{hash_mismatches} sample hashes differ from BF16")
    by_name = {row["task"]: row for row in rows}
    categories = {}
    for category, tasks in CATEGORIES.items():
        bf16_score = fmean(by_name[task]["bf16"] for task in tasks)
        candidate_score = fmean(by_name[task]["candidate"] for task in tasks)
        categories[category] = {
            "bf16": bf16_score,
            "candidate": candidate_score,
            "loss_pp": bf16_score - candidate_score,
        }
    bf16_macro = fmean(row["bf16"] for row in rows)
    candidate_macro = fmean(row["candidate"] for row in rows)
    summary = {
        "benchmark": "THUDM LongBench v1 main",
        "checkpoint_id": checkpoint_id,
        "task_count": len(rows),
        "sample_count": sum(TASKS.values()),
        "quantization_scope": "prefill_only",
        "decode_precision": "bfloat16",
        "decode_reuses_quantized_prefill_kv_cache": True,
        "bf16_macro": bf16_macro,
        "candidate_macro": candidate_macro,
        "loss_pp": bf16_macro - candidate_macro,
        "hash_mismatches": 0,
        "categories": categories,
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
