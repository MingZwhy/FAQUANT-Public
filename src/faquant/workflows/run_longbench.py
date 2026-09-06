#!/usr/bin/env python3
"""Multi-GPU LongBench with quantized prefill and BF16 decode.

Uses the canonical QAD loader for quantized prefill, then hands its KV cache to
a basis-compatible BF16 model for autoregressive decode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from faquant.qwen3 import DEFAULT_MODEL

TASKS = (
    ("gov_report", 200, 10),
    ("vcsum", 200, 10),
    ("multi_news", 200, 10),
    ("qmsum", 200, 8),
    ("narrativeqa", 200, 4),
    ("dureader", 200, 3),
    ("qasper", 200, 3),
    ("lcc", 500, 3),
    ("repobench-p", 500, 3),
    ("musique", 200, 2),
    ("hotpotqa", 200, 2),
    ("2wikimqa", 200, 2),
    ("multifieldqa_zh", 200, 2),
    ("multifieldqa_en", 150, 2),
    ("samsum", 200, 1),
    ("triviaqa", 200, 1),
    ("trec", 200, 1),
    ("lsht", 200, 1),
    ("passage_count", 200, 1),
    ("passage_retrieval_en", 200, 1),
    ("passage_retrieval_zh", 200, 1),
)
RAW_PROMPT_TASKS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}
NUM_SHARDS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--expected-exempt-layers",
        default="16,17,18",
        help="Comma-separated materialized BF16-protected layers.",
    )
    parser.add_argument(
        "--qk-mxfp8-layers",
        default="",
        help="Optional comma-separated MXFP8 QK layers; empty uses HiF4 QK.",
    )
    parser.add_argument(
        "--truncation-mode",
        choices=("left", "middle"),
        default="left",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated short LongBench names; empty selects all 21.",
    )
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    return parser.parse_args()


def layer_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def shard_range(count: int, index: int, num_shards: int) -> tuple[int, int]:
    return count * index // num_shards, count * (index + 1) // num_shards


def shard_done(
    path: Path,
    task_key: str,
    start: int,
    end: int,
    *,
    expected_exempt_layers: list[int],
    qk_mxfp8_layers: list[int],
    truncation_mode: str,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload["samples"][task_key]
        doc_ids = sorted(int(sample["doc_id"]) for sample in samples)
        runtime = payload.get("faquant_qad_runtime") or {}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        doc_ids == list(range(start, end))
        and runtime.get("quant_exempt_layers") == expected_exempt_layers
        and runtime.get("qk_mxfp8_layers") == qk_mxfp8_layers
        and runtime.get("truncation_mode", "left") == truncation_mode
        and runtime.get("quantization_scope") == "prefill_only"
        and runtime.get("decode_precision") == "bfloat16"
        and runtime.get("decode_reuses_quantized_prefill_kv_cache") is True
        and runtime.get("qk_matmul_quant") is True
        and runtime.get("pv_matmul_quant") is True
    )


def jobs(
    selected_tasks: frozenset[str],
    num_shards: int,
) -> list[tuple[str, int, int, int, int]]:
    out: list[tuple[str, int, int, int, int]] = []
    for name, count, cost in TASKS:
        if name not in selected_tasks:
            continue
        for index in range(num_shards):
            start, end = shard_range(count, index, num_shards)
            out.append((name, index, start, end, cost))
    out.sort(key=lambda item: (-item[4], item[0], item[1]))
    return out


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    expected_exempt_layers = layer_list(args.expected_exempt_layers)
    qk_mxfp8_layers = layer_list(args.qk_mxfp8_layers)
    all_task_names = frozenset(name for name, _count, _cost in TASKS)
    selected_tasks = (
        frozenset(item.strip() for item in args.tasks.split(",") if item.strip())
        if args.tasks
        else all_task_names
    )
    unknown_tasks = selected_tasks - all_task_names
    if unknown_tasks:
        raise ValueError(f"unknown LongBench tasks: {sorted(unknown_tasks)}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    receipt = json.loads(
        (args.checkpoint / "qad_init.json").read_text(encoding="utf-8")
    )
    if receipt.get("quant_exempt_layers") != expected_exempt_layers:
        raise ValueError(
            f"checkpoint exemptions {receipt.get('quant_exempt_layers')} do not "
            f"match --expected-exempt-layers {expected_exempt_layers}"
        )
    result_dir = args.result_dir
    shard_dir = result_dir / "shards"
    log_dir = result_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env_base["LM_EVAL_TRUNCATION_MODE"] = args.truncation_mode

    planned = jobs(selected_tasks, args.num_shards)
    pending = []
    skipped = 0
    for name, index, start, end, cost in planned:
        task_key = f"faquant_longbench_{name}"
        output = (
            shard_dir
            / f"{task_key}_shard_{index}_of_{args.num_shards}.json"
        )
        if shard_done(
            output,
            task_key,
            start,
            end,
            expected_exempt_layers=expected_exempt_layers,
            qk_mxfp8_layers=qk_mxfp8_layers,
            truncation_mode=args.truncation_mode,
        ):
            skipped += 1
            continue
        pending.append((name, index, start, end, cost, output))

    running: dict[int, subprocess.Popen[str]] = {}
    assigned: dict[int, str] = {}
    failed: list[str] = []
    started_at = time.time()
    queue = list(pending)

    def snapshot(extra: str = "") -> None:
        write_status(
            result_dir / "STATUS.json",
            {
                "checkpoint": str(args.checkpoint),
                "expected_exempt_layers": expected_exempt_layers,
                "qk_mxfp8_layers": qk_mxfp8_layers,
                "truncation_mode": args.truncation_mode,
                "quantization_scope": "prefill_only",
                "decode_precision": "bfloat16",
                "decode_reuses_quantized_prefill_kv_cache": True,
                "tasks": sorted(selected_tasks),
                "num_shards": args.num_shards,
                "pending": len(queue),
                "running": assigned,
                "skipped_done": skipped,
                "failed": failed,
                "elapsed_seconds": round(time.time() - started_at, 1),
                "note": extra,
            },
        )

    snapshot("supervisor started")
    while queue or running:
        finished = [gpu for gpu, proc in running.items() if proc.poll() is not None]
        for gpu in finished:
            proc = running.pop(gpu)
            tag = assigned.pop(gpu)
            if proc.returncode != 0:
                failed.append(f"{tag}:exit{proc.returncode}")
        while queue and len(running) < len(gpus):
            free = next(gpu for gpu in gpus if gpu not in running)
            name, index, start, end, _cost, output = queue.pop(0)
            task_key = f"faquant_longbench_{name}"
            tag = f"{task_key}_shard_{index}"
            log_path = log_dir / f"{tag}.log"
            cmd = [
                args.python,
                "-m",
                "faquant.workflows.eval_longbench",
                "--model",
                args.model,
                "--checkpoint",
                str(args.checkpoint),
                "--rotation",
                "hisq1024",
                "--tasks",
                task_key,
                "--qk-matmul-quant",
                "--pv-matmul-quant",
                "--post-rope-qk-rotation",
                "--pv-normalizer-mode",
                "quantized_same",
                "--qk-smooth-scales",
                str(args.scales),
                "--attention-kernel",
                "simulated",
                "--device",
                "cuda:0",
                "--batch-size",
                args.batch_size,
                "--max-length",
                "40960",
                "--skip-weight-hash",
                "--sample-start",
                str(start),
                "--sample-end",
                str(end),
                "--output",
                str(output),
            ]
            if name not in RAW_PROMPT_TASKS:
                cmd.append("--apply-chat-template")
            if qk_mxfp8_layers:
                cmd.extend(
                    [
                        "--qk-mxfp8-layers",
                        ",".join(str(layer) for layer in qk_mxfp8_layers),
                    ]
                )
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(free)
            handle = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_path.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
            )
            running[free] = handle
            assigned[free] = tag
        snapshot()
        time.sleep(20)

    snapshot("supervisor finished")
    if failed:
        raise SystemExit(f"failed shards: {failed}")
    print("LONGBENCH_SHARDS_DONE")


if __name__ == "__main__":
    main()
