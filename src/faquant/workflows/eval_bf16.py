"""Evaluate a BF16 Qwen3 model with MMLU or bundled LongBench tasks."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faquant.qwen3 import DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--middle-truncation", action="store_true")
    parser.add_argument("--sample-start", type=int)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM as BaseHFLM
    from lm_eval.tasks import TaskManager

    class HFLM(BaseHFLM):
        def tok_batch_encode(
            self,
            strings: list[str],
            padding_side: str = "left",
            left_truncate_len: int | None = None,
            truncation: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not args.middle_truncation or left_truncate_len is None:
                return super().tok_batch_encode(
                    strings,
                    padding_side=padding_side,
                    left_truncate_len=left_truncate_len,
                    truncation=truncation,
                )
            input_ids, attention_mask = super().tok_batch_encode(
                strings,
                padding_side=padding_side,
                left_truncate_len=None,
                truncation=truncation,
            )
            if input_ids.shape[-1] <= left_truncate_len:
                return input_ids, attention_mask
            if input_ids.shape[0] != 1:
                raise ValueError("middle truncation requires batch size 1")
            prefix_len = left_truncate_len // 2
            suffix_len = left_truncate_len - prefix_len
            return (
                torch.cat(
                    (input_ids[:, :prefix_len], input_ids[:, -suffix_len:]),
                    dim=-1,
                ),
                torch.cat(
                    (
                        attention_mask[:, :prefix_len],
                        attention_mask[:, -suffix_len:],
                    ),
                    dim=-1,
                ),
            )

    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    task_manager = TaskManager(
        include_path=str(files("faquant.tasks.longbench"))
    )
    eval_kwargs = {}
    if args.sample_start is not None or args.sample_end is not None:
        if (
            args.sample_start is None
            or args.sample_end is None
            or len(task_names) != 1
        ):
            raise ValueError("sample range requires one task and both endpoints")
        eval_kwargs["samples"] = {
            task_names[0]: list(range(args.sample_start, args.sample_end))
        }
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        enable_thinking=False,
    )
    result = evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        log_samples=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        apply_chat_template=args.apply_chat_template,
        task_manager=task_manager,
        metadata={
            "faquant_public": {
                "precision": "bfloat16",
                "middle_truncation": args.middle_truncation,
            }
        },
        **eval_kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
