from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class QADJsonlDataset(Dataset[dict[str, Any]]):
    """Small deterministic JSONL dataset of chat-formatted QAD examples."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages")
                if (
                    not isinstance(messages, list)
                    or not messages
                    or messages[-1].get("role") != "assistant"
                    or not str(messages[-1].get("content", "")).strip()
                ):
                    raise ValueError(
                        f"{self.path}:{line_number} must end in a non-empty "
                        "assistant message"
                    )
                self.records.append(record)
        if not self.records:
            raise ValueError(f"{self.path} contains no QAD examples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class AssistantOnlyDataCollator:
    """Apply the Qwen chat template and label only assistant answer tokens."""

    def __init__(
        self,
        tokenizer: object,
        *,
        max_length: int = 1024,
        max_answer_tokens: int = 256,
        prompt_head_tokens: int = 128,
    ) -> None:
        if max_length <= 0 or max_answer_tokens <= 0 or prompt_head_tokens < 0:
            raise ValueError("invalid QAD collator length configuration")
        if max_answer_tokens >= max_length:
            raise ValueError("max_answer_tokens must be smaller than max_length")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_tokens = max_answer_tokens
        self.prompt_head_tokens = prompt_head_tokens
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)

    def _tokenize(self, record: dict[str, Any]) -> tuple[list[int], list[int]]:
        messages = record["messages"]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "chat template does not expose an assistant-prefix boundary"
            )
        answer_ids = full_ids[len(prompt_ids) :]
        if not answer_ids:
            raise ValueError(f"QAD example {record.get('id')} has no answer tokens")
        if len(answer_ids) > self.max_answer_tokens:
            answer_ids = answer_ids[: self.max_answer_tokens - 1] + [
                self.eos_token_id
            ]

        prompt_budget = self.max_length - len(answer_ids)
        if len(prompt_ids) > prompt_budget:
            head = min(self.prompt_head_tokens, prompt_budget)
            tail = prompt_budget - head
            prompt_ids = (
                prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
            )
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        if not any(label != -100 for label in labels):
            raise RuntimeError("assistant-only truncation removed every target token")
        return input_ids, labels

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tokenized = [self._tokenize(record) for record in records]
        length = max(len(input_ids) for input_ids, _ in tokenized)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for input_ids, labels in tokenized:
            padding = length - len(input_ids)
            input_rows.append(input_ids + [self.pad_token_id] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(input_ids) + [0] * padding)
        batch = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }
        mixture = [record.get("gkd_mixture") for record in records]
        if any(value is not None for value in mixture):
            if not all(isinstance(value, dict) for value in mixture):
                raise ValueError("a batch cannot mix tagged and untagged GKD records")
            policies = [value.get("policy") for value in mixture]
            if any(policy not in ("student", "gold") for policy in policies):
                raise ValueError("GKD mixture policy must be 'student' or 'gold'")
            batch["task_weights"] = torch.tensor(
                [1.0 if policy == "gold" else 0.0 for policy in policies],
                dtype=torch.float32,
            )
            batch["kl_reverse_weights"] = torch.tensor(
                [1.0 if policy == "student" else 0.0 for policy in policies],
                dtype=torch.float32,
            )
        return batch
