import json

import pytest
import torch

from faquant.qad_data import AssistantOnlyDataCollator, QADJsonlDataset


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize and not enable_thinking
        ids = [1]
        for message in messages:
            role = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            ids.extend([role, *[40 + index for index, _ in enumerate(message["content"])]])
        if add_generation_prompt:
            ids.append(30)
        elif messages[-1]["role"] == "assistant":
            ids.append(self.eos_token_id)
        return ids


def _record(answer: str = "ok"):
    return {
        "id": "example",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_qad_jsonl_dataset_validates_assistant_tail(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    dataset = QADJsonlDataset(path)
    assert len(dataset) == 1
    assert dataset[0]["id"] == "example"

    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "bad"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="assistant"):
        QADJsonlDataset(path)


def test_assistant_collator_masks_prompt_and_preserves_answer() -> None:
    collator = AssistantOnlyDataCollator(
        StubTokenizer(),
        max_length=32,
        max_answer_tokens=8,
        prompt_head_tokens=4,
    )
    batch = collator([_record("abc"), _record("x")])
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert torch.all(
        batch["labels"][batch["labels"] != -100]
        == batch["input_ids"][batch["labels"] != -100]
    )
    assert torch.all((batch["labels"] != -100).sum(dim=-1) > 0)


def test_assistant_collator_truncates_prompt_before_answer() -> None:
    collator = AssistantOnlyDataCollator(
        StubTokenizer(),
        max_length=16,
        max_answer_tokens=6,
        prompt_head_tokens=3,
    )
    batch = collator([_record("long-answer")])
    assert batch["input_ids"].shape == (1, 16)
    answer = batch["labels"][0][batch["labels"][0] != -100]
    assert answer[-1].item() == StubTokenizer.eos_token_id
    assert len(answer) == 6


def test_assistant_collator_marks_only_gold_gkd_records_for_task_loss() -> None:
    collator = AssistantOnlyDataCollator(
        StubTokenizer(), max_length=32, max_answer_tokens=8,
    )
    student = _record("student") | {"gkd_mixture": {"policy": "student"}}
    gold = _record("gold") | {"gkd_mixture": {"policy": "gold"}}
    batch = collator([student, gold])
    torch.testing.assert_close(
        batch["task_weights"], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(
        batch["kl_reverse_weights"], torch.tensor([1.0, 0.0])
    )


def test_assistant_collator_rejects_partial_gkd_tagging() -> None:
    collator = AssistantOnlyDataCollator(
        StubTokenizer(), max_length=32, max_answer_tokens=8,
    )
    tagged = _record("student") | {"gkd_mixture": {"policy": "student"}}
    with pytest.raises(ValueError, match="tagged and untagged"):
        collator([tagged, _record("gold")])
