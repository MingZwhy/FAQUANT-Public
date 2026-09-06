from types import SimpleNamespace

import torch

from faquant.workflows.eval_longbench import (
    _prefill_quant_decode_bf16_generate,
)


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class FakePrefill:
    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(eos_token_id=4)
        self.__dict__["_faquant_bf16_decode_model"] = FakeDecode()

    def __call__(self, *, input_ids, use_cache, return_dict):
        assert use_cache and return_dict
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
        logits[:, -1, 3] = 1
        return SimpleNamespace(logits=logits, past_key_values=FakeCache(2))


class FakeDecode:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        *,
        input_ids,
        past_key_values,
        cache_position,
        use_cache,
        return_dict,
    ):
        assert input_ids.tolist() == [[3]]
        assert cache_position.tolist() == [2]
        assert use_cache and return_dict
        self.calls += 1
        past_key_values.length += 1
        logits = torch.zeros(1, 1, 5)
        logits[:, -1, 4] = 1
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_quantized_prefill_selects_first_token_then_bf16_decodes() -> None:
    model = FakePrefill()
    output = _prefill_quant_decode_bf16_generate(
        model,
        input_ids=torch.tensor([[1, 2]]),
        max_length=8,
        do_sample=False,
        num_beams=1,
    )
    assert output.tolist() == [[1, 2, 3, 4]]
    assert model.__dict__["_faquant_bf16_decode_model"].calls == 1
