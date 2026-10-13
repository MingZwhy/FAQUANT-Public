from types import SimpleNamespace

import torch

from faquant.workflows.eval_longbench import (
    _prefill_quant_decode_bf16_generate,
)


class FakeCache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length


class FakePrefill:
    def __init__(self, chunk_tokens: int | None = None) -> None:
        self.generation_config = SimpleNamespace(eos_token_id=4)
        self.__dict__["_faquant_bf16_decode_model"] = FakeDecode()
        self.__dict__["_faquant_prefill_chunk_tokens"] = chunk_tokens
        attention = SimpleNamespace(
            faquant_k_matmul_qdq_cache=object(), faquant_v_matmul_qdq_cache=object()
        )
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
        self.chunks = []

    def __call__(
        self,
        *,
        input_ids,
        past_key_values,
        cache_position,
        use_cache,
        return_dict,
        logits_to_keep,
    ):
        assert use_cache and return_dict and logits_to_keep == 1
        cache = past_key_values or FakeCache()
        assert cache_position.tolist() == list(
            range(cache.length, cache.length + input_ids.shape[1])
        )
        self.chunks.append(input_ids.tolist())
        cache.length += input_ids.shape[1]
        logits = torch.zeros(input_ids.shape[0], 1, 5)
        logits[:, -1, 3] = 1
        return SimpleNamespace(logits=logits, past_key_values=cache)


class FakeDecode:
    def __init__(self) -> None:
        self.positions = []

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
        assert use_cache and return_dict
        self.positions.append(cache_position.tolist())
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
    assert model.chunks == [[[1, 2]]]
    assert model.__dict__["_faquant_bf16_decode_model"].positions == [[2]]
    attention = model.model.layers[0].self_attn
    assert attention.faquant_k_matmul_qdq_cache is None
    assert attention.faquant_v_matmul_qdq_cache is None


def test_chunked_prefill_feeds_the_prompt_in_order() -> None:
    model = FakePrefill(chunk_tokens=2)
    output = _prefill_quant_decode_bf16_generate(
        model,
        input_ids=torch.tensor([[1, 2, 5, 6, 7]]),
        max_length=8,
        do_sample=False,
        num_beams=1,
    )
    assert output.tolist() == [[1, 2, 5, 6, 7, 3, 4]]
    assert model.chunks == [[[1, 2]], [[5, 6]], [[7]]]
    assert model.__dict__["_faquant_bf16_decode_model"].positions == [[5]]
