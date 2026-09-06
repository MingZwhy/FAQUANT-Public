from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from faquant.config import PV_NORMALIZER_MODES
from faquant.quantization import fake_quantize
from faquant.simulated_flash_attention import (
    _fake_quantize_tail,
    new_pv_normalizer_diagnostics,
    new_simulated_attention_stats,
    simulated_flash_attention_forward,
    summarize_pv_normalizer_diagnostics,
)


def _module(
    *,
    qk_quant: bool = False,
    pv_quant: bool = False,
    query_chunk: int = 3,
    key_chunk: int = 4,
    bits: int = 4,
    group_size: int = 4,
    pv_normalizer_mode: str = "unquantized",
) -> SimpleNamespace:
    return SimpleNamespace(
        is_causal=True,
        training=False,
        faquant_attention_query_chunk_size=query_chunk,
        faquant_attention_key_chunk_size=key_chunk,
        faquant_qk_matmul_quant=qk_quant,
        faquant_pv_matmul_quant=pv_quant,
        faquant_pv_normalizer_mode=pv_normalizer_mode,
        faquant_matmul_quant_kwargs={
            "bits": bits,
            "group_size": group_size,
            "symmetric": True,
            "clip_ratio": 1.0,
        },
        faquant_simulated_attention_stats=new_simulated_attention_stats(),
    )


def _repeat_kv(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    return tensor.repeat_interleave(repeats, dim=1)


def _dense_causal(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    key = _repeat_kv(key, repeats)
    value = _repeat_kv(value, repeats)
    scores = query.float() @ key.float().transpose(-1, -2)
    scores *= query.shape[-1] ** -0.5
    query_positions = torch.arange(query.shape[-2]) + key.shape[-2] - query.shape[-2]
    key_positions = torch.arange(key.shape[-2])
    allowed = key_positions[None, :] <= query_positions[:, None]
    scores.masked_fill_(~allowed.to(scores.device), -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities)
    return (probabilities @ value.float()).transpose(1, 2)


def _dense_padded(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros(
        query.shape[0],
        query.shape[-2],
        query.shape[1],
        query.shape[-1],
        dtype=torch.float32,
    )
    for batch in range(query.shape[0]):
        indices = mask[batch].nonzero().flatten()
        q = query[batch : batch + 1, :, indices]
        k = key[batch : batch + 1, :, indices]
        v = value[batch : batch + 1, :, indices]
        output[batch, indices] = _dense_causal(q, k, v)[0]
    return output


def test_streaming_matches_dense_causal_gqa_with_block_tails() -> None:
    torch.manual_seed(1)
    query = torch.randn(2, 4, 7, 8)
    key = torch.randn(2, 2, 11, 8)
    value = torch.randn(2, 2, 11, 8)
    actual, weights = simulated_flash_attention_forward(
        _module(query_chunk=3, key_chunk=4, group_size=8),
        query,
        key,
        value,
        None,
        scaling=8**-0.5,
    )
    expected = _dense_causal(query, key, value)
    assert weights is None
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_skipping_fully_future_tiles_is_bitwise_equal_to_reference_path() -> None:
    torch.manual_seed(19)
    query = torch.randn(1, 4, 11, 8)
    key = torch.randn(1, 2, 11, 8)
    value = torch.randn(1, 2, 11, 8)
    optimized_module = _module(
        qk_quant=True,
        pv_quant=True,
        query_chunk=3,
        key_chunk=2,
        group_size=4,
    )
    reference_module = _module(
        qk_quant=True,
        pv_quant=True,
        query_chunk=3,
        key_chunk=2,
        group_size=4,
    )
    reference_module.faquant_skip_fully_masked_tiles = False

    optimized, _ = simulated_flash_attention_forward(
        optimized_module, query, key, value, None
    )
    reference, _ = simulated_flash_attention_forward(
        reference_module, query, key, value, None
    )

    assert torch.equal(optimized, reference)
    assert (
        optimized_module.faquant_simulated_attention_stats["qk_tiles"]
        < reference_module.faquant_simulated_attention_stats["qk_tiles"]
    )
    assert (
        optimized_module.faquant_simulated_attention_stats["pv_tiles"]
        < reference_module.faquant_simulated_attention_stats["pv_tiles"]
    )

    explicit_mask = torch.ones(1, 11, dtype=torch.bool)
    optimized_mask_module = _module(
        qk_quant=True,
        pv_quant=True,
        query_chunk=3,
        key_chunk=2,
        group_size=4,
    )
    reference_mask_module = _module(
        qk_quant=True,
        pv_quant=True,
        query_chunk=3,
        key_chunk=2,
        group_size=4,
    )
    reference_mask_module.faquant_skip_fully_masked_tiles = False
    optimized_mask, _ = simulated_flash_attention_forward(
        optimized_mask_module, query, key, value, explicit_mask
    )
    reference_mask, _ = simulated_flash_attention_forward(
        reference_mask_module, query, key, value, explicit_mask
    )
    assert torch.equal(optimized_mask, reference_mask)


def test_bottom_right_causal_zeroes_queries_without_any_keys() -> None:
    torch.manual_seed(2)
    query = torch.randn(1, 2, 7, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=2, key_chunk=2),
        query,
        key,
        value,
        None,
        is_causal=None,
        softcap=0,
    )
    expected = _dense_causal(query, key, value)
    assert torch.count_nonzero(actual[:, :4]) == 0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_two_dimensional_padding_matches_unpadded_batches() -> None:
    torch.manual_seed(3)
    query = torch.randn(2, 4, 7, 4)
    key = torch.randn(2, 2, 7, 4)
    value = torch.randn(2, 2, 7, 4)
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=3, key_chunk=4), query, key, value, mask
    )
    expected = _dense_padded(query, key, value, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(actual[0, :2]) == 0
    assert torch.count_nonzero(actual[1, 4:]) == 0


def test_four_dimensional_additive_mask_has_zero_fully_masked_rows() -> None:
    torch.manual_seed(4)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.randn(1, 1, 5, 4)
    mask = torch.full((1, 1, 5, 5), torch.finfo(torch.float32).min)
    mask[..., 1:, 0] = 0
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=2, key_chunk=3), query, key, value, mask
    )
    assert torch.count_nonzero(actual[:, 0]) == 0
    assert torch.isfinite(actual).all()


def test_float16_additive_mask_keeps_fully_masked_rows_zero() -> None:
    torch.manual_seed(9)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    mask = torch.full((1, 1, 3, 3), torch.finfo(torch.float16).min, dtype=torch.float16)
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=2, key_chunk=2), query, key, value, mask
    )
    assert torch.count_nonzero(actual) == 0
    assert torch.isfinite(actual).all()


def test_decode_with_padding_and_sliding_window() -> None:
    torch.manual_seed(8)
    query = torch.randn(2, 4, 1, 4)
    key = torch.randn(2, 2, 7, 4)
    value = torch.randn(2, 2, 7, 4)
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=1, key_chunk=3),
        query,
        key,
        value,
        mask,
        sliding_window=3,
    )
    repeated_key = _repeat_kv(key, 2)
    repeated_value = _repeat_kv(value, 2)
    expected = torch.empty_like(actual, dtype=torch.float32)
    for batch in range(2):
        indices = mask[batch].nonzero().flatten()[-3:]
        scores = (
            query[batch].float()
            @ repeated_key[batch, :, indices].float().transpose(-1, -2)
        ) * 0.5
        probabilities = torch.softmax(scores, dim=-1)
        expected[batch, 0] = (
            probabilities @ repeated_value[batch, :, indices].float()
        )[:, 0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_decode_kv_qdq_cache_is_bitwise_equal_to_full_requantization() -> None:
    torch.manual_seed(18)
    query_heads = 4
    key = torch.randn(1, 2, 9, 8)
    value = torch.randn(1, 2, 9, 8)
    cached_module = _module(
        qk_quant=True,
        pv_quant=True,
        query_chunk=1,
        key_chunk=4,
        group_size=4,
    )

    for length in (6, 7, 8, 9):
        query = torch.randn(1, query_heads, 1, 8)
        actual, _ = simulated_flash_attention_forward(
            cached_module,
            query,
            key[..., :length, :],
            value[..., :length, :],
            None,
        )
        reference, _ = simulated_flash_attention_forward(
            _module(
                qk_quant=True,
                pv_quant=True,
                query_chunk=1,
                key_chunk=4,
                group_size=4,
            ),
            query,
            key[..., :length, :],
            value[..., :length, :],
            None,
        )
        assert torch.equal(actual, reference)
        assert cached_module.faquant_k_matmul_qdq_cache.shape[-2] == length
        assert cached_module.faquant_v_matmul_qdq_cache.shape[-2] == length


def test_noncausal_sliding_window_masks_both_sides() -> None:
    torch.manual_seed(14)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=2, key_chunk=3),
        query,
        key,
        value,
        None,
        is_causal=False,
        sliding_window=2,
    )
    scores = (query.float() @ key.float().transpose(-1, -2)) * 0.5
    positions = torch.arange(5)
    allowed = (positions[None, :] - positions[:, None]).abs() < 2
    scores.masked_fill_(~allowed, -torch.inf)
    expected = (torch.softmax(scores, dim=-1) @ value.float()).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_multi_token_decode_with_left_padding() -> None:
    torch.manual_seed(11)
    query = torch.randn(2, 4, 3, 4)
    key = torch.randn(2, 2, 7, 4)
    value = torch.randn(2, 2, 7, 4)
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=2, key_chunk=3), query, key, value, mask
    )
    repeated_key = _repeat_kv(key, 2)
    repeated_value = _repeat_kv(value, 2)
    expected = torch.empty_like(actual, dtype=torch.float32)
    for batch in range(2):
        indices = mask[batch].nonzero().flatten()
        scores = (
            query[batch].float()
            @ repeated_key[batch, :, indices].float().transpose(-1, -2)
        ) * 0.5
        query_positions = torch.arange(3) + indices.numel() - 3
        key_positions = torch.arange(indices.numel())
        scores.masked_fill_(
            key_positions[None, :] > query_positions[:, None], -torch.inf
        )
        probabilities = torch.softmax(scores, dim=-1)
        expected[batch] = (
            probabilities @ repeated_value[batch, :, indices].float()
        ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_packed_cu_seqlens_are_rejected_explicitly() -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    try:
        simulated_flash_attention_forward(
            _module(),
            query,
            key,
            value,
            None,
            cu_seq_lens_q=torch.tensor([0, 3]),
        )
    except NotImplementedError as error:
        assert "packed" in str(error)
    else:
        raise AssertionError("packed simulator input was not rejected")


def test_left_padding_position_ids_are_not_mistaken_for_packing() -> None:
    torch.manual_seed(10)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.randn(1, 1, 5, 4)
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    position_ids = torch.tensor([[1, 1, 0, 1, 2]])
    actual, _ = simulated_flash_attention_forward(
        _module(query_chunk=2, key_chunk=3),
        query,
        key,
        value,
        mask,
        position_ids=position_ids,
    )
    expected = _dense_padded(query, key, value, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_qk_pv_quantization_modes_are_independent_and_counted() -> None:
    torch.manual_seed(5)
    query = torch.randn(1, 4, 5, 4) * 3
    key = torch.randn(1, 2, 7, 4) * 2
    value = torch.randn(1, 2, 7, 4) * 4
    outputs = {}
    for name, qk_quant, pv_quant in (
        ("none", False, False),
        ("qk", True, False),
        ("pv", False, True),
        ("both", True, True),
    ):
        module = _module(
            qk_quant=qk_quant,
            pv_quant=pv_quant,
            query_chunk=3,
            key_chunk=4,
            bits=2,
        )
        outputs[name], _ = simulated_flash_attention_forward(
            module, query, key, value, None
        )
        stats = module.faquant_simulated_attention_stats
        assert stats["calls"] == 1
        assert stats["qk_tiles"] == stats["pv_tiles"] == 4
        assert stats["qk_quantized_tiles"] == (4 if qk_quant else 0)
        assert stats["pv_quantized_tiles"] == (4 if pv_quant else 0)
        assert stats["q_quant_calls"] == (2 if qk_quant else 0)
        assert stats["k_quant_calls"] == (4 if qk_quant else 0)
        assert stats["p_quant_calls"] == (4 if pv_quant else 0)
        assert stats["v_quant_calls"] == (4 if pv_quant else 0)
        for operand, enabled in (
            ("q", qk_quant),
            ("k", qk_quant),
            ("p", pv_quant),
            ("v", pv_quant),
        ):
            assert (stats[f"{operand}_quantized_values"] > 0) is enabled
            assert (stats[f"{operand}_quant_blocks"] > 0) is enabled
            expected_padding = 20 if operand == "p" and enabled else 0
            assert stats[f"{operand}_quant_padding_values"] == expected_padding
        assert torch.isfinite(outputs[name]).all()
    assert not torch.equal(outputs["none"], outputs["qk"])
    assert not torch.equal(outputs["none"], outputs["pv"])
    assert not torch.equal(outputs["qk"], outputs["both"])


def test_single_tile_quantization_matches_operand_level_reference() -> None:
    torch.manual_seed(12)
    query = torch.randn(1, 2, 2, 4) * 3
    key = torch.randn(1, 2, 3, 4) * 2
    value = torch.randn(1, 2, 3, 4) * 4
    quant_kwargs = {
        "bits": 2,
        "group_size": 4,
        "symmetric": True,
        "clip_ratio": 1.0,
    }

    def quantize_probability_tail(probabilities: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(probabilities, (0, 1))
        return fake_quantize(padded, **quant_kwargs)[..., :3]

    for qk_quant, pv_quant in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        actual, _ = simulated_flash_attention_forward(
            _module(
                qk_quant=qk_quant,
                pv_quant=pv_quant,
                query_chunk=4,
                key_chunk=4,
                bits=2,
                group_size=4,
            ),
            query,
            key,
            value,
            None,
            is_causal=False,
        )
        query_operand = fake_quantize(query, **quant_kwargs) if qk_quant else query
        key_operand = fake_quantize(key, **quant_kwargs) if qk_quant else key
        scores = (query_operand.float() @ key_operand.float().transpose(-1, -2)) * 0.5
        probabilities = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        denominator = probabilities.sum(dim=-1, keepdim=True)
        probability_operand = (
            quantize_probability_tail(probabilities) if pv_quant else probabilities
        )
        value_operand = fake_quantize(value, **quant_kwargs) if pv_quant else value
        expected = (probability_operand.float() @ value_operand.float()) / denominator
        torch.testing.assert_close(
            actual, expected.transpose(1, 2), atol=2e-6, rtol=2e-6
        )


def test_pv_normalizer_probe_leaves_outputs_and_stats_untouched() -> None:
    torch.manual_seed(13)
    query = torch.randn(1, 4, 6, 4)
    key = torch.randn(1, 2, 9, 4)
    value = torch.randn(1, 2, 9, 4)
    probed = _module(qk_quant=True, pv_quant=True, query_chunk=2, key_chunk=4)
    probed.faquant_pv_normalizer_diagnostics = new_pv_normalizer_diagnostics()
    plain = _module(qk_quant=True, pv_quant=True, query_chunk=2, key_chunk=4)

    probed_output, _ = simulated_flash_attention_forward(
        probed, query, key, value, None
    )
    plain_output, _ = simulated_flash_attention_forward(plain, query, key, value, None)

    assert torch.equal(probed_output, plain_output)
    assert (
        probed.faquant_simulated_attention_stats
        == plain.faquant_simulated_attention_stats
    )
    assert probed.faquant_pv_normalizer_diagnostics["rows"] > 0


def test_pv_normalizer_probe_is_idle_without_pv_quantization() -> None:
    torch.manual_seed(14)
    query = torch.randn(1, 2, 4, 4)
    key = torch.randn(1, 1, 6, 4)
    value = torch.randn(1, 1, 6, 4)
    module = _module(qk_quant=True, pv_quant=False, query_chunk=2, key_chunk=3)
    module.faquant_pv_normalizer_diagnostics = new_pv_normalizer_diagnostics()

    simulated_flash_attention_forward(module, query, key, value, None)

    summary = summarize_pv_normalizer_diagnostics(
        module.faquant_pv_normalizer_diagnostics
    )
    assert summary["rows"] == 0.0
    assert "mass_ratio_mean" not in summary


def test_pv_normalizer_probe_measures_dense_mass_ratio() -> None:
    torch.manual_seed(12)
    query = torch.randn(1, 2, 2, 4) * 3
    key = torch.randn(1, 2, 3, 4) * 2
    value = torch.randn(1, 2, 3, 4) * 4
    quant_kwargs = {
        "bits": 2,
        "group_size": 4,
        "symmetric": True,
        "clip_ratio": 1.0,
    }
    module = _module(pv_quant=True, query_chunk=4, key_chunk=4, bits=2, group_size=4)
    module.faquant_pv_normalizer_diagnostics = new_pv_normalizer_diagnostics()

    simulated_flash_attention_forward(
        module, query, key, value, None, is_causal=False
    )

    scores = (query.float() @ key.float().transpose(-1, -2)) * 0.5
    probabilities = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    padded = torch.nn.functional.pad(probabilities, (0, 1))
    quantized = fake_quantize(padded, **quant_kwargs)[..., :3]
    ratios = quantized.sum(dim=-1) / probabilities.sum(dim=-1)
    error = quantized - probabilities
    expected_l2 = float(
        error.pow(2).sum().sqrt() / probabilities.pow(2).sum().sqrt()
    )
    expected_collapse = float(
        torch.count_nonzero(quantized.eq(0) & probabilities.gt(0))
    ) / float(probabilities.numel())

    summary = summarize_pv_normalizer_diagnostics(
        module.faquant_pv_normalizer_diagnostics
    )
    assert summary["rows"] == float(ratios.numel())
    assert abs(summary["mass_ratio_mean"] - float(ratios.mean())) < 1e-6
    assert abs(summary["mass_ratio_min"] - float(ratios.min())) < 1e-6
    assert abs(summary["mass_ratio_max"] - float(ratios.max())) < 1e-6
    assert abs(summary["p_relative_l2"] - expected_l2) < 1e-6
    assert abs(summary["p_zero_collapse_rate"] - expected_collapse) < 1e-12
    # The probe must see a real mismatch here, otherwise it proves nothing.
    assert summary["p_zero_collapse_rate"] > 0.0


def test_pv_normalizer_probe_rescale_is_consistent_across_key_tiles() -> None:
    # Constant scores make every tile-local probability exactly 1.0, which an
    # absmax-scaled grid represents exactly. A correct online-softmax rescale must
    # therefore keep the shadow denominator equal to the real one across many key
    # tiles; rescaling only one of the two would show up as a drift from 1.
    query = torch.zeros(1, 2, 6, 4)
    key = torch.zeros(1, 1, 12, 4)
    value = torch.randn(1, 1, 12, 4)
    module = _module(pv_quant=True, query_chunk=3, key_chunk=4, bits=4, group_size=4)
    module.faquant_pv_normalizer_diagnostics = new_pv_normalizer_diagnostics()

    simulated_flash_attention_forward(module, query, key, value, None)

    summary = summarize_pv_normalizer_diagnostics(
        module.faquant_pv_normalizer_diagnostics
    )
    assert summary["rows"] > 0.0
    assert abs(summary["mass_ratio_mean"] - 1.0) < 1e-9
    assert abs(summary["mass_ratio_min"] - 1.0) < 1e-9
    assert abs(summary["mass_ratio_max"] - 1.0) < 1e-9
    assert summary["p_zero_collapse_rate"] == 0.0


def test_pv_normalizer_modes_match_reference_without_quantization() -> None:
    torch.manual_seed(21)
    query = torch.randn(1, 4, 6, 8)
    key = torch.randn(1, 2, 9, 8)
    value = torch.randn(1, 2, 9, 8)
    expected = _dense_causal(query, key, value)

    outputs = {}
    for mode in PV_NORMALIZER_MODES:
        outputs[mode], _ = simulated_flash_attention_forward(
            _module(
                query_chunk=2, key_chunk=4, group_size=8, pv_normalizer_mode=mode
            ),
            query,
            key,
            value,
            None,
        )
        torch.testing.assert_close(outputs[mode], expected, atol=2e-6, rtol=2e-6)
    # With no PV quantization the two normalizer sources are the same tensor, so
    # the mode must not even perturb the last bit.
    assert torch.equal(outputs["unquantized"], outputs["quantized_same"])


def test_pv_normalizer_modes_match_when_probabilities_are_exactly_representable() -> None:
    # Constant scores make every tile-local probability exactly 1.0, so P_hat == P
    # and the two normalizers are numerically the same accumulation.
    query = torch.zeros(1, 2, 6, 4)
    key = torch.zeros(1, 1, 12, 4)
    value = torch.randn(1, 1, 12, 4)

    outputs = {}
    for mode in PV_NORMALIZER_MODES:
        module = _module(
            pv_quant=True,
            query_chunk=3,
            key_chunk=4,
            bits=4,
            group_size=4,
            pv_normalizer_mode=mode,
        )
        outputs[mode], _ = simulated_flash_attention_forward(
            module, query, key, value, None
        )
        assert module.faquant_simulated_attention_stats["p_quantized_values"] > 0

    assert torch.equal(outputs["unquantized"], outputs["quantized_same"])


def test_quantized_same_normalizer_is_exactly_one_for_unit_values() -> None:
    # V = 1 turns attention into a convex combination of ones, so any correctly
    # normalized output is 1. Under quantized_same the numerator and denominator
    # are the same row sums of P_hat, which holds however coarse P_hat is; the
    # historical mismatch instead divides sum(P_hat) by sum(P) and drifts.
    torch.manual_seed(22)
    query = torch.randn(1, 4, 5, 4) * 2
    key = torch.randn(1, 2, 5, 4) * 2
    value = torch.ones(1, 2, 5, 4)

    reordered, _ = simulated_flash_attention_forward(
        _module(
            pv_quant=True,
            query_chunk=2,
            key_chunk=4,
            bits=2,
            group_size=4,
            pv_normalizer_mode="quantized_same",
        ),
        query,
        key,
        value,
        None,
    )
    mismatched, _ = simulated_flash_attention_forward(
        _module(pv_quant=True, query_chunk=2, key_chunk=4, bits=2, group_size=4),
        query,
        key,
        value,
        None,
    )

    torch.testing.assert_close(
        reordered, torch.ones_like(reordered), atol=1e-6, rtol=1e-6
    )
    assert float((mismatched - 1.0).abs().max()) > 1e-3


def test_quantized_same_normalizer_is_stable_across_tile_shapes() -> None:
    # Key tiles are whole multiples of the quantization group, so retiling only
    # reschedules the running max. Absmax-symmetric quantization commutes with the
    # positive per-tile rescale, so both normalizers must be tiling-invariant.
    torch.manual_seed(23)
    query = torch.randn(1, 4, 8, 4)
    key = torch.randn(1, 2, 16, 4)
    value = torch.randn(1, 2, 16, 4)

    for mode in PV_NORMALIZER_MODES:
        baseline = None
        for query_chunk, key_chunk in ((2, 4), (4, 8), (8, 16)):
            actual, _ = simulated_flash_attention_forward(
                _module(
                    qk_quant=True,
                    pv_quant=True,
                    query_chunk=query_chunk,
                    key_chunk=key_chunk,
                    bits=4,
                    group_size=4,
                    pv_normalizer_mode=mode,
                ),
                query,
                key,
                value,
                None,
            )
            if baseline is None:
                baseline = actual
            else:
                torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-6)


def test_quantized_same_normalizer_matches_dense_gqa_with_masked_short_tail() -> None:
    # GQA (4 query heads over 2 KV heads), bottom-right causal masking and a key
    # length that leaves a 3-wide tail group, all in one key tile so the whole
    # computation has a closed-form dense reference.
    torch.manual_seed(24)
    query = torch.randn(1, 4, 3, 4) * 2
    key = torch.randn(1, 2, 7, 4) * 2
    value = torch.randn(1, 2, 7, 4) * 2
    quant_kwargs = {
        "bits": 3,
        "group_size": 4,
        "symmetric": True,
        "clip_ratio": 1.0,
    }

    actual, _ = simulated_flash_attention_forward(
        _module(
            pv_quant=True,
            query_chunk=4,
            key_chunk=8,
            bits=3,
            group_size=4,
            pv_normalizer_mode="quantized_same",
        ),
        query,
        key,
        value,
        None,
    )

    repeated_key = _repeat_kv(key, 2)
    repeated_value = _repeat_kv(value, 2)
    scores = (query.float() @ repeated_key.float().transpose(-1, -2)) * 0.5
    query_positions = torch.arange(3) + 7 - 3
    allowed = torch.arange(7)[None, :] <= query_positions[:, None]
    scores = scores.masked_fill(~allowed, -torch.inf)
    probabilities = torch.where(
        allowed,
        torch.exp(scores - scores.amax(dim=-1, keepdim=True)),
        torch.zeros_like(scores),
    )
    padded = torch.nn.functional.pad(probabilities, (0, 1))
    quantized_probabilities = fake_quantize(padded, **quant_kwargs)[..., :7]
    quantized_value = fake_quantize(repeated_value, **quant_kwargs)
    expected = (
        quantized_probabilities @ quantized_value.float()
    ) / quantized_probabilities.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected.transpose(1, 2), atol=2e-6, rtol=2e-6)


def test_mass_ratio_probe_reports_the_same_histogram_in_both_modes() -> None:
    # The probe always compares sum(P_hat) against sum(P) by shadowing whichever
    # source the mode does not use, so its verdict on how much mass the normalizer
    # loses cannot depend on which normalizer is currently installed.
    torch.manual_seed(25)
    query = torch.randn(1, 4, 6, 4) * 2
    key = torch.randn(1, 2, 9, 4) * 2
    value = torch.randn(1, 2, 9, 4) * 2

    summaries = {}
    for mode in PV_NORMALIZER_MODES:
        module = _module(
            pv_quant=True,
            query_chunk=2,
            key_chunk=4,
            bits=2,
            group_size=4,
            pv_normalizer_mode=mode,
        )
        module.faquant_pv_normalizer_diagnostics = new_pv_normalizer_diagnostics()
        simulated_flash_attention_forward(module, query, key, value, None)
        summaries[mode] = summarize_pv_normalizer_diagnostics(
            module.faquant_pv_normalizer_diagnostics
        )

    mismatched, reordered = summaries["unquantized"], summaries["quantized_same"]
    assert mismatched["rows"] == reordered["rows"]
    assert mismatched["mass_ratio_histogram"] == reordered["mass_ratio_histogram"]
    assert sum(mismatched["mass_ratio_histogram"]) == mismatched["rows"]
    for field in (
        "mass_ratio_mean",
        "mass_ratio_min",
        "mass_ratio_max",
        "mass_ratio_fraction_below_one",
        "p_relative_l2",
        "p_zero_collapse_rate",
    ):
        assert mismatched[field] == reordered[field]
    # A probe that measured nothing would satisfy the equalities above vacuously.
    assert mismatched["mean_relative_scale_error"] < -1e-3
    assert mismatched["mass_ratio_fraction_below_one"] > 0.5


def test_tile_quantizer_keeps_a_gradient_path_without_changing_the_forward():
    """Attention-aware QAD needs gradients through the quantized operands.

    ``fake_quantize`` returns a tensor with no ``grad_fn``.  Left alone that is
    invisible during evaluation but would silently detach q/k/v_proj from the
    loss the moment the attention core is quantized during training, so the
    tile quantizer applies a straight-through estimator whenever its input
    carries a gradient.  The forward has to stay bit-exact, since the deployed
    numerics must not depend on whether autograd is recording.
    """

    torch.manual_seed(0)
    kwargs = dict(
        quant_format="hif4", bits=4, group_size=64, symmetric=True, clip_ratio=1.0
    )
    # Both an aligned tile and one whose final group is short.
    for width in (128, 100):
        tensor = torch.randn(2, 5, width)

        frozen = _fake_quantize_tail(tensor, **kwargs)
        assert not frozen.requires_grad
        assert frozen.shape == tensor.shape

        tracked = tensor.clone().requires_grad_(True)
        differentiable = _fake_quantize_tail(tracked, **kwargs)
        assert differentiable.requires_grad
        assert torch.equal(differentiable, frozen)

        differentiable.sum().backward()
        assert tracked.grad is not None
        assert torch.equal(tracked.grad, torch.ones_like(tracked))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The inference simulator updates online-softmax state in place. "
        "QAD uses the differentiable kernel in qad_attention instead."
    ),
)
def test_simulated_attention_is_differentiable_end_to_end():
    torch.manual_seed(0)
    shape = (1, 2, 6, 8)
    tracked = tuple(
        torch.randn(shape, dtype=torch.float32).requires_grad_(True) for _ in range(3)
    )
    module = _module(
        qk_quant=True, pv_quant=True, query_chunk=3, key_chunk=4, bits=3, group_size=4
    )
    output, _ = simulated_flash_attention_forward(module, *tracked, None)
    output.sum().backward()
    assert all(tensor.grad is not None for tensor in tracked)
