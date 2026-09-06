"""The differentiable core must be the deployed core, not something like it.

``quantized_eager_attention_forward`` exists so QAD can train against the QK/PV
quantization the deliverable is measured with.  That is only worth anything if
the two agree numerically, so the claim in the module docstring -- that the
tiled kernel's recurrence collapses to the single-tile formula once a tile
covers every key -- is tested here rather than argued.

The second test measures what is genuinely different: at the deployed 128-key
tiling, P is quantized in the scale each tile's own running max sets, so a
three-tile row hands HiF4 different inputs than a one-tile row.  That gap is
recorded rather than asserted away, because it is the honest distance between
what training sees and what evaluation reports.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from faquant.qad_attention import quantized_eager_attention_forward
from faquant.simulated_flash_attention import (
    new_simulated_attention_stats,
    simulated_flash_attention_forward,
)


def _module(
    *,
    qk_quant: bool,
    pv_quant: bool,
    key_chunk: int,
    query_chunk: int = 512,
    pv_normalizer_mode: str = "unquantized",
    quant_format: str = "hif4",
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
            "quant_format": quant_format,
            "bits": 4,
            "group_size": 64,
            "symmetric": True,
            "clip_ratio": 1.0,
        },
        faquant_simulated_attention_stats=new_simulated_attention_stats(),
    )


def _inputs(
    *, batch: int = 2, heads: int = 4, kv_heads: int = 2, length: int = 96, dim: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    query = torch.randn(batch, heads, length, dim, dtype=torch.float32)
    key = torch.randn(batch, kv_heads, length, dim, dtype=torch.float32)
    value = torch.randn(batch, kv_heads, length, dim, dtype=torch.float32)
    return query, key, value


def _causal_additive_mask(length: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length)
    allowed = positions[None, :] <= positions[:, None]
    mask = torch.zeros(1, 1, length, length, dtype=dtype)
    return mask.masked_fill(~allowed[None, None], torch.finfo(dtype).min)


@pytest.mark.parametrize(
    "qk_quant,pv_quant,normalizer",
    [
        (False, False, "unquantized"),
        (True, False, "unquantized"),
        (True, True, "unquantized"),
        (True, True, "quantized_same"),
    ],
)
def test_matches_simulated_kernel_when_one_tile_covers_every_key(
    qk_quant: bool, pv_quant: bool, normalizer: str
) -> None:
    query, key, value = _inputs()
    length = query.shape[-2]

    simulated, _ = simulated_flash_attention_forward(
        _module(
            qk_quant=qk_quant,
            pv_quant=pv_quant,
            key_chunk=length,
            pv_normalizer_mode=normalizer,
        ),
        query,
        key,
        value,
        None,
    )
    differentiable, _ = quantized_eager_attention_forward(
        _module(
            qk_quant=qk_quant,
            pv_quant=pv_quant,
            key_chunk=length,
            pv_normalizer_mode=normalizer,
        ),
        query,
        key,
        value,
        _causal_additive_mask(length, query.dtype),
        scaling=query.shape[-1] ** -0.5,
    )

    assert simulated.shape == differentiable.shape
    torch.testing.assert_close(differentiable, simulated, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mask_kind", ["none", "two_dimensional", "four_dimensional"])
def test_causality_survives_every_mask_convention(mask_kind: str) -> None:
    """The failure this guards against is silent, not loud.

    Loaded with ``flash_attention_2`` the model passes a 2-D padding mask or
    ``None`` and expects the core to be causal on its own. A core that only
    understood 4-D additive masks would quietly attend to the future and still
    produce plausible losses, so each convention is checked against the
    simulated kernel, which resolves causality the same way.
    """

    query, key, value = _inputs(length=64)
    length = query.shape[-2]
    common = dict(qk_quant=True, pv_quant=True, pv_normalizer_mode="quantized_same")

    if mask_kind == "none":
        simulated_mask = differentiable_mask = None
    elif mask_kind == "two_dimensional":
        padding = torch.ones(query.shape[0], length, dtype=torch.long)
        padding[1, :8] = 0
        simulated_mask = differentiable_mask = padding
    else:
        simulated_mask = None
        differentiable_mask = _causal_additive_mask(length, query.dtype)

    simulated, _ = simulated_flash_attention_forward(
        _module(key_chunk=length, **common), query, key, value, simulated_mask
    )
    differentiable, _ = quantized_eager_attention_forward(
        _module(key_chunk=length, **common),
        query,
        key,
        value,
        differentiable_mask,
        scaling=query.shape[-1] ** -0.5,
    )
    torch.testing.assert_close(differentiable, simulated, atol=1e-5, rtol=1e-5)

    # An explicitly non-causal reference must disagree, or the test above would
    # pass just as happily on a core that ignored causality altogether.
    if mask_kind == "none":
        module = _module(key_chunk=length, **common)
        module.is_causal = False
        bidirectional, _ = quantized_eager_attention_forward(
            module, query, key, value, None, scaling=query.shape[-1] ** -0.5
        )
        assert not torch.allclose(bidirectional, differentiable, atol=1e-3)


def test_fully_masked_rows_do_not_poison_gradients() -> None:
    """Left padding produces rows with nothing to attend to; NaN must not spread."""

    query, key, value = _inputs(length=32)
    query.requires_grad_(True)
    padding = torch.ones(query.shape[0], 32, dtype=torch.long)
    padding[0, :4] = 0

    output, _ = quantized_eager_attention_forward(
        _module(qk_quant=True, pv_quant=True, key_chunk=32),
        query,
        key,
        value,
        padding,
        scaling=query.shape[-1] ** -0.5,
    )
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(query.grad).all()


def test_gradients_reach_every_operand() -> None:
    """The whole point: q/k/v must stay attached to the loss through the core."""

    query, key, value = _inputs(length=64)
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    output, _ = quantized_eager_attention_forward(
        _module(
            qk_quant=True,
            pv_quant=True,
            key_chunk=64,
            pv_normalizer_mode="quantized_same",
        ),
        query,
        key,
        value,
        _causal_additive_mask(64, query.dtype),
        scaling=query.shape[-1] ** -0.5,
    )
    output.square().mean().backward()

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is not finite"
        assert tensor.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_deployed_tiling_shifts_the_result_only_slightly() -> None:
    """Quantify the one real difference: P's scale is set per tile.

    Training on one tile and reporting on three is only defensible if the two
    land in the same place.  The bound here is deliberately loose enough to
    pass and tight enough to fail if tiling ever became a first-order effect;
    the measured value is printed so a regression shows up as a number.
    """

    query, key, value = _inputs(length=96)
    length = query.shape[-2]
    common = dict(qk_quant=True, pv_quant=True, pv_normalizer_mode="quantized_same")

    tiled, _ = simulated_flash_attention_forward(
        _module(key_chunk=32, query_chunk=32, **common), query, key, value, None
    )
    single, _ = simulated_flash_attention_forward(
        _module(key_chunk=length, **common), query, key, value, None
    )

    difference = (tiled - single).abs().max().item()
    reference = single.abs().max().item()
    print(f"\ntiled-vs-single max abs difference {difference:.3e} "
          f"({100.0 * difference / reference:.3f}% of output scale)")
    assert difference < 0.05 * reference
