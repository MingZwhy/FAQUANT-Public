"""A deployed-path attention core that a QAD run can actually backpropagate.

QAD trains 252 HiF4 linears while the attention core runs exact; deployment
then quantizes the QK and PV products on top.  Measuring both readouts on the
same checkpoints (doc section 11) showed what that costs: the attention chain
is 1.05 pp of the 2.29 pp gap at the PTQ start, and across fourteen checkpoints
its size wanders between 0.39 and 1.45 pp -- a spread larger than everything
QAD wins.  Training cannot damp what it cannot see, so half the noise on the
deliverable metric is structural.

``simulated_flash_attention_forward`` is the deployed core, but it cannot be
trained through: it accumulates its running max, denominator and output in
place, and autograd rejects that.  Rewriting it is not an option either, since
it is the instrument every historical number was measured with.

So this is a second implementation of the same arithmetic, in one tile.  That
is not an approximation of the tiled kernel -- it is the tiled kernel's own
formula at ``key_chunk_size >= key_length``: the running max then starts at
-inf on the only pass, ``previous_scale`` is identically zero, and the
recurrence collapses to

    p = exp(s - rowmax(s));  out = quant(p) @ quant(v) / sum(normalizer)

which is what is written below, out of place and differentiable.  Sequences in
the QAD corpus cap at 300 tokens, so materializing the score matrix costs about
0.77 GB across all 36 layers.

Both quantizers are ``_fake_quantize_tail``, the same call the deployed kernel
makes, which is already straight-through wrapped precisely so the operands stay
attached to the loss when the core is trained.

The one thing that does *not* follow from the collapse argument is whether the
deployed 128-key tiling matters: P is quantized before normalization, in the
scale its own tile's running max sets, so a 3-tile row and a 1-tile row hand
HiF4 slightly different inputs.  ``tests/test_qad_attention.py`` measures that
directly against the simulator instead of assuming it away.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import PV_NORMALIZER_MODES
from .simulated_flash_attention import (
    _fake_quantize_tail,
    _repeat_kv,
    _two_dimensional_mask,
)


def _resolve_allowed(
    module: nn.Module,
    attention_mask: Optional[torch.Tensor],
    *,
    query_length: int,
    key_length: int,
    device: torch.device,
    is_causal: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(boolean allowed, additive mask)`` for the score matrix.

    Which of the two arrives depends on how the model was loaded, and getting
    it wrong is silent: under ``flash_attention_2`` transformers hands down a
    2-D padding mask or ``None`` and leaves causality to the kernel, so a core
    that simply adds whatever it is given would train a *bidirectional* model
    and never say so.  Causality is therefore reconstructed here whenever the
    mask does not already carry it, exactly as the simulated kernel does.
    """

    allowed: Optional[torch.Tensor] = None
    additive: Optional[torch.Tensor] = None

    if attention_mask is not None:
        if attention_mask.ndim == 2:
            query_valid, key_valid = _two_dimensional_mask(
                attention_mask, query_length=query_length, key_length=key_length
            )
            allowed = query_valid[:, None, :, None] & key_valid[:, None, None, :]
        elif attention_mask.ndim in (3, 4):
            mask = (
                attention_mask
                if attention_mask.ndim == 4
                else attention_mask[:, None, :, :]
            )
            mask = mask[..., :key_length]
            if mask.dtype == torch.bool:
                allowed = mask
            else:
                # A float mask from sdpa/eager already encodes causality; adding
                # it again below would be harmless but reconstructing causality
                # on top of it would not, so this branch opts out of that.
                additive = mask
        else:
            raise ValueError("attention mask must be rank 2, 3, or 4")

    if is_causal and additive is None:
        # Queries sit at the end of the key sequence when a cache is present.
        query_positions = torch.arange(
            key_length - query_length, key_length, device=device
        )
        key_positions = torch.arange(key_length, device=device)
        causal = key_positions[None, :] <= query_positions[:, None]
        allowed = causal if allowed is None else (allowed & causal)

    return allowed, additive


def quantized_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Single-tile deployed attention with gradients through every operand."""

    del dropout  # QAD runs without attention dropout.
    if sliding_window is not None:
        raise NotImplementedError(
            "the differentiable attention core does not implement sliding "
            "windows; Qwen3-8B attends fully, so reaching this means the model "
            "changed and the core has to change with it"
        )

    quant_kwargs = module.faquant_matmul_quant_kwargs
    qk_quant_kwargs = getattr(
        module, "faquant_qk_matmul_quant_kwargs", quant_kwargs
    )
    pv_quant_kwargs = getattr(
        module, "faquant_pv_matmul_quant_kwargs", quant_kwargs
    )
    qk_quant = bool(getattr(module, "faquant_qk_matmul_quant", False))
    pv_quant = bool(getattr(module, "faquant_pv_matmul_quant", False))
    normalizer_mode = getattr(module, "faquant_pv_normalizer_mode", "unquantized")
    if normalizer_mode not in PV_NORMALIZER_MODES:
        raise ValueError(
            f"unsupported pv_normalizer_mode {normalizer_mode!r}; "
            f"expected one of {sorted(PV_NORMALIZER_MODES)}"
        )

    head_dim = query.shape[-1]
    scale = float(scaling if scaling is not None else head_dim**-0.5)
    repeats = query.shape[1] // key.shape[1]

    # Quantize K and V before GQA repetition, as the deployed kernel does. The
    # groups only span head_dim, so repeating afterwards is bitwise identical
    # and the repeated copies cost nothing extra to quantize.
    key_operand = (
        _fake_quantize_tail(key, **qk_quant_kwargs) if qk_quant else key
    )
    value_operand = (
        _fake_quantize_tail(value, **pv_quant_kwargs) if pv_quant else value
    )
    key_operand = _repeat_kv(key_operand, repeats)
    value_operand = _repeat_kv(value_operand, repeats)
    query_operand = (
        _fake_quantize_tail(query, **qk_quant_kwargs) if qk_quant else query
    )

    is_causal = kwargs.get("is_causal")
    is_causal = bool(
        getattr(module, "is_causal", True) if is_causal is None else is_causal
    )
    allowed, additive = _resolve_allowed(
        module,
        attention_mask,
        query_length=query.shape[-2],
        key_length=key_operand.shape[-2],
        device=query.device,
        is_causal=is_causal,
    )

    scores = torch.matmul(query_operand, key_operand.transpose(2, 3)).float() * scale
    if additive is not None:
        scores = scores + additive.float()
    if allowed is not None:
        scores = scores.masked_fill(~allowed, -torch.inf)

    # Unnormalized exponentials in the row-max scale: this is the tensor the
    # deployed kernel quantizes, not the softmax output. Quantizing after
    # normalization would put P on a different grid and understate the damage.
    row_max = scores.amax(dim=-1, keepdim=True)
    # A fully masked row leaves row_max at -inf, and -inf minus -inf is a NaN
    # that would propagate into the gradient of every other row through the
    # shared reduction. Pinning it to zero costs nothing: every score in such a
    # row is -inf, so the exponentials are zero either way.
    row_max = torch.where(
        torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
    )
    probabilities = torch.exp(scores - row_max)

    probability_operand = (
        _fake_quantize_tail(probabilities, **pv_quant_kwargs)
        if pv_quant
        else probabilities
    )
    # P-Reordering: numerator and denominator read the same quantized P, so the
    # quantization error partly cancels in the ratio instead of accumulating.
    normalizer_source = (
        probability_operand
        if normalizer_mode == "quantized_same"
        else probabilities
    )
    denominator = normalizer_source.sum(dim=-1, keepdim=True)
    accumulator = torch.matmul(probability_operand, value_operand.float())
    output = torch.where(
        denominator > 0,
        accumulator / denominator.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(accumulator),
    )
    return output.to(query.dtype).transpose(1, 2).contiguous(), None
