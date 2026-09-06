from __future__ import annotations

import math
from typing import Optional

import torch

from .config import PV_NORMALIZER_MODES
from .quantization import fake_quantize


SIMULATED_ATTENTION_IMPLEMENTATION = "faquant_simulated"

_QUANTIZED_OPERANDS = ("q", "k", "p", "v")


def new_simulated_attention_stats() -> dict[str, int]:
    """Return zeroed counters for one simulated-attention runtime."""

    stats = {
        "calls": 0,
        "qk_tiles": 0,
        "pv_tiles": 0,
        "qk_quantized_tiles": 0,
        "pv_quantized_tiles": 0,
    }
    for operand in _QUANTIZED_OPERANDS:
        stats[f"{operand}_quant_calls"] = 0
        stats[f"{operand}_quantized_values"] = 0
        stats[f"{operand}_quant_blocks"] = 0
        stats[f"{operand}_quant_padding_values"] = 0
    return stats


PV_MASS_RATIO_BIN_EDGES = (
    0.5,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    1.0,
    1.01,
    1.05,
    1.1,
    1.3,
)
"""Right-open bucket edges for the ``sum(P_hat)/sum(P)`` histogram.

Moments alone cannot tell a uniform scale error from a bimodal one, and the
distribution shape decides whether a normalizer fix is a clean global correction
or something that helps some rows while hurting others.
"""

_PV_DIAGNOSTIC_KEYS = (
    "rows",
    "mass_ratio_sum",
    "mass_ratio_sq_sum",
    "mass_ratio_min",
    "mass_ratio_max",
    "p_values",
    "p_zeroed_values",
    "p_error_sq_sum",
    "p_reference_sq_sum",
)
_PV_SCALAR_DIAGNOSTIC_KEYS = tuple(
    key
    for key in _PV_DIAGNOSTIC_KEYS
    if key not in ("mass_ratio_min", "mass_ratio_max")
)


def new_pv_normalizer_diagnostics() -> dict[str, float]:
    """Return zeroed accumulators for the PV normalizer-mismatch probe.

    ``mass_ratio`` is the per-query-row ratio between the softmax denominator
    accumulated from the quantized probabilities and the one accumulated from the
    unquantized probabilities. It is measured the same way under both
    ``PV_NORMALIZER_MODES``: whichever source the mode does not use is tracked in
    a shadow accumulator, so the reported ratio always means ``sum(P_hat)/sum(P)``
    and stays comparable across modes. Its distance from 1 is the attention-output
    scale error that ``quantized_same`` removes. Attaching this dict to an
    attention module only adds shadow accounting; the returned attention output is
    unchanged.
    """

    diagnostics: dict[str, object] = dict.fromkeys(_PV_DIAGNOSTIC_KEYS, 0.0)
    diagnostics["mass_ratio_min"] = float("inf")
    diagnostics["mass_ratio_max"] = float("-inf")
    diagnostics["mass_ratio_histogram"] = [0.0] * (len(PV_MASS_RATIO_BIN_EDGES) + 1)
    return diagnostics


def summarize_pv_normalizer_diagnostics(
    diagnostics: dict[str, object],
) -> dict[str, object]:
    """Reduce probe accumulators to interpretable per-run scalars."""

    rows = float(diagnostics["rows"])
    values = float(diagnostics["p_values"])
    reference_energy = float(diagnostics["p_reference_sq_sum"])
    summary: dict[str, object] = {"rows": rows, "p_values": values}
    if rows > 0:
        histogram = list(diagnostics["mass_ratio_histogram"])
        below_one = PV_MASS_RATIO_BIN_EDGES.index(1.0) + 1
        mean = float(diagnostics["mass_ratio_sum"]) / rows
        variance = max(
            float(diagnostics["mass_ratio_sq_sum"]) / rows - mean * mean, 0.0
        )
        summary["mass_ratio_mean"] = mean
        summary["mass_ratio_std"] = math.sqrt(variance)
        summary["mass_ratio_min"] = float(diagnostics["mass_ratio_min"])
        summary["mass_ratio_max"] = float(diagnostics["mass_ratio_max"])
        summary["mass_ratio_bin_edges"] = list(PV_MASS_RATIO_BIN_EDGES)
        summary["mass_ratio_histogram"] = histogram
        summary["mass_ratio_fraction_below_one"] = sum(histogram[:below_one]) / rows
        # The quantity P-Reordering would remove from the output scale.
        summary["mean_relative_scale_error"] = mean - 1.0
    if values > 0:
        summary["p_zero_collapse_rate"] = (
            float(diagnostics["p_zeroed_values"]) / values
        )
    if reference_energy > 0:
        summary["p_relative_l2"] = math.sqrt(
            float(diagnostics["p_error_sq_sum"]) / reference_energy
        )
    return summary


def _record_probability_error(
    diagnostics: dict[str, object],
    *,
    probabilities: torch.Tensor,
    probability_operand: torch.Tensor,
) -> None:
    """Accumulate elementwise P quantization error for one PV tile."""

    reference = probabilities.float()
    error = probability_operand.float() - reference
    diagnostics["p_values"] += float(reference.numel())
    diagnostics["p_zeroed_values"] += float(
        torch.count_nonzero(probability_operand.eq(0) & probabilities.gt(0))
    )
    diagnostics["p_error_sq_sum"] += float(error.pow(2).sum())
    diagnostics["p_reference_sq_sum"] += float(reference.pow(2).sum())


def _record_mass_ratio(
    diagnostics: dict[str, object],
    *,
    quantized: torch.Tensor,
    unquantized: torch.Tensor,
) -> None:
    """Accumulate the per-row quantized/unquantized denominator ratio.

    Only fully accumulated rows reach here, and both denominators were rescaled
    by the same running-max factors, so the ratio is free of online-softmax
    bookkeeping and reflects P quantization alone.
    """

    valid = unquantized > 0
    if not bool(torch.any(valid)):
        return
    ratio = (quantized[valid].double() / unquantized[valid].double()).flatten()
    diagnostics["rows"] += float(ratio.numel())
    diagnostics["mass_ratio_sum"] += float(ratio.sum())
    diagnostics["mass_ratio_sq_sum"] += float(ratio.pow(2).sum())
    diagnostics["mass_ratio_min"] = min(
        diagnostics["mass_ratio_min"], float(ratio.min())
    )
    diagnostics["mass_ratio_max"] = max(
        diagnostics["mass_ratio_max"], float(ratio.max())
    )
    edges = torch.tensor(
        PV_MASS_RATIO_BIN_EDGES, device=ratio.device, dtype=ratio.dtype
    )
    counts = torch.bincount(
        torch.bucketize(ratio, edges),
        minlength=len(PV_MASS_RATIO_BIN_EDGES) + 1,
    )
    histogram = diagnostics["mass_ratio_histogram"]
    for index, count in enumerate(counts.tolist()):
        histogram[index] += float(count)


def _record_quantized_operand(
    stats: dict[str, int],
    operand: str,
    tensor: torch.Tensor,
    *,
    group_size: int,
    repeats: int = 1,
) -> None:
    """Record the exact logical and padded geometry of one operand QDQ."""

    if operand not in _QUANTIZED_OPERANDS:
        raise ValueError(f"unknown attention operand {operand!r}")
    if repeats <= 0:
        raise ValueError("quantized operand repeats must be positive")
    width = tensor.shape[-1]
    effective_group_size = width if group_size == -1 else group_size
    groups_per_vector = math.ceil(width / effective_group_size)
    vectors = tensor.numel() // width
    padding_per_vector = groups_per_vector * effective_group_size - width
    stats[f"{operand}_quant_calls"] += repeats
    stats[f"{operand}_quantized_values"] += tensor.numel() * repeats
    stats[f"{operand}_quant_blocks"] += vectors * groups_per_vector * repeats
    stats[f"{operand}_quant_padding_values"] += (
        vectors * padding_per_vector * repeats
    )


def register_simulated_attention_backend() -> None:
    """Register a portable HF backend that reuses FA's compact 2-D mask."""

    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        flash_attention_mask,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(
        SIMULATED_ATTENTION_IMPLEMENTATION, simulated_flash_attention_forward
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        SIMULATED_ATTENTION_IMPLEMENTATION, flash_attention_mask
    )


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Match Qwen3/FlashAttention's contiguous GQA head mapping."""

    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


def _cached_decode_quantize(
    module: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    operand: str,
    enabled: bool,
    query_length: int,
    quant_kwargs: dict[str, object],
) -> torch.Tensor:
    """Reuse token-local K/V QDQ values while a decode cache grows.

    Attention matmul HiF4 groups only over ``head_dim``. Consequently, adding
    one sequence position cannot change the fitted metadata or QDQ values of
    any existing position. Decode may therefore quantize only the appended
    token and concatenate it to the prior QDQ tensor, exactly matching a full
    re-quantization of the growing K/V cache.
    """

    if operand not in ("k", "v"):
        raise ValueError(f"decode QDQ cache only supports K/V, got {operand!r}")
    cache_name = f"faquant_{operand}_matmul_qdq_cache"
    if not enabled:
        setattr(module, cache_name, None)
        return tensor

    cached = getattr(module, cache_name, None)
    can_append = (
        query_length == 1
        and isinstance(cached, torch.Tensor)
        and cached.device == tensor.device
        and cached.dtype == tensor.dtype
        and cached.shape[:-2] == tensor.shape[:-2]
        and cached.shape[-1] == tensor.shape[-1]
        and cached.shape[-2] + 1 == tensor.shape[-2]
    )
    if can_append:
        quantized_tail = _fake_quantize_tail(
            tensor[..., cached.shape[-2] :, :],
            **quant_kwargs,
        )
        quantized = torch.cat((cached, quantized_tail), dim=-2)
    else:
        quantized = _fake_quantize_tail(tensor, **quant_kwargs)
    setattr(module, cache_name, quantized)
    return quantized


def _fake_quantize_tail(
    tensor: torch.Tensor,
    *,
    quant_format: str = "int",
    bits: int,
    group_size: int,
    symmetric: bool,
    clip_ratio: float,
) -> torch.Tensor:
    """Fake-quantize a tile whose final group may be shorter than group_size.

    ``fake_quantize`` returns a tensor with no ``grad_fn``, which is harmless
    for evaluation but would silently sever q/k/v_proj from the loss if the
    attention core were quantized during QAD training. Wrapping the result in
    the usual straight-through estimator keeps the forward numerics bit-exact
    and restores the gradient path; the plain result is returned when nothing
    upstream needs a gradient, so inference does not pay for the extra tensors.
    """

    if group_size == -1 or tensor.shape[-1] % group_size == 0:
        quantized = fake_quantize(
            tensor,
            quant_format=quant_format,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )
    else:
        remainder = tensor.shape[-1] % group_size
        padding = group_size - remainder
        padded = torch.nn.functional.pad(tensor, (0, padding))
        quantized = fake_quantize(
            padded,
            quant_format=quant_format,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
        )
        quantized = quantized[..., : tensor.shape[-1]]
    if tensor.requires_grad:
        return tensor + (quantized - tensor).detach()
    return quantized


def _two_dimensional_mask(
    attention_mask: torch.Tensor,
    *,
    query_length: int,
    key_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask.shape[-1] < key_length:
        raise ValueError(
            "2-D attention mask is shorter than the simulated key sequence"
        )
    key_valid = attention_mask[:, -key_length:].to(dtype=torch.bool)
    if query_length > key_valid.shape[-1]:
        missing = query_length - key_valid.shape[-1]
        query_valid = torch.cat(
            (
                torch.zeros(
                    key_valid.shape[0],
                    missing,
                    device=key_valid.device,
                    dtype=torch.bool,
                ),
                key_valid,
            ),
            dim=-1,
        )
    else:
        query_valid = key_valid[:, -query_length:]
    return query_valid, key_valid


def _streaming_mask(
    *,
    batch_size: int,
    query_start: int,
    query_stop: int,
    key_start: int,
    key_stop: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    is_causal: bool,
    sliding_window: Optional[int],
    query_valid: Optional[torch.Tensor],
    key_valid: Optional[torch.Tensor],
) -> torch.Tensor:
    query_positions = torch.arange(query_start, query_stop, device=device)
    key_positions = torch.arange(key_start, key_stop, device=device)
    allowed = torch.ones(
        batch_size,
        1,
        query_stop - query_start,
        key_stop - key_start,
        device=device,
        dtype=torch.bool,
    )

    if query_valid is None or key_valid is None:
        anchors = query_positions + key_length - query_length
        if is_causal:
            allowed &= key_positions[None, :] <= anchors[:, None]
        if sliding_window is not None:
            allowed &= key_positions[None, :] > anchors[:, None] - sliding_window
            if not is_causal:
                allowed &= key_positions[None, :] < anchors[:, None] + sliding_window
        return allowed

    # FlashAttention unpads each batch element before applying bottom-right
    # causal masking. Ranks reproduce that behavior without moving the tensors.
    key_ranks = key_valid.long().cumsum(dim=-1) - 1
    query_ranks = query_valid.long().cumsum(dim=-1) - 1
    key_counts = key_valid.sum(dim=-1, keepdim=True)
    query_counts = query_valid.sum(dim=-1, keepdim=True)
    anchors = query_ranks[:, query_start:query_stop] + key_counts - query_counts
    tile_key_ranks = key_ranks[:, key_start:key_stop]
    allowed &= query_valid[:, None, query_start:query_stop, None]
    allowed &= key_valid[:, None, None, key_start:key_stop]
    if is_causal:
        allowed &= tile_key_ranks[:, None, None, :] <= anchors[:, None, :, None]
    if sliding_window is not None:
        allowed &= tile_key_ranks[:, None, None, :] > (
            anchors[:, None, :, None] - sliding_window
        )
        if not is_causal:
            allowed &= tile_key_ranks[:, None, None, :] < (
                anchors[:, None, :, None] + sliding_window
            )
    return allowed


def _apply_four_dimensional_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    query_start: int,
    query_stop: int,
    key_start: int,
    key_stop: int,
    query_length: int,
) -> torch.Tensor:
    mask_query_length = attention_mask.shape[-2]
    if mask_query_length == 1:
        query_slice = slice(0, 1)
    elif mask_query_length >= query_length:
        offset = mask_query_length - query_length
        query_slice = slice(offset + query_start, offset + query_stop)
    else:
        raise ValueError("4-D attention mask is shorter than the query sequence")
    mask = attention_mask[..., query_slice, key_start:key_stop]
    if mask.dtype == torch.bool or not mask.is_floating_point():
        return scores.masked_fill(~mask.to(dtype=torch.bool), -torch.inf)
    invalid = torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min / 2)
    mask = mask.to(device=scores.device, dtype=scores.dtype)
    invalid = invalid.to(device=scores.device)
    return (scores + mask).masked_fill(invalid, -torch.inf)


def simulated_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Pure-PyTorch tiled FlashAttention simulation for inference.

    The online-softmax max, denominator, and output accumulator stay FP32.
    Optional QK quantization fake-quantizes Q/K tile operands. Optional PV
    quantization fake-quantizes the tile-local ``exp(score-running_max)`` and V
    operands; the denominator still uses the unquantized tile probabilities.
    The implementation deliberately does not materialize the full probability
    matrix.
    """

    if dropout:
        raise ValueError("simulated FlashAttention only supports dropout=0 inference")
    if softcap is not None and softcap < 0:
        raise ValueError("softcap must be non-negative")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if kwargs.get("output_attentions", False):
        raise ValueError(
            "simulated FlashAttention does not materialize attention weights"
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("simulated FlashAttention expects rank-4 Q/K/V tensors")
    if key.shape != value.shape:
        raise ValueError("simulated FlashAttention requires matching K/V shapes")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("simulated FlashAttention received incompatible Q/K shapes")

    if any(
        kwargs.get(name) is not None
        for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
    ):
        raise NotImplementedError(
            "packed cu_seqlens are not supported by the simulator"
        )
    position_ids = kwargs.get("position_ids")
    if (
        attention_mask is None
        and isinstance(position_ids, torch.Tensor)
        and position_ids.shape[-1] > 1
    ):
        if torch.any(position_ids[..., 1:] <= position_ids[..., :-1]):
            raise NotImplementedError("packed/reset position_ids are not supported")

    # Static caches can expose a backing tensor longer than the valid sequence.
    valid_key_length = key.shape[-2]
    if attention_mask is not None:
        valid_key_length = min(valid_key_length, attention_mask.shape[-1])
    cache_position = kwargs.get("cache_position")
    if isinstance(cache_position, torch.Tensor) and cache_position.numel():
        valid_key_length = min(valid_key_length, int(cache_position.max().item()) + 1)
    if valid_key_length < key.shape[-2]:
        key = key[:, :, :valid_key_length]
        value = value[:, :, :valid_key_length]

    batch_size, query_heads, query_length, head_dim = query.shape
    _, kv_heads, key_length, _ = key.shape
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if any(size == 0 for size in query.shape) or key_length == 0:
        raise ValueError("simulated FlashAttention does not support zero dimensions")

    kv_repeats = query_heads // kv_heads
    scale = float(scaling if scaling is not None else head_dim**-0.5)
    query_chunk = int(module.faquant_attention_query_chunk_size)
    key_chunk = int(module.faquant_attention_key_chunk_size)
    quant_kwargs = module.faquant_matmul_quant_kwargs
    qk_quant_kwargs = getattr(
        module, "faquant_qk_matmul_quant_kwargs", quant_kwargs
    )
    pv_quant_kwargs = getattr(
        module, "faquant_pv_matmul_quant_kwargs", quant_kwargs
    )
    qk_quant = bool(module.faquant_qk_matmul_quant)
    pv_quant = bool(module.faquant_pv_matmul_quant)
    pv_normalizer_mode = getattr(
        module, "faquant_pv_normalizer_mode", "unquantized"
    )
    if pv_normalizer_mode not in PV_NORMALIZER_MODES:
        raise ValueError(
            f"unsupported pv_normalizer_mode {pv_normalizer_mode!r}; "
            f"expected one of {PV_NORMALIZER_MODES}"
        )
    quantized_same_normalizer = pv_normalizer_mode == "quantized_same"
    pv_diagnostics = getattr(module, "faquant_pv_normalizer_diagnostics", None)
    collect_diagnostics = pv_quant and isinstance(pv_diagnostics, dict)
    requested_causal = kwargs.get("is_causal")
    is_causal = bool(
        getattr(module, "is_causal", True)
        if requested_causal is None
        else requested_causal
    )

    query_valid = key_valid = None
    if attention_mask is not None and attention_mask.ndim == 2:
        # Generation commonly supplies an explicit all-one mask even for a
        # single unpadded prompt.  During multi-token prefill this is exactly
        # the mask-free bottom-right causal case, so normalize it to ``None``
        # and enable fully-future tile elision.  Restrict the device
        # synchronization to prefill; decode has only one query tile and
        # cannot benefit from the optimization.
        all_valid_prefill = (
            getattr(module, "faquant_skip_fully_masked_tiles", True)
            and batch_size == 1
            and query_length > 1
            and bool(torch.all(attention_mask[:, -key_length:]).item())
        )
        if all_valid_prefill:
            attention_mask = None
        else:
            query_valid, key_valid = _two_dimensional_mask(
                attention_mask,
                query_length=query_length,
                key_length=key_length,
            )
    elif attention_mask is not None and attention_mask.ndim not in (3, 4):
        raise ValueError("attention mask must be rank 2, 3, or 4")
    if attention_mask is not None and attention_mask.ndim == 3:
        attention_mask = attention_mask[:, None, :, :]

    # K/V QDQ groups only along head_dim. Quantize before GQA repetition and
    # reuse unchanged historical positions as the autoregressive cache grows.
    # Both transformations are token/head local, so this is bitwise identical
    # to quantizing the fully repeated K/V tensor on every attention call.
    quantized_key = _cached_decode_quantize(
        module,
        key,
        operand="k",
        enabled=qk_quant,
        query_length=query_length,
        quant_kwargs=qk_quant_kwargs,
    )
    quantized_value = _cached_decode_quantize(
        module,
        value,
        operand="v",
        enabled=pv_quant,
        query_length=query_length,
        quant_kwargs=pv_quant_kwargs,
    )
    key = _repeat_kv(key, kv_repeats)
    value = _repeat_kv(value, kv_repeats)
    quantized_key = _repeat_kv(quantized_key, kv_repeats)
    quantized_value = _repeat_kv(quantized_value, kv_repeats)

    quantized_query = (
        _fake_quantize_tail(query, **qk_quant_kwargs) if qk_quant else query
    )
    output = torch.zeros_like(query)
    qk_tiles = pv_tiles = 0

    def process_equal_query_tiles(
        *,
        query_start: int,
        tile_rows: int,
        tile_count: int,
    ) -> None:
        """Process equal-sized logical query tiles in one batched GPU call."""

        nonlocal qk_tiles, pv_tiles
        query_stop = query_start + tile_rows * tile_count
        query_tiles = (
            query[:, :, query_start:query_stop]
            .reshape(batch_size, query_heads, tile_count, tile_rows, head_dim)
            .permute(2, 0, 1, 3, 4)
        )
        query_operands = (
            quantized_query[:, :, query_start:query_stop]
            .reshape(batch_size, query_heads, tile_count, tile_rows, head_dim)
            .permute(2, 0, 1, 3, 4)
        )
        if qk_quant:
            _record_quantized_operand(
                module.faquant_simulated_attention_stats,
                "q",
                query_tiles[0],
                group_size=int(qk_quant_kwargs["group_size"]),
                repeats=tile_count,
            )

        state_shape = (tile_count, batch_size, query_heads, tile_rows, 1)
        running_max = torch.full(
            state_shape,
            -torch.inf,
            device=query.device,
            dtype=torch.float32,
        )
        denominator = torch.zeros_like(running_max)
        shadow_denominator = (
            torch.zeros_like(running_max) if collect_diagnostics else None
        )
        accumulator = torch.zeros(
            tile_count,
            batch_size,
            query_heads,
            tile_rows,
            head_dim,
            device=query.device,
            dtype=torch.float32,
        )

        for key_start in range(0, key_length, key_chunk):
            key_stop = min(key_start + key_chunk, key_length)
            active_tile_start = 0
            if (
                getattr(module, "faquant_skip_fully_masked_tiles", True)
                and attention_mask is None
                and is_causal
                and sliding_window is None
            ):
                # Bottom-right causal alignment gives query position ``q`` the
                # last valid key ``q + key_length - query_length``.  For a
                # fixed key tile, every equal-sized query tile before this
                # index is therefore entirely in the future.  Leaving its
                # online-softmax state untouched is bitwise identical to
                # processing a score tile containing only ``-inf``, while
                # avoiding its QK, P-QDQ, and PV work altogether.
                offset = key_length - query_length
                active_tile_start = max(
                    0,
                    min(
                        tile_count,
                        (key_start - offset - query_start) // tile_rows,
                    ),
                )
            active_tile_count = tile_count - active_tile_start
            if active_tile_count == 0:
                continue

            active_query_start = query_start + active_tile_start * tile_rows
            active_query_operands = query_operands[active_tile_start:]
            active_running_max = running_max[active_tile_start:]
            active_denominator = denominator[active_tile_start:]
            active_shadow = (
                shadow_denominator[active_tile_start:] if collect_diagnostics else None
            )
            active_accumulator = accumulator[active_tile_start:]
            key_tile = key[:, :, key_start:key_stop]
            value_tile = value[:, :, key_start:key_stop]
            if qk_quant:
                key_operand = quantized_key[:, :, key_start:key_stop]
                _record_quantized_operand(
                    module.faquant_simulated_attention_stats,
                    "k",
                    key_tile,
                    group_size=int(qk_quant_kwargs["group_size"]),
                    repeats=active_tile_count,
                )
            else:
                key_operand = key_tile

            scores = torch.matmul(
                active_query_operands.float(),
                key_operand[None].float().transpose(-1, -2),
            )
            scores.mul_(scale)
            if softcap:
                scores = torch.tanh(scores / softcap) * softcap

            if attention_mask is not None and attention_mask.ndim == 4:
                flat_scores = (
                    scores.permute(1, 2, 0, 3, 4)
                    .reshape(
                        batch_size,
                        query_heads,
                        active_tile_count * tile_rows,
                        key_stop - key_start,
                    )
                )
                flat_scores = _apply_four_dimensional_mask(
                    flat_scores,
                    attention_mask,
                    query_start=active_query_start,
                    query_stop=query_stop,
                    key_start=key_start,
                    key_stop=key_stop,
                    query_length=query_length,
                )
                scores = (
                    flat_scores.reshape(
                        batch_size,
                        query_heads,
                        active_tile_count,
                        tile_rows,
                        key_stop - key_start,
                    )
                    .permute(2, 0, 1, 3, 4)
                    .contiguous()
                )
            else:
                allowed = _streaming_mask(
                    batch_size=batch_size,
                    query_start=active_query_start,
                    query_stop=query_stop,
                    key_start=key_start,
                    key_stop=key_stop,
                    query_length=query_length,
                    key_length=key_length,
                    device=query.device,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
                    query_valid=query_valid,
                    key_valid=key_valid,
                )
                allowed = (
                    allowed.reshape(
                        batch_size,
                        1,
                        active_tile_count,
                        tile_rows,
                        key_stop - key_start,
                    )
                    .permute(2, 0, 1, 3, 4)
                )
                scores.masked_fill_(~allowed, -torch.inf)

            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(active_running_max, block_max)
            previous_scale = torch.where(
                torch.isfinite(active_running_max),
                torch.exp(active_running_max - new_max),
                torch.zeros_like(active_running_max),
            )
            probabilities = torch.where(
                torch.isfinite(scores),
                torch.exp(scores - new_max),
                torch.zeros_like(scores),
            )
            if pv_quant:
                probability_operand = _fake_quantize_tail(
                    probabilities, **pv_quant_kwargs
                )
                value_operand = quantized_value[:, :, key_start:key_stop]
                _record_quantized_operand(
                    module.faquant_simulated_attention_stats,
                    "p",
                    probabilities[0],
                    group_size=int(pv_quant_kwargs["group_size"]),
                    repeats=active_tile_count,
                )
                _record_quantized_operand(
                    module.faquant_simulated_attention_stats,
                    "v",
                    value_tile,
                    group_size=int(pv_quant_kwargs["group_size"]),
                    repeats=active_tile_count,
                )
            else:
                probability_operand = probabilities
                value_operand = value_tile
            # P-Reordering keeps numerator and normalizer on the same P, which is
            # what appending a ones column to V computes in one PV GEMM. Both
            # sources share ``previous_scale``, so the running-max rescale can
            # never desynchronize them.
            normalizer_source = (
                probability_operand if quantized_same_normalizer else probabilities
            )
            active_denominator = (
                previous_scale * active_denominator
                + normalizer_source.sum(dim=-1, keepdim=True)
            )
            if collect_diagnostics:
                shadow_source = (
                    probabilities if quantized_same_normalizer else probability_operand
                )
                active_shadow = (
                    previous_scale * active_shadow
                    + shadow_source.sum(dim=-1, keepdim=True)
                )
                _record_probability_error(
                    pv_diagnostics,
                    probabilities=probabilities,
                    probability_operand=probability_operand,
                )
            active_accumulator = previous_scale * active_accumulator + torch.matmul(
                probability_operand.float(), value_operand[None].float()
            )
            denominator[active_tile_start:] = active_denominator
            accumulator[active_tile_start:] = active_accumulator
            if collect_diagnostics:
                shadow_denominator[active_tile_start:] = active_shadow
            running_max[active_tile_start:] = new_max
            qk_tiles += active_tile_count
            pv_tiles += active_tile_count

        if collect_diagnostics:
            quantized_denominator, unquantized_denominator = (
                (denominator, shadow_denominator)
                if quantized_same_normalizer
                else (shadow_denominator, denominator)
            )
            _record_mass_ratio(
                pv_diagnostics,
                quantized=quantized_denominator,
                unquantized=unquantized_denominator,
            )
        normalized = torch.where(
            denominator > 0,
            accumulator / denominator.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(accumulator),
        )
        output[:, :, query_start:query_stop] = (
            normalized.permute(1, 2, 0, 3, 4)
            .reshape(
                batch_size,
                query_heads,
                tile_count * tile_rows,
                head_dim,
            )
            .to(query.dtype)
        )

    full_tile_count = query_length // query_chunk
    if full_tile_count:
        process_equal_query_tiles(
            query_start=0,
            tile_rows=query_chunk,
            tile_count=full_tile_count,
        )
    tail_start = full_tile_count * query_chunk
    if tail_start < query_length:
        process_equal_query_tiles(
            query_start=tail_start,
            tile_rows=query_length - tail_start,
            tile_count=1,
        )

    stats = module.faquant_simulated_attention_stats
    stats["calls"] += 1
    stats["qk_tiles"] += qk_tiles
    stats["pv_tiles"] += pv_tiles
    if qk_quant:
        stats["qk_quantized_tiles"] += qk_tiles
    if pv_quant:
        stats["pv_quantized_tiles"] += pv_tiles
    return output.transpose(1, 2).contiguous(), None
