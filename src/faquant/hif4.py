from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.nn import functional as F


HIF4_BLOCK_SIZE = 64
"""Number of S1P2 values sharing one HiF4 metadata block."""

HIF4_SCALE_LEVELS = os.environ.get("FAQUANT_HIF4_SCALE_LEVELS", "123")
"""Active scale levels: ``123`` is the official format (an E6M2 scale per 64
values, then a 1-bit exponent per 8 and per 4 values). ``12`` drops the per-4
exponent and ``13`` the per-8 one. With a single micro-exponent the largest
multiplier is 2 instead of 4, so the base scale covers the block maximum as
``max / 3.5`` instead of ``max / 7``, and the remaining exponent uses the rule
the official encoder applies to its last level. Read from the environment so
weights, activations and attention operands of a process all use one format."""
HIF4_ACT_SCALE_LEVELS = os.environ.get("FAQUANT_HIF4_ACT_SCALE_LEVELS", HIF4_SCALE_LEVELS)
"""The same choice for direct-cast tensors (linear inputs and attention operands)
when it should differ from the weights, to attribute a variant's loss."""
HIF4_ATTN_SCALE_LEVELS = os.environ.get("FAQUANT_HIF4_ATTN_SCALE_LEVELS", HIF4_ACT_SCALE_LEVELS)
"""The same choice for the attention matmul operands (Q, K, P, V) alone."""
for _name, _value in (
    ("FAQUANT_HIF4_SCALE_LEVELS", HIF4_SCALE_LEVELS),
    ("FAQUANT_HIF4_ACT_SCALE_LEVELS", HIF4_ACT_SCALE_LEVELS),
    ("FAQUANT_HIF4_ATTN_SCALE_LEVELS", HIF4_ATTN_SCALE_LEVELS),
):
    if _value not in ("123", "12", "13"):
        raise ValueError(f"{_name} must be 123, 12 or 13, got {_value!r}")
HIF4_ATTN_OPERAND_SCALE_LEVELS = dict.fromkeys(("q", "k", "p", "v"), HIF4_ATTN_SCALE_LEVELS)
"""Per-operand override of the attention variant, e.g.
``FAQUANT_HIF4_ATTN_OPERAND_LEVELS=p=123,v=123``, to split a variant's attention
loss between Q, K, P and V."""
for _entry in filter(None, os.environ.get("FAQUANT_HIF4_ATTN_OPERAND_LEVELS", "").split(",")):
    _operand, _, _value = _entry.partition("=")
    if _operand not in HIF4_ATTN_OPERAND_SCALE_LEVELS or _value not in ("123", "12", "13"):
        raise ValueError(
            f"FAQUANT_HIF4_ATTN_OPERAND_LEVELS entries must be q|k|p|v=123|12|13, got {_entry!r}"
        )
    HIF4_ATTN_OPERAND_SCALE_LEVELS[_operand] = _value

HIF4_REFERENCE_REPOSITORY = (
    "https://github.com/GCC-HiFloat/HiFloat4-Quantization_Library.git"
)
HIF4_REFERENCE_COMMIT = "6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4"
HIF4_FAKE_QUANT_BACKEND = "faquant_torch_exact_reference"


@dataclass(frozen=True)
class HiF4Parameters:
    """Broadcastable conversion parameters for one or more HiF4 blocks.

    ``quant_multiplier`` contains the BF16-rounded reciprocal of the E6M2
    scale and both micro-exponents. ``dequant_scale`` contains the matching
    forward scale. Keeping the two values separate is intentional: the
    reference conversion rounds the reciprocal to BF16, so replacing the
    multiplication with an ordinary FP32 division changes boundary cases.
    """

    quant_multiplier: torch.Tensor
    dequant_scale: torch.Tensor


@dataclass(frozen=True)
class CompactHiF4Parameters:
    """Compact official HiF4 metadata for tensors grouped in blocks of 64.

    ``scale`` and ``reciprocal`` are stored once per 64-value block. The two
    micro-exponents are powers of two, so their integer exponents are sufficient
    and avoid retaining two full-size expanded tensors during QAD.
    """

    scale: torch.Tensor
    reciprocal: torch.Tensor
    scale_lv2_exponent: torch.Tensor
    scale_lv3_exponent: torch.Tensor


def _require_floating_point(x: torch.Tensor) -> None:
    if not x.is_floating_point():
        raise TypeError("HiF4 fake quantization expects a floating-point tensor")


def _round_to_bfloat16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(x.dtype)


def _e6m2_scale(
    maximum: torch.Tensor, *, divisor: float = 7.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``maximum / divisor`` to the official E6M2 base scale.

    This follows ``quant_cy/base/QFuncs/hifx.py`` from the pinned
    GCC-HiFloat reference implementation, including its BF16 multiply and
    reciprocal rounding stages. The official format divides by 7; the
    two-level variants divide by 3.5.
    """

    reciprocal_divisor = _round_to_bfloat16(torch.ones_like(maximum) / divisor)
    scale = _round_to_bfloat16(maximum * reciprocal_divisor)
    scale = scale.clamp(min=2.0**-48, max=49152.0)

    # Match the reference's explicit BF16 round-to-nearest-even operation
    # before reducing the significand to E6M2.
    exponent = torch.floor(torch.log2(scale))
    bf16_significand = scale * torch.exp2(7.0 - exponent)
    scale = torch.round(bf16_significand) * torch.exp2(exponent - 7.0)

    exponent = torch.floor(torch.log2(scale))
    scale = torch.round(scale * torch.exp2(2.0 - exponent)) * torch.exp2(exponent - 2.0)
    reciprocal = _round_to_bfloat16(1.0 / scale)
    return scale, reciprocal


def _step_e6m2_scale(
    scale: torch.Tensor, steps: torch.Tensor | float
) -> torch.Tensor:
    """Move a legal E6M2 scale by whole steps along the E6M2 grid.

    E6M2 keeps two mantissa bits, so each octave holds exactly four legal
    scales, ``{1.0, 1.25, 1.5, 1.75} * 2^e``. Indexing them as ``4e + m`` turns
    the grid into the integers, and stepping is then addition with the carry
    into the exponent handled for free. Scaling the block maximum by a constant
    instead cannot do this: the ratio between neighbouring grid points varies
    from 1.143 to 1.25 across an octave, so a fixed factor lands on the
    neighbour only for part of the range.
    """

    exponent = torch.floor(torch.log2(scale))
    mantissa = torch.round(scale * torch.exp2(2.0 - exponent))
    index = 4.0 * exponent + (mantissa - 4.0) + steps
    stepped_exponent = torch.floor(index / 4.0)
    stepped_mantissa = index - 4.0 * stepped_exponent + 4.0
    return (stepped_mantissa * torch.exp2(stepped_exponent - 2.0)).clamp(
        min=2.0**-48, max=49152.0
    )


@torch.no_grad()
def fit_hif4_compact_parameters(
    blocks: torch.Tensor,
    *,
    max_scale: torch.Tensor | float = 1.0,
    scale_steps: torch.Tensor | float = 0.0,
    levels: str | None = None,
) -> CompactHiF4Parameters:
    """Fit compact HiF4 metadata for tensors whose final axis is exactly 64.

    ``levels`` defaults to the weight variant, ``HIF4_SCALE_LEVELS``.

    ``max_scale`` shrinks the block maximum the scale is derived from, and
    ``scale_steps`` moves the fitted scale along the E6M2 grid. Both trade
    clipping against step size, but only ``scale_steps`` can go both ways.

    That matters because HiF4 rounds the scale to nearest on the E6M2 grid
    rather than taking a ceiling. Roughly half of Qwen3-8B's blocks therefore
    round *down* and are already clipping before any search runs, and they
    carry about half the baseline squared error; see
    ``scripts/probe_hif4_scale_rounding_direction.py``. Stepping up is not
    usually their best move, since coarsening all 64 values tends to cost more
    than the clipped maximum saves, but for a sizeable minority of them it is,
    and a search offering only ``max_scale < 1`` can never propose it. Both
    arguments default to the no-op value and reproduce the pinned reference
    bit for bit.
    """

    _require_floating_point(blocks)
    if blocks.ndim == 0:
        raise ValueError("HiF4 parameter fitting requires at least one dimension")
    if blocks.shape[-1] != HIF4_BLOCK_SIZE:
        raise ValueError(
            f"HiF4 parameter fitting requires a final dimension of "
            f"{HIF4_BLOCK_SIZE}, got {blocks.shape[-1]}"
        )

    work = blocks.float().reshape(*blocks.shape[:-1], 8, 2, 4)
    magnitude = work.abs()
    maximum_lv3 = magnitude.amax(dim=-1, keepdim=True)
    maximum_lv2 = maximum_lv3.amax(dim=-2, keepdim=True)
    maximum_lv1 = maximum_lv2.amax(dim=-3, keepdim=True)
    if not isinstance(max_scale, torch.Tensor):
        max_scale = torch.as_tensor(
            max_scale, device=maximum_lv1.device, dtype=maximum_lv1.dtype
        )
    if bool((max_scale != 1.0).any()):
        maximum_lv1 = maximum_lv1 * max_scale

    levels = levels or HIF4_SCALE_LEVELS
    scale, reciprocal = _e6m2_scale(maximum_lv1, divisor=7.0 if levels == "123" else 3.5)
    if not isinstance(scale_steps, torch.Tensor):
        scale_steps = torch.as_tensor(
            scale_steps, device=scale.device, dtype=scale.dtype
        )
    if bool((scale_steps != 0.0).any()):
        scale = _step_e6m2_scale(scale, scale_steps)
        reciprocal = _round_to_bfloat16(1.0 / scale)
    if levels == "123":
        scale_lv2_exponent = torch.floor(
            (maximum_lv2 * reciprocal).clamp(0.0, 4.0) / 4.0
        )
        scale_lv2 = torch.exp2(scale_lv2_exponent)
        scale_lv3_exponent = torch.floor(
            (maximum_lv3 * reciprocal / scale_lv2).clamp(0.0, 2.0) / 2.0
        )
    elif levels == "12":
        scale_lv2_exponent = torch.floor((maximum_lv2 * reciprocal).clamp(0.0, 2.0) / 2.0)
        scale_lv3_exponent = torch.zeros_like(maximum_lv3)
    else:
        scale_lv2_exponent = torch.zeros_like(maximum_lv2)
        scale_lv3_exponent = torch.floor((maximum_lv3 * reciprocal).clamp(0.0, 2.0) / 2.0)
    return CompactHiF4Parameters(
        scale=scale,
        reciprocal=reciprocal,
        scale_lv2_exponent=scale_lv2_exponent.to(torch.int8),
        scale_lv3_exponent=scale_lv3_exponent.to(torch.int8),
    )


def expand_hif4_parameters(
    compact: CompactHiF4Parameters,
    *,
    block_shape: torch.Size | tuple[int, ...],
) -> HiF4Parameters:
    """Expand compact metadata to the per-value representation used by GPTQ."""

    block_shape = tuple(block_shape)
    if not block_shape or block_shape[-1] != HIF4_BLOCK_SIZE:
        raise ValueError("HiF4 block_shape must end in 64")
    work_shape = (*block_shape[:-1], 8, 2, 4)
    scale_lv2 = torch.exp2(compact.scale_lv2_exponent.float())
    scale_lv3 = torch.exp2(compact.scale_lv3_exponent.float())

    quant_multiplier = (
        compact.reciprocal / scale_lv2 / scale_lv3
    ).expand(work_shape)
    dequant_scale = (compact.scale * scale_lv2 * scale_lv3).expand(work_shape)
    return HiF4Parameters(
        quant_multiplier=quant_multiplier.reshape(block_shape),
        dequant_scale=dequant_scale.reshape(block_shape),
    )


@torch.no_grad()
def fit_hif4_parameters(blocks: torch.Tensor, *, levels: str | None = None) -> HiF4Parameters:
    """Fit expanded HiF4 metadata for backward-compatible GPTQ callers."""

    compact = fit_hif4_compact_parameters(blocks, levels=levels)
    return expand_hif4_parameters(compact, block_shape=blocks.shape)


@torch.no_grad()
def quantize_hif4_with_parameters(
    x: torch.Tensor,
    parameters: HiF4Parameters,
) -> torch.Tensor:
    """Quantize/dequantize values with already-fitted HiF4 metadata."""

    _require_floating_point(x)
    if (
        x.shape != parameters.quant_multiplier.shape
        or x.shape != parameters.dequant_scale.shape
    ):
        raise ValueError("HiF4 values and fitted parameters must have matching shapes")
    work = x.float()
    mantissa = torch.floor(work.abs() * parameters.quant_multiplier * 4.0 + 0.5) / 4.0
    mantissa.clamp_(max=1.75)
    output = torch.sign(work) * mantissa * parameters.dequant_scale
    return output.to(dtype=x.dtype)


@torch.no_grad()
def quantize_hif4_with_compact_parameters(
    blocks: torch.Tensor,
    parameters: CompactHiF4Parameters,
) -> torch.Tensor:
    """Quantize 64-value blocks using fixed compact HiF4 metadata."""

    expanded = expand_hif4_parameters(parameters, block_shape=blocks.shape)
    return quantize_hif4_with_parameters(blocks, expanded)


@torch.no_grad()
def project_hif4_source_to_target_cells(
    source: torch.Tensor,
    target: torch.Tensor,
    parameters: CompactHiF4Parameters,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Project a floating source into the fixed HiF4 cells of ``target``.

    GPTQ returns deployed dequantized values, while QAT benefits from keeping a
    latent floating-point master.  This projection retains as much of
    ``source`` as possible without changing any code produced by the fixed
    GPTQ metadata.  It therefore preserves the exact step-0 deployed model.
    """

    if source.shape != target.shape or source.ndim < 1:
        raise ValueError("HiF4 source and target must have the same non-scalar shape")
    if source.shape[-1] != HIF4_BLOCK_SIZE:
        raise ValueError("HiF4 cell projection requires a final dimension of 64")
    expanded = expand_hif4_parameters(parameters, block_shape=source.shape)
    multiplier = expanded.quant_multiplier
    dequant_scale = expanded.dequant_scale

    # Work in the normalized magnitude consumed by the official round rule:
    # floor(abs(x) * multiplier * 4 + 0.5) / 4, saturated at 1.75.
    code_index = torch.round(target.float().abs() / dequant_scale * 4.0).clamp_(0, 7)
    source_normalized = source.float() * multiplier
    source_sign = torch.sign(target.float())
    magnitude = source_normalized.abs()
    lower = (code_index - 0.5).clamp_min_(0.0) / 4.0
    upper = (code_index + 0.5) / 4.0

    # Stay one FP32 representable value inside decision boundaries.  Saturated
    # code 7 has no finite upper boundary.
    lower_inside = torch.nextafter(lower, torch.full_like(lower, float("inf")))
    upper_inside = torch.nextafter(upper, torch.full_like(upper, float("-inf")))
    projected_magnitude = torch.maximum(magnitude, lower_inside)
    projected_magnitude = torch.where(
        code_index.eq(7),
        projected_magnitude,
        torch.minimum(projected_magnitude, upper_inside),
    )
    projected = source_sign * projected_magnitude / multiplier
    zero_projected = source_normalized.clamp(
        min=-upper_inside,
        max=upper_inside,
    ) / multiplier
    projected = torch.where(code_index.eq(0), zero_projected, projected)

    # A later BF16 cast can move boundary values. Fall back only those values
    # to their deployed target, retaining the latent residual everywhere else.
    projected = projected.to(source.dtype if output_dtype is None else output_dtype)
    actual = quantize_hif4_with_parameters(projected, expanded)
    mismatch = actual.ne(target.to(actual.dtype))
    if torch.any(mismatch):
        projected = torch.where(mismatch, target.to(projected.dtype), projected)
    return projected


@torch.no_grad()
def _group_for_hif4(x: torch.Tensor, *, what: str) -> tuple[torch.Tensor, int]:
    """Zero-pad the final axis to a multiple of 64 and split it into blocks."""

    _require_floating_point(x)
    if x.ndim == 0:
        raise ValueError(f"HiF4 {what} requires at least one dimension")
    original_size = x.shape[-1]
    if original_size == 0:
        raise ValueError(f"HiF4 {what} does not support an empty axis")
    padding = (-original_size) % HIF4_BLOCK_SIZE
    padded = F.pad(x, (0, padding)) if padding else x
    return padded.reshape(*padded.shape[:-1], -1, HIF4_BLOCK_SIZE), original_size


@torch.no_grad()
def fit_hif4_compact_parameters_for_axis(x: torch.Tensor) -> CompactHiF4Parameters:
    """Compact metadata for the grid ``fake_quantize_hif4`` would snap ``x`` to.

    ``fit_hif4_compact_parameters`` needs a final axis of exactly 64, which the
    GPTQ path can guarantee because it has already reshaped the weight into
    blocks.  A plain RTN weight arrives at whatever width the layer has, so the
    padding has to be repeated here; describing the unpadded tensor would yield
    metadata for a different grid from the one the weights were snapped to, and
    QAT step-0 parity would fail by exactly that discrepancy.
    """

    grouped, _ = _group_for_hif4(x, what="parameter fitting")
    return fit_hif4_compact_parameters(grouped)


@torch.no_grad()
def fake_quantize_hif4(x: torch.Tensor, *, levels: str | None = None) -> torch.Tensor:
    """Official HiF4 direct-cast fake quantization along the final axis.

    HiF4 always uses groups of 64. A short final group is zero-padded exactly
    like the upstream ``quant_dequant_float`` helper and sliced back afterward.
    ``levels`` defaults to the direct-cast variant, ``HIF4_ACT_SCALE_LEVELS``.
    """

    grouped, original_size = _group_for_hif4(x, what="fake quantization")
    parameters = fit_hif4_parameters(grouped, levels=levels or HIF4_ACT_SCALE_LEVELS)
    output = quantize_hif4_with_parameters(grouped, parameters).reshape(
        *grouped.shape[:-2], -1
    )
    return output[..., :original_size]
