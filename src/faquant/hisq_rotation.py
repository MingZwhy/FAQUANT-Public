from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .rotation import chunked_block_hadamard_transform


@dataclass(frozen=True)
class HiSQRotation:
    """Deterministic per-linear input rotation used by official HiSQRot4."""

    signs: torch.Tensor
    permutation: torch.Tensor
    block_size: int
    layer_seed: int


def derive_hisq_rotation(
    layer_name: str,
    in_features: int,
    *,
    seed: int = 17,
    block_size: int = 1024,
    device: torch.device | str = "cpu",
) -> HiSQRotation:
    """Reproduce HiSQRot4's sign/permutation/block-Hadamard construction."""

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("HiSQ rotation block_size must be a positive power of two")
    if in_features % block_size:
        raise ValueError(
            f"in_features={in_features} is not divisible by block_size={block_size}"
        )
    digest = hashlib.sha256(f"{seed}::{layer_name}".encode("utf-8")).digest()
    layer_seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(layer_seed)
    signs = torch.randint(0, 2, (in_features,), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(device=device, dtype=torch.int8)
    permutation = torch.randperm(
        in_features, generator=generator, dtype=torch.int64
    ).to(device=device)
    return HiSQRotation(signs, permutation, block_size, layer_seed)


def apply_hisq_rotation(
    tensor: torch.Tensor,
    rotation: HiSQRotation,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply ``diag(sign) @ permutation @ block-Hadamard`` on the last axis."""

    if tensor.shape[-1] != rotation.signs.numel():
        raise ValueError("tensor and HiSQ rotation dimensions do not align")
    work = tensor.float() * rotation.signs.to(
        device=tensor.device, dtype=torch.float32
    )
    work = work.index_select(
        -1, rotation.permutation.to(device=tensor.device)
    )
    work = chunked_block_hadamard_transform(
        work,
        block_size=rotation.block_size,
        chunk_rows=256,
    )
    return work.to(dtype=output_dtype or tensor.dtype)


@torch.no_grad()
def precondition_linear_for_hisq_rotation(
    linear: nn.Linear,
    rotation: HiSQRotation,
) -> torch.Tensor:
    """Fold the input rotation into a Linear weight and persist runtime metadata.

    Returns the FP32 transformed weight so GPTQ can avoid an intermediate BF16
    rounding step.  Runtime applies the same orthogonal transform to the input,
    preserving ``F.linear(x, W)`` before quantization.
    """

    transformed_weight = apply_hisq_rotation(
        linear.weight.detach().float(), rotation, output_dtype=torch.float32
    )
    linear.weight.copy_(transformed_weight.to(linear.weight.dtype))
    linear.faquant_input_rotation_signs = rotation.signs
    linear.faquant_input_rotation_permutation = rotation.permutation
    linear.faquant_input_rotation_block_size = rotation.block_size
    return transformed_weight


def resolve_hisq_block_size(
    layer_name: str,
    in_features: int,
    *,
    block_size: int,
    down_proj_block_size: int | None = None,
) -> int:
    """Pick the HiSQ block for one linear, allowing a wider block on down_proj.

    Every projection but down_proj takes a 4096-wide input, so the default 1024
    block mixes a quarter of it. down_proj takes 12288 and gets a twelfth, which
    is not enough to break up Qwen3's SwiGLU massive activations: its Hessian
    diagonal still spans 71x from max to median against 3-9x elsewhere.

    Widening only this one projection is deliberately not the same thing as the
    global block-4096 setting that was already ablated and lost, because that
    one also promoted the 4096-wide inputs to full width.
    """

    if down_proj_block_size is None or not layer_name.endswith("down_proj"):
        return block_size
    if in_features % down_proj_block_size:
        raise ValueError(
            f"{layer_name} width {in_features} is not divisible by "
            f"down_proj block {down_proj_block_size}"
        )
    return down_proj_block_size


def hisq_rotation_source(
    layer_name: str, sharing: str, *, seed: int, global_seed: int
) -> tuple[str, int]:
    """Name and seed that ``layer_name``'s rotation is derived from under ``sharing``.

    "norm" keys q/k/v by their layer's input_layernorm and gate/up by its
    post_attention_layernorm. "global" returns the key and seed of
    ``apply_qwen3_global_rotation``'s residual rotation, so the online version
    rotates every reader exactly as the folded one does. o_proj and down_proj
    read no residual stream and always keep their own.
    """

    if sharing not in ("linear", "norm", "global"):
        raise ValueError(f"unknown HiSQ rotation sharing {sharing!r}")
    parent, _, projection = layer_name.rpartition(".")
    if sharing == "linear" or projection in ("o_proj", "down_proj"):
        return layer_name, seed
    if sharing == "global":
        return "global_residual", global_seed
    if projection in ("q_proj", "k_proj", "v_proj"):
        norm = "input_layernorm"
    elif projection in ("gate_proj", "up_proj"):
        norm = "post_attention_layernorm"
    else:
        raise ValueError(f"{layer_name} reads no RMSNorm a rotation could be keyed by")
    return f"{parent.rpartition('.')[0]}.{norm}", seed


QWEN_RESIDUAL_READERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def restrict_hisq_rotation(rotation: HiSQRotation, construction: str) -> HiSQRotation:
    """Keep part of the HiSQ construction: "signs" drops the permutation, "plain" the signs too."""

    if construction == "full":
        return rotation
    if construction not in ("signs", "plain"):
        raise ValueError(f"unknown HiSQ construction {construction!r}")
    permutation = torch.arange(
        rotation.permutation.numel(), device=rotation.permutation.device
    )
    signs = rotation.signs if construction == "signs" else torch.ones_like(rotation.signs)
    return HiSQRotation(signs, permutation, rotation.block_size, rotation.layer_seed)


def derive_qwen_hisq_rotation(
    layer_name: str,
    in_features: int,
    *,
    seed: int,
    block_size: int,
    down_proj_block_size: int | None = None,
    sharing: str = "linear",
    global_seed: int = 0,
    reader_construction: str = "full",
    od_construction: str = "full",
    reader_block_size: int | None = None,
    o_construction: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[HiSQRotation, str]:
    """The rotation one Qwen projection takes under a HiSQ variant, and the key it came from.

    ``o_construction`` overrides ``od_construction`` for o_proj alone.
    """

    source, source_seed = hisq_rotation_source(
        layer_name, sharing, seed=seed, global_seed=global_seed
    )
    projection = layer_name.rsplit(".", 1)[-1]
    reader = projection in QWEN_RESIDUAL_READERS
    size = resolve_hisq_block_size(
        layer_name,
        in_features,
        block_size=block_size,
        down_proj_block_size=down_proj_block_size,
    )
    if reader and reader_block_size is not None:
        size = reader_block_size
    rotation = derive_hisq_rotation(
        source, in_features, seed=source_seed, block_size=size, device=device
    )
    if reader:
        construction = reader_construction
    elif projection == "o_proj" and o_construction is not None:
        construction = o_construction
    else:
        construction = od_construction
    return restrict_hisq_rotation(rotation, construction), source


def _iter_qwen_projection_linears(
    model: nn.Module, exempt: frozenset[tuple[int, str]] = frozenset()
):
    for layer_index, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if (layer_index, name) in exempt:
                continue
            yield (
                f"model.layers.{layer_index}.self_attn.{name}",
                getattr(layer.self_attn, name),
            )
        for name in ("gate_proj", "up_proj", "down_proj"):
            if (layer_index, name) in exempt:
                continue
            yield (
                f"model.layers.{layer_index}.mlp.{name}",
                getattr(layer.mlp, name),
            )


@torch.inference_mode()
def apply_qwen_hisq_input_rotation(
    model: nn.Module,
    *,
    block_size: int = 1024,
    down_proj_block_size: int | None = None,
    seed: int = 17,
    exempt_modules: Iterable[tuple[int, str]] = (),
    on_rotated: Callable[[str], None] | None = None,
    projections: Iterable[str] = (),
    sharing: str = "linear",
    global_seed: int = 0,
    reader_construction: str = "full",
    od_construction: str = "full",
    reader_block_size: int | None = None,
    o_construction: str | None = None,
) -> dict[str, object]:
    """Precondition the Qwen projection linears for the RTN HiSQ cell.

    Projections in ``exempt_modules`` are left alone: they stay in BF16, so
    rotating their weights would leave them in a basis nothing undoes. A
    non-empty ``projections`` restricts the rotation to those names; the
    remaining arguments pick the variant (see ``derive_qwen_hisq_rotation``).

    ``on_rotated(layer_name)`` runs right after each projection is rotated.
    Quantizing there consumes the FP32 override before the next projection
    allocates its own; otherwise every override is alive at once, about three
    times the BF16 model.
    """

    exempt = frozenset(exempt_modules)
    selected = frozenset(projections)
    layer_stats: dict[str, dict[str, int]] = {}
    for layer_name, linear in _iter_qwen_projection_linears(model, exempt):
        if selected and layer_name.rsplit(".", 1)[-1] not in selected:
            continue
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Qwen HiSQ expected nn.Linear at {layer_name}")
        rotation, source = derive_qwen_hisq_rotation(
            layer_name,
            linear.in_features,
            seed=seed,
            block_size=block_size,
            down_proj_block_size=down_proj_block_size,
            sharing=sharing,
            global_seed=global_seed,
            reader_construction=reader_construction,
            od_construction=od_construction,
            reader_block_size=reader_block_size,
            o_construction=o_construction,
            device=linear.weight.device,
        )
        transformed_weight = precondition_linear_for_hisq_rotation(
            linear, rotation
        )
        # Preserve the FP32 post-rotation weight until direct HiF4 casting.
        # Copying through the BF16 module parameter would introduce an
        # unrelated intermediate rounding step absent from official Stage 2.
        linear.faquant_weight_override = transformed_weight
        layer_stats[layer_name] = {
            "input_features": linear.in_features,
            "block_size": rotation.block_size,
            "blocks": linear.in_features // rotation.block_size,
            "layer_seed": rotation.layer_seed,
            "source": source,
        }
        del linear, transformed_weight
        if on_rotated is not None:
            on_rotated(layer_name)
    return {
        "source": "GCC-HiFloat/HiSQRot4",
        "source_commit": "84cdcf7b393d2665eedb5cf26e11f590f6852c46",
        "construction": "per-linear sign -> permutation -> block-Hadamard",
        "block_size": block_size,
        "down_proj_block_size": down_proj_block_size,
        "seed": seed,
        "sharing": sharing,
        "reader_construction": reader_construction,
        "od_construction": od_construction,
        "o_construction": o_construction,
        "reader_block_size": reader_block_size,
        "layers": layer_stats,
    }
