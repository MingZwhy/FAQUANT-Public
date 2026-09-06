from __future__ import annotations

import hashlib
from collections.abc import Iterable
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
) -> dict[str, object]:
    """Precondition all Qwen projection linears for the RTN HiSQ cell.

    Projections in ``exempt_modules`` are left alone: they stay in BF16, so
    rotating their weights would leave them in a basis nothing undoes.
    """

    exempt = frozenset(exempt_modules)
    layer_stats: dict[str, dict[str, int]] = {}
    for layer_name, linear in _iter_qwen_projection_linears(model, exempt):
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Qwen HiSQ expected nn.Linear at {layer_name}")
        rotation = derive_hisq_rotation(
            layer_name,
            linear.in_features,
            seed=seed,
            block_size=resolve_hisq_block_size(
                layer_name,
                linear.in_features,
                block_size=block_size,
                down_proj_block_size=down_proj_block_size,
            ),
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
        }
    return {
        "source": "GCC-HiFloat/HiSQRot4",
        "source_commit": "84cdcf7b393d2665eedb5cf26e11f590f6852c46",
        "construction": "per-linear sign -> permutation -> block-Hadamard",
        "block_size": block_size,
        "down_proj_block_size": down_proj_block_size,
        "seed": seed,
        "layers": layer_stats,
    }
