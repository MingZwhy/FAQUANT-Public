from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the final axis."""

    size = x.shape[-1]
    if not _is_power_of_two(size):
        raise ValueError(f"Hadamard dimension must be a power of two, got {size}")
    original_shape = x.shape
    output = x.reshape(-1, size)
    stride = 1
    while stride < size:
        output = output.reshape(-1, size // (2 * stride), 2, stride)
        left = output[:, :, 0, :]
        right = output[:, :, 1, :]
        output = torch.cat((left + right, left - right), dim=2).reshape(-1, size)
        stride *= 2
    return (output / math.sqrt(size)).reshape(original_shape)


def _hadamard12(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
            [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
            [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
            [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
            [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1],
            [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
            [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
            [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1],
            [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1],
        ],
        device=device,
        dtype=dtype,
    )


def generalized_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Hadamard transform for power-of-two or 12*power-of-two axes."""

    size = x.shape[-1]
    if _is_power_of_two(size):
        return hadamard_transform(x)
    if size % 12 or not _is_power_of_two(size // 12):
        raise ValueError(f"unsupported generalized Hadamard dimension: {size}")
    original_shape = x.shape
    work = x.reshape(-1, size, 1)
    while work.shape[1] > 12:
        work = work.reshape(work.shape[0], work.shape[1] // 2, 2, work.shape[2])
        left = work[:, :, 0, :]
        right = work[:, :, 1, :]
        work = torch.cat((left + right, left - right), dim=2)
    work = _hadamard12(work.device, work.dtype).unsqueeze(0) @ work
    return (work.reshape(original_shape) / math.sqrt(size)).to(x.dtype)


def chunked_generalized_hadamard_transform(
    x: torch.Tensor, *, chunk_rows: int = 256
) -> torch.Tensor:
    """Apply the generalized transform with bounded FP32 working memory."""

    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    output = torch.empty_like(flat)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        transformed = generalized_hadamard_transform(flat[start:stop].float())
        output[start:stop].copy_(transformed.to(x.dtype))
    return output.reshape(shape)


class OnlineHadamardLinear(nn.Module):
    """Apply an exact online rotation immediately before a linear layer."""

    faquant_online_hadamard = True

    def __init__(self, module: nn.Linear, *, block_size: int | None = None) -> None:
        super().__init__()
        self.module = module
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.faquant_online_hadamard_block_size = block_size

    @property
    def weight(self) -> nn.Parameter:
        return self.module.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.module.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.faquant_online_hadamard_block_size is None:
            rotated = chunked_generalized_hadamard_transform(x)
        else:
            rotated = chunked_block_hadamard_transform(
                x, block_size=self.faquant_online_hadamard_block_size
            )
        return self.module(rotated)


def chunked_block_hadamard_transform(
    x: torch.Tensor, *, block_size: int, chunk_rows: int = 256
) -> torch.Tensor:
    """Apply an independent normalized Hadamard transform to feature blocks."""

    if not _is_power_of_two(block_size) or x.shape[-1] % block_size:
        raise ValueError(
            f"dimension {x.shape[-1]} must be divisible by power-of-two "
            f"block_size={block_size}"
        )
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    output = torch.empty_like(flat)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        work = flat[start:stop].float().reshape(-1, block_size)
        transformed = hadamard_transform(work).reshape(stop - start, shape[-1])
        output[start:stop].copy_(transformed.to(x.dtype))
    return output.reshape(shape)


def _right_random_hadamard(
    tensor: torch.Tensor, signs: torch.Tensor, *, chunk_rows: int = 1024
) -> torch.Tensor:
    """Compute tensor @ (diag(signs) @ H) in bounded working memory."""

    result = torch.empty_like(tensor)
    signs = signs.to(device=tensor.device, dtype=torch.float32)
    flat = tensor.reshape(-1, tensor.shape[-1])
    out = result.reshape_as(flat)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        work = flat[start:stop].float() * signs
        out[start:stop].copy_(hadamard_transform(work).to(tensor.dtype))
    return result


def rotate_qwen3_hidden_state(
    hidden_state: torch.Tensor,
    signs: torch.Tensor,
    *,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Map an unrotated Qwen residual hidden state into the rotated basis."""

    if hidden_state.shape[-1] != signs.numel():
        raise ValueError("hidden state and Qwen rotation signs must align")
    return _right_random_hadamard(
        hidden_state,
        signs,
        chunk_rows=chunk_rows,
    )


def _right_random_hadamard_transpose(
    tensor: torch.Tensor, signs: torch.Tensor, *, chunk_rows: int = 1024
) -> torch.Tensor:
    """Compute tensor @ (diag(signs) @ H).T in bounded working memory."""

    result = torch.empty_like(tensor)
    signs = signs.to(device=tensor.device, dtype=torch.float32)
    flat = tensor.reshape(-1, tensor.shape[-1])
    out = result.reshape_as(flat)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        work = hadamard_transform(flat[start:stop].float()) * signs
        out[start:stop].copy_(work.to(tensor.dtype))
    return result


def _left_random_hadamard_transpose(
    tensor: torch.Tensor, signs: torch.Tensor, *, chunk_rows: int = 1024
) -> torch.Tensor:
    """Compute (diag(signs) @ H).T @ tensor."""

    transformed = _right_random_hadamard(tensor.T.contiguous(), signs, chunk_rows=chunk_rows)
    return transformed.T.contiguous()


def _right_hadamard_blocks(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    shape = weight.shape
    if shape[-1] % block_size:
        raise ValueError(f"dimension {shape[-1]} is not divisible by {block_size}")
    work = weight.float().reshape(-1, block_size)
    return hadamard_transform(work).reshape(shape).to(weight.dtype)


def _right_generalized_hadamard(weight: torch.Tensor, *, chunk_rows: int = 1024) -> torch.Tensor:
    result = torch.empty_like(weight)
    for start in range(0, weight.shape[0], chunk_rows):
        stop = min(start + chunk_rows, weight.shape[0])
        transformed = generalized_hadamard_transform(weight[start:stop].float())
        result[start:stop].copy_(transformed.to(weight.dtype))
    return result


def _left_hadamard_blocks(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    return _right_hadamard_blocks(weight.T.contiguous(), block_size).T.contiguous()


def _fuse_norm(norm: nn.Module, linears: tuple[nn.Linear, ...]) -> None:
    gamma = norm.weight.detach()
    for linear in linears:
        linear.weight.data.mul_(gamma.to(linear.weight.device, linear.weight.dtype))
    norm.weight.data.fill_(1)


@torch.inference_mode()
def apply_qwen3_value_head_rotation(
    model: nn.Module,
    *,
    exempt_modules: Iterable[tuple[int, str]] = (),
    block_sizes: dict[int, int] | None = None,
) -> dict[str, object]:
    """Fold a per-head Hadamard through V and the attention output.

    ``apply_qwen3_global_rotation`` already contains this pair, but the HiSQ
    recipe rotates each Linear's *input* instead and therefore leaves the
    head-internal geometry of V, and of the attention output feeding o_proj,
    untouched. Both are HiF4 operands there, so both keep their raw channel
    outliers.

    The two folds cancel exactly around FlashAttention: rotating V's output
    channels rotates the attention output identically, and o_proj's weight
    undoes it. Nothing runs online, and the transform composes with a HiSQ
    input rotation on o_proj because each is separately exact.
    """

    head_dim = model.config.head_dim
    if not _is_power_of_two(head_dim):
        raise ValueError(f"Qwen3 head_dim must be a power of two, got {head_dim}")
    # The fold is a matched pair: it rotates v_proj's output channels and undoes
    # it in o_proj's columns. Exempting one without the other would leave the
    # layer's arithmetic unbalanced, so refuse rather than silently miscompute.
    exempt = frozenset(exempt_modules)
    skipped = sorted(
        index
        for index in range(len(model.model.layers))
        if (index, "v_proj") in exempt or (index, "o_proj") in exempt
    )
    unpaired = [
        index
        for index in skipped
        if ((index, "v_proj") in exempt) != ((index, "o_proj") in exempt)
    ]
    if unpaired:
        raise ValueError(
            "the value-head fold pairs v_proj with o_proj, so layers "
            f"{unpaired} must exempt both or neither"
        )
    rotated = 0
    used_block_sizes: dict[str, int] = {}
    for index, layer in enumerate(model.model.layers):
        if index in skipped:
            continue
        block_size = (
            head_dim if block_sizes is None else block_sizes.get(index, head_dim)
        )
        if not _is_power_of_two(block_size) or head_dim % block_size:
            raise ValueError(
                f"value-head block size {block_size} must divide head_dim={head_dim}"
            )
        rotated += 1
        used_block_sizes[str(index)] = block_size
        attention = layer.self_attn
        if attention.v_proj.bias is not None:
            # A bias would leave the fold inexact: it is added after the
            # rotated projection but is not itself rotated.
            raise ValueError("Qwen3 v_proj is expected to have no bias")
        attention.v_proj.weight.data.copy_(
            _left_hadamard_blocks(attention.v_proj.weight.data, block_size)
        )
        attention.o_proj.weight.data.copy_(
            _right_hadamard_blocks(attention.o_proj.weight.data, block_size)
        )
    return {
        "construction": "per-head Hadamard folded into v_proj rows and o_proj columns",
        "head_dim": head_dim,
        "layers": rotated,
        "skipped_layers": skipped,
        "block_sizes": used_block_sizes,
        "online_cost": "none",
    }


@torch.inference_mode()
def apply_qwen3_global_rotation(
    model: nn.Module, *, seed: int = 0, online_hadamard: bool = True
) -> torch.Tensor:
    """Bake QuaRot-style residual and per-head Hadamard rotations into Qwen3.

    The residual-stream rotation is randomized and global. Value/output head
    rotations are deterministic. Q/K online rotation is installed separately by
    the Qwen3 attention adapter because it must happen after RoPE.
    """

    hidden_size = model.config.hidden_size
    head_dim = model.config.head_dim
    if not _is_power_of_two(hidden_size) or not _is_power_of_two(head_dim):
        raise ValueError("Qwen3 hidden_size and head_dim must be powers of two")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (hidden_size,), generator=generator, dtype=torch.int8)
    signs = signs.mul(2).sub(1)

    # Fold learned RMSNorm scales into their consumers before rotating.
    for layer in model.model.layers:
        _fuse_norm(
            layer.input_layernorm,
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
        )
        _fuse_norm(
            layer.post_attention_layernorm,
            (layer.mlp.gate_proj, layer.mlp.up_proj),
        )
    _fuse_norm(model.model.norm, (model.lm_head,))

    model.model.embed_tokens.weight.data.copy_(
        _right_random_hadamard(model.model.embed_tokens.weight.data, signs)
    )
    model.lm_head.weight.data.copy_(
        _right_random_hadamard(model.lm_head.weight.data, signs)
    )

    for layer in model.model.layers:
        for linear in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
        ):
            linear.weight.data.copy_(_right_random_hadamard(linear.weight.data, signs))

        for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
            linear.weight.data.copy_(
                _left_random_hadamard_transpose(linear.weight.data, signs)
            )
            if linear.bias is not None:
                bias = linear.bias.data.reshape(-1, 1)
                linear.bias.data.copy_(
                    _left_random_hadamard_transpose(bias, signs).reshape(-1)
                )

        # H rotations on V and the attention output cancel exactly around FA.
        layer.self_attn.v_proj.weight.data.copy_(
            _left_hadamard_blocks(layer.self_attn.v_proj.weight.data, head_dim)
        )
        layer.self_attn.o_proj.weight.data.copy_(
            _right_hadamard_blocks(layer.self_attn.o_proj.weight.data, head_dim)
        )

        if online_hadamard:
            # Rotate the large post-SiLU intermediate activation immediately
            # before down_proj. Qwen3-8B uses 12288 = 12 * 1024.
            layer.mlp.down_proj.weight.data.copy_(
                _right_generalized_hadamard(layer.mlp.down_proj.weight.data)
            )
            layer.mlp.down_proj = OnlineHadamardLinear(layer.mlp.down_proj)

    model.register_buffer("faquant_rotation_signs", signs, persistent=True)
    return signs
