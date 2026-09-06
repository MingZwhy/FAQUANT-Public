from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import resolve_quant_exempt_modules
from .hif4 import (
    HIF4_BLOCK_SIZE,
    fake_quantize_hif4,
    fit_hif4_compact_parameters_for_axis,
)
from .mxfp import MXFP8_BLOCK_SIZE, fake_quantize_mxfp8e4m3


def _group_last_dim(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(x.shape)
    if group_size == -1:
        group_size = shape[-1]
    if group_size <= 0 or shape[-1] % group_size:
        raise ValueError(
            f"last dimension {shape[-1]} must be divisible by group_size={group_size}"
        )
    return x.reshape(*shape[:-1], shape[-1] // group_size, group_size), shape


def fake_quantize(
    x: torch.Tensor,
    *,
    quant_format: str = "int",
    bits: int = 4,
    group_size: int = -1,
    symmetric: bool = True,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Quantize then dequantize groups on the last axis.

    Activations are therefore per-token/per-group, while a 2-D weight tensor is
    per-output-channel/per-group. The returned tensor keeps the original dtype.
    """

    if quant_format == "hif4":
        if bits != 4:
            raise ValueError("HiF4 requires bits=4")
        if group_size not in (-1, HIF4_BLOCK_SIZE):
            raise ValueError("HiF4 requires group_size=64 (or -1 for direct use)")
        if not symmetric:
            raise ValueError("HiF4 does not support asymmetric quantization")
        if clip_ratio != 1.0:
            raise ValueError("HiF4 direct-cast does not support clipping")
        return fake_quantize_hif4(x)
    if quant_format == "mxfp8e4m3":
        if bits != 8:
            raise ValueError("MXFP8 E4M3 requires bits=8")
        if group_size != MXFP8_BLOCK_SIZE:
            raise ValueError("MXFP8 E4M3 requires group_size=32")
        if not symmetric or clip_ratio != 1.0:
            raise ValueError("MXFP8 E4M3 does not support asymmetric/clipped QDQ")
        return fake_quantize_mxfp8e4m3(x)
    if quant_format != "int":
        raise ValueError(f"unsupported quant_format={quant_format!r}")
    if bits >= 16:
        return x
    if not x.is_floating_point():
        raise TypeError("fake_quantize expects a floating-point tensor")
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be in (0, 1]")

    grouped, shape = _group_last_dim(x, group_size)
    work = grouped.float()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        absmax = work.abs().amax(dim=-1, keepdim=True) * clip_ratio
        scale = (absmax / qmax).clamp_min(torch.finfo(torch.float32).eps)
        quant = torch.round(work / scale).clamp_(qmin, qmax)
        output = quant * scale
    else:
        qmin, qmax = 0, 2**bits - 1
        xmin = work.amin(dim=-1, keepdim=True) * clip_ratio
        xmax = work.amax(dim=-1, keepdim=True) * clip_ratio
        xmin = torch.minimum(xmin, torch.zeros_like(xmin))
        xmax = torch.maximum(xmax, torch.zeros_like(xmax))
        scale = ((xmax - xmin) / qmax).clamp_min(torch.finfo(torch.float32).eps)
        zero = torch.round(-xmin / scale).clamp_(qmin, qmax)
        quant = torch.round(work / scale + zero).clamp_(qmin, qmax)
        output = (quant - zero) * scale
    return output.reshape(shape).to(dtype=x.dtype)


class FakeQuantLinear(nn.Module):
    """Linear layer with dynamic input and precomputed fake-quantized weights."""

    faquant_quantizes_input = True

    def __init__(
        self,
        linear: nn.Linear,
        *,
        quant_format: str = "int",
        bits: int,
        weight_group_size: int,
        activation_group_size: int,
        symmetric: bool,
        clip_ratio: float,
        quantize_weight: bool = True,
        online_hadamard: bool = False,
        online_hadamard_block_size: int | None = None,
        input_scale: torch.Tensor | None = None,
        input_rotation_signs: torch.Tensor | None = None,
        input_rotation_permutation: torch.Tensor | None = None,
        input_rotation_block_size: int | None = None,
        weight_override: torch.Tensor | None = None,
        activation_min: float | None = None,
        activation_max: float | None = None,
        activation_compand_log_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quant_format = quant_format
        self.bits = bits
        self.activation_group_size = activation_group_size
        self.symmetric = symmetric
        self.clip_ratio = clip_ratio
        self.online_hadamard = online_hadamard
        self.online_hadamard_block_size = online_hadamard_block_size
        self.register_buffer(
            "input_scale",
            None if input_scale is None else input_scale.detach().float().clone(),
        )
        self.register_buffer(
            "input_rotation_signs",
            None
            if input_rotation_signs is None
            else input_rotation_signs.detach().to(torch.int8).clone(),
        )
        self.register_buffer(
            "input_rotation_permutation",
            None
            if input_rotation_permutation is None
            else input_rotation_permutation.detach().long().clone(),
        )
        self.input_rotation_block_size = input_rotation_block_size
        self.register_buffer(
            "activation_min",
            None
            if activation_min is None
            else torch.as_tensor(activation_min).detach().float().clone(),
        )
        self.register_buffer(
            "activation_max",
            None
            if activation_max is None
            else torch.as_tensor(activation_max).detach().float().clone(),
        )
        self.register_buffer(
            "activation_compand_log_scale",
            None
            if activation_compand_log_scale is None
            else activation_compand_log_scale.detach().float().clone(),
        )
        weight = (
            linear.weight.detach()
            if weight_override is None
            else weight_override.detach()
        )
        if quantize_weight:
            weight = fake_quantize(
                weight,
                quant_format=quant_format,
                bits=bits,
                group_size=weight_group_size,
                symmetric=symmetric,
                clip_ratio=clip_ratio,
            )
            weight = weight.to(dtype=linear.weight.dtype)
        # Quantizer setup runs under ``torch.inference_mode``. Materialize
        # ordinary tensors so an offloaded module can move CPU -> CUDA again;
        # inference tensors have no version counter and fail in F.linear after
        # that second device transition.
        with torch.inference_mode(False):
            weight = weight.detach().clone()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            with torch.inference_mode(False):
                bias = linear.bias.detach().clone()
            self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            x = x * self.input_scale.to(device=x.device, dtype=x.dtype)
        if self.input_rotation_signs is not None:
            from .rotation import chunked_block_hadamard_transform

            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            rotated = torch.empty_like(flat)
            signs = self.input_rotation_signs.to(device=x.device, dtype=torch.float32)
            permutation = self.input_rotation_permutation.to(device=x.device)
            for start in range(0, flat.shape[0], 256):
                stop = min(start + 256, flat.shape[0])
                work = flat[start:stop].float() * signs
                work = work.index_select(-1, permutation)
                if self.input_rotation_block_size is not None:
                    work = chunked_block_hadamard_transform(
                        work,
                        block_size=self.input_rotation_block_size,
                        chunk_rows=256,
                    )
                rotated[start:stop].copy_(work.to(x.dtype))
            x = rotated.reshape(shape)
        if self.activation_min is not None or self.activation_max is not None:
            x = x.clamp(min=self.activation_min, max=self.activation_max)
        compand_scale = None
        if self.activation_compand_log_scale is not None:
            log_scale = self.activation_compand_log_scale
            if log_scale.numel() != self.in_features:
                if self.in_features % log_scale.numel():
                    raise RuntimeError("invalid activation companding shape")
                log_scale = log_scale.repeat_interleave(
                    self.in_features // log_scale.numel()
                )
            compand_scale = log_scale.clamp(-2.079, 2.079).exp()
            compand_scale = compand_scale.to(device=x.device, dtype=x.dtype)
            x = x * compand_scale
        if self.online_hadamard:
            from .rotation import (
                chunked_block_hadamard_transform,
                generalized_hadamard_transform,
            )

            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            quantized = torch.empty_like(flat)
            for start in range(0, flat.shape[0], 256):
                stop = min(start + 256, flat.shape[0])
                if self.online_hadamard_block_size is None:
                    rotated = generalized_hadamard_transform(flat[start:stop].float())
                else:
                    rotated = chunked_block_hadamard_transform(
                        flat[start:stop].float(),
                        block_size=self.online_hadamard_block_size,
                        chunk_rows=256,
                    )
                quantized[start:stop].copy_(
                    fake_quantize(
                        rotated.to(x.dtype),
                        quant_format=self.quant_format,
                        bits=self.bits,
                        group_size=self.activation_group_size,
                        symmetric=self.symmetric,
                        clip_ratio=self.clip_ratio,
                    )
                )
            x = quantized.reshape(shape)
        else:
            x = fake_quantize(
                x,
                quant_format=self.quant_format,
                bits=self.bits,
                group_size=self.activation_group_size,
                symmetric=self.symmetric,
                clip_ratio=self.clip_ratio,
            )
        if compand_scale is not None:
            x = x / compand_scale
        return F.linear(x, self.weight, self.bias)


def replace_linear_with_fake_quant(
    parent: nn.Module, name: str, config: object, *, weights_prequantized: bool
) -> None:
    linear = getattr(parent, name)
    if isinstance(linear, FakeQuantLinear):
        return
    online_hadamard = bool(getattr(linear, "faquant_online_hadamard", False))
    online_hadamard_block_size = getattr(
        linear, "faquant_online_hadamard_block_size", None
    )
    if online_hadamard:
        linear = linear.module
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"{parent.__class__.__name__}.{name} is not nn.Linear")
    input_scale = getattr(linear, "faquant_input_scale", None)
    input_rotation_signs = getattr(linear, "faquant_input_rotation_signs", None)
    input_rotation_permutation = getattr(
        linear, "faquant_input_rotation_permutation", None
    )
    input_rotation_block_size = getattr(
        linear, "faquant_input_rotation_block_size", None
    )
    weight_override = getattr(linear, "faquant_weight_override", None)
    activation_min = getattr(linear, "faquant_activation_min", None)
    activation_max = getattr(linear, "faquant_activation_max", None)
    activation_compand_log_scale = getattr(
        linear, "faquant_activation_compand_log_scale", None
    )
    # GPTQ records the HiF4 grid it fitted so that QAT can later re-quantize a
    # drifting latent weight against exactly the grid the stored weight already
    # sits on.  RTN fits the same grid inside fake_quantize_hif4 and discards
    # it, which is why HiF4QATLinear refuses a non-GPTQ student.  Fitting it
    # here, from the same tensor and with the same padding the quantizer uses,
    # makes an RTN model a usable QAD starting point without touching either
    # quantizer.
    if (
        not weights_prequantized
        and config.quant_format == "hif4"
        and getattr(config, "qad_capture_weight_metadata", False)
        and getattr(linear, "faquant_hif4_qat_parameters", None) is None
    ):
        source = weight_override if weight_override is not None else linear.weight
        linear.faquant_hif4_qat_parameters = fit_hif4_compact_parameters_for_axis(
            source.detach()
        )
    replacement = FakeQuantLinear(
            linear,
            quant_format=config.quant_format,
            bits=config.bits,
            weight_group_size=config.weight_group_size,
            activation_group_size=config.activation_group_size,
            symmetric=config.symmetric,
            clip_ratio=config.clip_ratio,
            quantize_weight=not weights_prequantized,
            online_hadamard=online_hadamard,
            online_hadamard_block_size=online_hadamard_block_size,
            input_scale=input_scale,
            input_rotation_signs=input_rotation_signs,
            input_rotation_permutation=input_rotation_permutation,
            input_rotation_block_size=input_rotation_block_size,
            weight_override=None if weights_prequantized else weight_override,
            activation_min=activation_min,
            activation_max=activation_max,
            activation_compand_log_scale=activation_compand_log_scale,
    )
    hif4_qat_parameters = getattr(
        linear, "faquant_hif4_qat_parameters", None
    )
    if hif4_qat_parameters is not None:
        replacement.faquant_hif4_qat_parameters = hif4_qat_parameters
    hif4_qat_master = getattr(linear, "faquant_hif4_qat_master", None)
    if hif4_qat_master is not None:
        replacement.faquant_hif4_qat_master = hif4_qat_master
    setattr(parent, name, replacement)


@torch.inference_mode()
def quantize_qwen3_linears(
    model: nn.Module, config: object, *, weights_prequantized: bool = False
) -> None:
    """Install fake W/A quantization on selected Qwen3 projection matrices."""

    target = config.quant_target
    if target == "none":
        return
    exempt = resolve_quant_exempt_modules(config, len(model.model.layers))
    for index, layer in enumerate(model.model.layers):
        if target in ("attention", "all"):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if (index, name) in exempt:
                    continue
                replace_linear_with_fake_quant(
                    layer.self_attn,
                    name,
                    config,
                    weights_prequantized=weights_prequantized,
                )
        if target in ("ffn", "all"):
            for name in ("gate_proj", "up_proj", "down_proj"):
                if (index, name) in exempt:
                    continue
                replace_linear_with_fake_quant(
                    layer.mlp,
                    name,
                    config,
                    weights_prequantized=weights_prequantized,
                )
