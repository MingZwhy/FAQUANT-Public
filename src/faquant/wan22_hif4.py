"""Plain HiF4 W4A4 on the transformer blocks of the Wan2.2 DiT experts.

Every block Linear of both experts (self/cross-attention q, k, v, o and the two FFN
matrices: 10 per block, 400 per expert, the layers HiSQRot4 quantizes) gets HiF4
round-to-nearest weights and on-the-fly HiF4 inputs, with no smoothing, rotation or
clipping. Both use blocks of 64 along the input axis, so inputs are per token.

- Weights are quantized once from the parameters the BF16 baseline computes with
  (bf16 under ``convert_model_dtype``), or given already quantized (GPTQ,
  ``scripts/calibrate_wan22_gptq.py``). HiF4 values are exact in bf16: a 3-bit
  element times an E6M2 scale and two powers of two has at most 6 significant bits.
- Inputs are quantized as they arrive: fp32 under the bf16 autocast of
  ``WanT2V.generate`` (the block modulation runs in fp32), as HiSQRot4's own
  wrappers do. ``F.linear`` then casts the exact HiF4 values to bf16.
- Rows are processed in chunks only to bound the fp32 temporaries of the quantizer.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .hif4 import fake_quantize_hif4

BLOCK_LINEARS = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)


def faquant_qdq(levels: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda x: fake_quantize_hif4(x, levels=levels)


def quant_cy_qdq(hifloat4_root: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """The CUDA HiFX4 kernel HiSQRot4 uses (official three-level format only)."""

    import sys

    sys.path.insert(0, f"{hifloat4_root}/hifx4_gpu")
    from quant_cy import QType, quant_dequant_float

    qtype = QType("hifx4").dim(-1)
    return lambda x: quant_dequant_float(x, qtype, force_fp32=True)


class HiF4Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        *,
        weight_qdq: Callable[[torch.Tensor], torch.Tensor],
        input_qdq: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str,
        chunk_elements: int = 1 << 26,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        source = linear.weight.detach()
        if weight is None:
            quantized = weight_qdq(source.to(device=device, dtype=torch.float32))
        elif weight.shape != source.shape:
            raise ValueError(f"HiF4 weight of shape {tuple(weight.shape)} for a Linear of {tuple(source.shape)}")
        else:
            quantized = weight.to(device=device, dtype=torch.float32)
        stored = quantized.to(source.dtype)
        if not torch.equal(stored.float(), quantized):
            raise RuntimeError(f"HiF4 weight is not exactly representable in {source.dtype}")
        self.weight = nn.Parameter(stored.to(source.device), requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.input_qdq = input_qdq
        self.chunk_rows = max(1, chunk_elements // self.in_features)
        # called as collector(module, quantized_rows, first_row_index) for every chunk (calibration)
        self.collector: Callable[[nn.Module, torch.Tensor, int], None] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        out = None
        for start in range(0, rows.shape[0], self.chunk_rows):
            part = self.input_qdq(rows[start:start + self.chunk_rows])
            if self.collector is not None:
                self.collector(self, part, start)
            y = F.linear(part, self.weight, self.bias)
            if out is None:
                out = y.new_empty(rows.shape[0], y.shape[-1])
            out[start:start + part.shape[0]] = y
        return out.reshape(*x.shape[:-1], self.out_features)


@torch.no_grad()
def fold_value_head_hadamard(model: nn.Module, device: torch.device | str = "cuda") -> int:
    """Fold a per-head Hadamard through every self-attention's v (rows and bias) and o (columns).

    As in the Qwen recipe: V, and with it the attention output, is rotated per head, and o
    undoes the rotation, so nothing runs online. Exact in high precision (the normalized
    Hadamard is orthogonal and symmetric); computed in fp32, stored in the weights' dtype.
    """
    from .rotation import hadamard_transform

    for block in model.blocks:
        att = block.self_attn
        n, d = att.num_heads, att.head_dim
        wv = att.v.weight.to(device, torch.float32)
        wv = hadamard_transform(wv.reshape(n, d, -1).transpose(1, 2)).transpose(1, 2).reshape(n * d, -1)
        att.v.weight.copy_(wv.to(att.v.weight.dtype))
        if att.v.bias is not None:
            bv = hadamard_transform(att.v.bias.to(device, torch.float32).reshape(n, d)).reshape(-1)
            att.v.bias.copy_(bv.to(att.v.bias.dtype))
        wo = att.o.weight.to(device, torch.float32)
        wo = hadamard_transform(wo.reshape(wo.shape[0], n, d)).reshape(wo.shape[0], -1)
        att.o.weight.copy_(wo.to(att.o.weight.dtype))
    return len(model.blocks)


@torch.no_grad()
def quantize_dit_blocks(
    model: nn.Module,
    *,
    weight_qdq: Callable[[torch.Tensor], torch.Tensor],
    input_qdq: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cuda",
    weights: Mapping[str, torch.Tensor] | None = None,
) -> int:
    """Replace the block Linears of one WanModel in place; returns how many.

    ``weights`` maps every ``blocks.{i}.{path}`` to its HiF4 weight, used instead of RTN."""

    count = 0
    for index, block in enumerate(model.blocks):
        for path in BLOCK_LINEARS:
            parent_name, _, child = path.rpartition(".")
            parent = block.get_submodule(parent_name)
            linear = getattr(parent, child)
            if type(linear) is not nn.Linear:
                raise TypeError(f"{path} is {type(linear).__name__}, not nn.Linear")
            given = None if weights is None else weights[f"blocks.{index}.{path}"]
            setattr(parent, child, HiF4Linear(linear, weight_qdq=weight_qdq, input_qdq=input_qdq, device=device,
                                              weight=given))
            count += 1
    left = [name for name, module in model.named_modules() if name.startswith("blocks.") and type(module) is nn.Linear]
    if left:
        raise RuntimeError(f"block Linears left unquantized: {left[:8]}")
    return count
