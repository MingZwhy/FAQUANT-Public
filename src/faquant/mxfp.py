from __future__ import annotations

import importlib
from functools import lru_cache

import torch


MXFP8_BLOCK_SIZE = 32
MXFP8_FORMAT = "mxfp8e4m3"


@lru_cache(maxsize=1)
def _vendor_quantizer():
    """Load the optional upstream MXFP reference package."""

    try:
        quant_cy = importlib.import_module("quant_cy")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MXFP is not used by the released recipe; install the upstream "
            "HiFloat reference package only for optional MXFP experiments"
        ) from error
    return quant_cy, quant_cy.QType(MXFP8_FORMAT).dim(-1)


@torch.no_grad()
def fake_quantize_mxfp8e4m3(x: torch.Tensor) -> torch.Tensor:
    """QDQ with the pinned MXFP8 E4M3, UE8M0, block-32 implementation."""

    if not x.is_floating_point():
        raise TypeError("MXFP8 fake quantization expects floating point")
    if x.shape[-1] % MXFP8_BLOCK_SIZE:
        raise ValueError("MXFP8 requires the final dimension to be divisible by 32")
    quant_cy, qtype = _vendor_quantizer()
    return quant_cy.quant_dequant_float(
        x.contiguous(),
        qtype,
        force_py=not x.is_cuda,
        force_fp32=not x.is_cuda,
    )
