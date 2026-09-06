from __future__ import annotations

import torch
from torch import nn


def smoothquant_input_scale(
    weight: torch.Tensor,
    activation_absmax: torch.Tensor,
    *,
    alpha: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return the per-input-channel scale used by GCC HiSQRot4.

    The paired exact transform is ``x' = x * scale`` and
    ``W' = W / scale``.
    """

    if weight.ndim != 2:
        raise ValueError("SmoothQuant weight must be a matrix")
    activation_absmax = activation_absmax.detach().float().reshape(-1)
    if activation_absmax.numel() != weight.shape[1]:
        raise ValueError(
            "SmoothQuant activation channels must match linear.in_features"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be in [0, 1]")
    if eps <= 0.0:
        raise ValueError("SmoothQuant eps must be positive")
    if not torch.isfinite(activation_absmax).all():
        raise ValueError("SmoothQuant activation statistics must be finite")

    activation_mask = activation_absmax.to(weight.device).clamp_min(eps)
    weight_mask = weight.detach().float().abs().amax(dim=0).clamp_min(eps)
    scale = (
        weight_mask.pow(alpha) / activation_mask.pow(1.0 - alpha)
    ).clamp_min(eps)
    if not torch.isfinite(scale).all():
        raise RuntimeError("SmoothQuant produced a non-finite scale")
    return scale


@torch.inference_mode()
def precondition_linear_for_smoothquant(
    linear: nn.Linear,
    scale: torch.Tensor,
) -> None:
    """Fold the weight side of an exact per-linear SmoothQuant transform."""

    if not isinstance(linear, nn.Linear):
        raise TypeError("SmoothQuant preconditioning expects nn.Linear")
    scale = scale.detach().float().reshape(-1)
    if scale.numel() != linear.in_features:
        raise ValueError("SmoothQuant scale must match linear.in_features")
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("SmoothQuant scale must be finite and positive")
    linear.weight.data.div_(
        scale.to(device=linear.weight.device, dtype=linear.weight.dtype).unsqueeze(0)
    )
    linear.faquant_input_scale = scale.to(linear.weight.device)
