from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualAffineStats:
    layers: int
    hidden_size: int
    parameters: int


@dataclass(frozen=True)
class ResidualAffineProjectionStats:
    parameters: int
    maximum_active: int
    active_before: int
    active_after: int
    pruned: int


@torch.no_grad()
def project_qwen3_residual_affine(
    model: nn.Module,
    *,
    maximum_active_fraction: float,
) -> ResidualAffineProjectionStats:
    """Cap the number of deployed non-identity legacy BF16 gains.

    The optimizer may continue refining all latent FP32 log-scales, but only
    the largest-magnitude gains are allowed to cross the deployed BF16
    identity cell at a given step.  This is a code-space trust region rather
    than a norm penalty on an optimizer representation.
    """

    if not 0.0 < maximum_active_fraction <= 1.0:
        raise ValueError("maximum active fraction must be in (0, 1]")
    parameter = model._parameters.get("faquant_residual_log_scale")
    if parameter is None:
        raise ValueError("residual affine is not installed")
    mode = getattr(model, "_faquant_residual_affine_multiply_mode", None)
    if mode != "legacy":
        raise ValueError("active-gain projection requires legacy multiply mode")

    flat = parameter.view(-1)
    effective = flat.clamp(-0.1, 0.1).exp().to(torch.bfloat16)
    active = effective.ne(1.0)
    active_before = int(active.sum().item())
    maximum_active = int(flat.numel() * maximum_active_fraction)
    maximum_active = max(1, min(flat.numel(), maximum_active))
    if active_before > maximum_active:
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        scores = flat[active_indices].abs()
        keep_local = torch.topk(
            scores,
            k=maximum_active,
            largest=True,
            sorted=False,
        ).indices
        keep = torch.zeros_like(active)
        keep[active_indices[keep_local]] = True
        flat[active & ~keep] = 0.0
    active_after = int(
        flat.clamp(-0.1, 0.1).exp().to(torch.bfloat16).ne(1.0).sum().item()
    )
    return ResidualAffineProjectionStats(
        parameters=flat.numel(),
        maximum_active=maximum_active,
        active_before=active_before,
        active_after=active_after,
        pruned=active_before - active_after,
    )


def _scaled_hidden(
    hidden: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    multiply_mode: str,
) -> torch.Tensor:
    if multiply_mode == "legacy":
        scale = log_scale.clamp(-0.1, 0.1).exp().to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return hidden * scale
    if multiply_mode == "fp32":
        # Keep the learned FP32 gain through the multiply and round only the
        # corrected residual.  Casting the gain first collapses almost every
        # sub-ULP update to exactly one in BF16.
        corrected = hidden.float() * log_scale.clamp(-0.1, 0.1).exp().to(
            device=hidden.device,
            dtype=torch.float32,
        )
        return corrected.to(hidden.dtype)
    raise ValueError(f"unsupported residual affine multiply mode {multiply_mode!r}")


def install_qwen3_residual_affine(
    model: nn.Module,
    *,
    trainable: bool,
    multiply_mode: str = "legacy",
) -> ResidualAffineStats:
    """Install a step-0 identity diagonal gain after every decoder layer.

    HiSQ is per-Linear, so the student residual stream remains in the same
    basis as the BF16 teacher.  These gains can therefore correct accumulated
    W4A4 error without adding a Linear or changing any HiF4 weight code.
    """

    if multiply_mode not in ("legacy", "fp32"):
        raise ValueError(
            "residual affine multiply mode must be 'legacy' or 'fp32'"
        )
    layers = model.model.layers
    layer_count = len(layers)
    hidden_size = int(model.config.hidden_size)
    expected_shape = (layer_count, hidden_size)
    parameter = model._parameters.get("faquant_residual_log_scale")
    if parameter is None:
        existing = model._buffers.pop("faquant_residual_log_scale", None)
        if existing is None:
            existing = torch.zeros(
                expected_shape,
                device=next(model.parameters()).device,
                dtype=torch.float32,
            )
        if tuple(existing.shape) != expected_shape:
            raise ValueError(
                "residual affine shape mismatch: "
                f"expected {expected_shape}, got {tuple(existing.shape)}"
            )
        parameter = nn.Parameter(
            existing.detach().float().clone(),
            requires_grad=trainable,
        )
        model.register_parameter("faquant_residual_log_scale", parameter)
    else:
        if tuple(parameter.shape) != expected_shape:
            raise ValueError(
                "residual affine parameter shape mismatch: "
                f"expected {expected_shape}, got {tuple(parameter.shape)}"
            )
        parameter.requires_grad_(trainable)

    model._faquant_residual_affine_multiply_mode = multiply_mode
    handles = getattr(model, "_faquant_residual_affine_handles", None)
    if handles is None:
        handles = []
        for layer_index, layer in enumerate(layers):
            def hook(
                _module: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: object,
                *,
                index: int = layer_index,
            ) -> object:
                current = model.faquant_residual_log_scale[index]
                mode = model._faquant_residual_affine_multiply_mode
                if isinstance(output, tuple):
                    return (
                        _scaled_hidden(
                            output[0], current, multiply_mode=mode
                        ),
                        *output[1:],
                    )
                if isinstance(output, torch.Tensor):
                    return _scaled_hidden(output, current, multiply_mode=mode)
                raise TypeError(
                    "Qwen3 decoder layer output must be a tensor or tuple"
                )

            handles.append(layer.register_forward_hook(hook))
        model._faquant_residual_affine_handles = handles
    return ResidualAffineStats(
        layers=layer_count,
        hidden_size=hidden_size,
        parameters=parameter.numel(),
    )
