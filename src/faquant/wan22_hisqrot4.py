"""Run the released HiSQRot4 (GCC-HiFloat) Wan2.2 W4A4 path within one GPU's memory.

HiSQRot4 replaces every Linear of both DiT experts with PTQPreparedLinearWrapper
(SmoothQuant mask, per-layer 1024-block Hadamard, timestep/branch lookup clamp,
HiF4 QDQ of the input, GEMM with the prepared HiF4 weights). Three changes make
it fit at 1280x720x81 without changing a single output value:

- Device moves carry the prepared tensors. They live in a dict, so Module.to()
  skips them and, after the high->low expert switch, both experts' weights
  would stay on the GPU.
- One bf16 copy of the weights. The Linear inputs are fp32 under the bf16
  autocast of WanT2V.generate, so the wrapper caches an fp32 copy of the fp16
  weights next to them (84 GB per expert), while F.linear casts both operands to
  bf16 anyway. HiF4 grid values are exact in bf16 (checked per layer).
- Row chunks. Every step of the wrapper is row-wise (the lookup clamp uses fixed
  per-step bounds), so chunking rows only bounds the fp32 temporaries.

The outputs were checked bit for bit against the unpatched wrapper on real layers.
"""

from __future__ import annotations

import torch

_PATCHED = False


def patch_wrapper(wrapper_cls, chunk_rows: int = 16384) -> None:
    global _PATCHED
    if _PATCHED:
        return
    original_apply = wrapper_cls._apply
    original_get_weight = wrapper_cls._get_weight
    original_forward = wrapper_cls.forward

    def follow_device(self, fn, *args, **kwargs):
        out = original_apply(self, fn, *args, **kwargs)
        for state in [self.default_state, *self.bank_states]:
            for key in ("weight_fp4", "bias", "smoothquant_channel_mask"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    moved = fn(value)
                    if moved.dtype == value.dtype:
                        state[key] = moved
        return out

    def bf16_weight(self, state, device, dtype):
        weight = state["weight_fp4"]
        if weight.dtype != torch.bfloat16:
            cast = weight.to(torch.bfloat16)
            if not torch.equal(cast.to(weight.dtype), weight):
                raise RuntimeError("prepared weight is not exactly representable in bf16")
            state["weight_fp4"] = weight = cast
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
            return weight.to(device=device)
        return original_get_weight(self, state, device, dtype)

    def chunked_forward(self, x):
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] <= chunk_rows:
            return original_forward(self, x)
        outs = [original_forward(self, part) for part in rows.split(chunk_rows)]
        return torch.cat(outs).reshape(*x.shape[:-1], -1)

    wrapper_cls._apply = follow_device
    wrapper_cls._get_weight = bf16_weight
    wrapper_cls.forward = chunked_forward
    wrapper_cls._faquant_originals = (original_apply, original_get_weight, original_forward)
    _PATCHED = True
