"""Split a Qwen3 model's decoder layers over the visible GPUs without peer access.

A direct GPU-to-GPU copy makes PyTorch enable peer access. On driver 580.105.08,
one multi-GPU process setting up peer access while another multi-GPU process on
the same node allocated, freed or exited deadlocked the driver (every process
stuck in ``os_acquire_rwlock``). Staging those copies through host memory never
touches peer access, so several split models can share a node.
"""

from __future__ import annotations

import torch

_ORIGINAL: dict[str, object] = {}


def _target_device(args: tuple, kwargs: dict) -> torch.device | None:
    device = kwargs.get("device")
    if device is None and args:
        first = args[0]
        if isinstance(first, torch.Tensor):
            return first.device
        if isinstance(first, (str, torch.device)) or (
            isinstance(first, int) and not isinstance(first, bool)
        ):
            device = first
    if device is None:
        return None
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def install_cpu_staged_transfers() -> None:
    """Route every Python-level GPU-to-GPU tensor move through host memory.

    Covers ``Tensor.to``, ``Tensor.copy_`` and ``Tensor.cuda``, which is how
    ``nn.Module.to``, accelerate's dispatch hooks and transformers move tensors.
    Code that mixes devices inside one C++ op fails loudly instead of copying.
    """

    if _ORIGINAL:
        return
    to, copy_, cuda = torch.Tensor.to, torch.Tensor.copy_, torch.Tensor.cuda
    _ORIGINAL.update(to=to, copy_=copy_, cuda=cuda)

    def staged_to(self, *args, **kwargs):
        target = _target_device(args, kwargs)
        if target is not None and target.type == "cuda" and self.is_cuda and self.device != target:
            return to(to(self, "cpu"), *args, **kwargs)
        return to(self, *args, **kwargs)

    def staged_copy_(self, src, non_blocking=False):
        if (
            self.is_cuda
            and isinstance(src, torch.Tensor)
            and src.is_cuda
            and src.device != self.device
        ):
            src = to(src, "cpu")
        return copy_(self, src, non_blocking=non_blocking)

    def staged_cuda(self, device=None, non_blocking=False, **kwargs):
        target = _target_device((device,), {}) if device is not None else _target_device(("cuda",), {})
        if self.is_cuda and self.device != target:
            self = to(self, "cpu")
        return cuda(self, device, non_blocking=non_blocking, **kwargs)

    torch.Tensor.to = staged_to
    torch.Tensor.copy_ = staged_copy_
    torch.Tensor.cuda = staged_cuda


def layer_device_map(
    num_layers: int, num_devices: int, *, first_device_extra_layers: int = 0
) -> dict[str, int]:
    """Contiguous decoder blocks per GPU; embeddings, final norm, rotary and lm_head on GPU 0.

    GPU 0 also holds the embedding and the LM head, so it takes
    ``first_device_extra_layers`` fewer decoder layers to even out memory.
    """

    mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 0, "lm_head": 0}
    total = num_layers + first_device_extra_layers
    for index in range(num_layers):
        mapping[f"model.layers.{index}"] = min(
            num_devices - 1, (index + first_device_extra_layers) * num_devices // total
        )
    return mapping
