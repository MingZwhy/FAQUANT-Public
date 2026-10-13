"""Run a CPU-resident Qwen3 model one decoder layer at a time on one GPU.

A 32B model with HiF4 metadata does not fit one card, and splitting it across
cards needs peer access, which can deadlock the NVIDIA driver on some systems.
Keeping the weights on CPU and bringing in one layer at a time needs neither.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


def move_layer(module: nn.Module, device: torch.device | str) -> None:
    """Move a module together with the tensor attributes FA-Quant hangs on it.

    ``Module.to`` only sees parameters and buffers; Smooth-QK scales, HiSQ
    rotation metadata and the HiF4 parameters GPTQ captures are plain
    attributes named ``faquant_*`` and would otherwise stay behind.
    """

    # Outside inference mode, so the moved tensors stay ordinary tensors that
    # can move again later.
    with torch.inference_mode(False):
        module.to(device)
        for sub in module.modules():
            for name, value in list(vars(sub).items()):
                if not name.startswith("faquant_"):
                    continue
                if isinstance(value, torch.Tensor):
                    setattr(sub, name, value.to(device))
                elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                    fields = {
                        field.name: getattr(value, field.name)
                        for field in dataclasses.fields(value)
                    }
                    if fields and all(
                        isinstance(item, torch.Tensor) for item in fields.values()
                    ):
                        setattr(
                            sub,
                            name,
                            dataclasses.replace(
                                value,
                                **{key: item.to(device) for key, item in fields.items()},
                            ),
                        )


@contextmanager
def layerwise_execution(model: nn.Module, device: torch.device | str) -> Iterator[None]:
    """Forward a CPU-resident model with only one decoder layer on ``device``.

    Embeddings, rotary tables, the final norm and the head stay on ``device``
    for the duration; each decoder layer is moved in by a pre-hook and back out
    by a post-hook, so activations never leave the GPU.
    """

    core = model.model
    resident = [core.embed_tokens, core.rotary_emb, core.norm, model.lm_head]
    for module in resident:
        move_layer(module, device)
    handles = []
    for layer in core.layers:
        handles.append(
            layer.register_forward_pre_hook(
                lambda module, _args: move_layer(module, device)
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda module, _args, _output: move_layer(module, "cpu")
            )
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module in resident:
            move_layer(module, "cpu")
