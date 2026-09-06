from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import torch
from torch import nn

from .hif4 import CompactHiF4Parameters, HIF4_BLOCK_SIZE
from .qad import qwen3_hif4_qad_config
from .qad_quantization import (
    QATConversionStats,
    enable_hif4_qat,
    materialize_training_tensors,
    promote_hif4_activation_clips,
    promote_hif4_activation_companding,
    promote_hif4_input_scales,
)
from .quantization import FakeQuantLinear
from .qwen3 import prepare_model


@dataclass(frozen=True)
class QADCheckpointLoadStats:
    qat: QATConversionStats
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _placeholder_parameters(module: FakeQuantLinear) -> CompactHiF4Parameters:
    out_features, in_features = module.weight.shape
    groups = in_features // HIF4_BLOCK_SIZE
    leading = (out_features, groups)
    device = module.weight.device
    return CompactHiF4Parameters(
        scale=torch.ones(
            (*leading, 1, 1, 1),
            device=device,
            dtype=torch.float32,
        ),
        reciprocal=torch.ones(
            (*leading, 1, 1, 1),
            device=device,
            dtype=torch.float32,
        ),
        scale_lv2_exponent=torch.zeros(
            (*leading, 8, 1, 1),
            device=device,
            dtype=torch.int8,
        ),
        scale_lv3_exponent=torch.zeros(
            (*leading, 8, 2, 1),
            device=device,
            dtype=torch.int8,
        ),
    )


def build_qad_student_skeleton(
    model: nn.Module,
    *,
    rotation: str,
    seed: int = 0,
    hisq_block_size: int = 1024,
    hisq_down_proj_block_size: int | None = None,
    hisq_seed: int = 17,
    quant_exempt_layers: tuple[int, ...] = (),
    quant_exempt_modules: tuple[str, ...] = (),
) -> QATConversionStats:
    """Create the module/buffer structure needed to load a saved QAD init."""

    config = replace(
        qwen3_hif4_qad_config(
            rotation=rotation,
            seed=seed,
            hisq_block_size=hisq_block_size,
            hisq_down_proj_block_size=hisq_down_proj_block_size,
            hisq_seed=hisq_seed,
            quant_exempt_layers=quant_exempt_layers,
            quant_exempt_modules=quant_exempt_modules,
        ),
        weight_quant="rtn",
        qad_capture_weight_metadata=False,
    )
    prepare_model(model, config)
    modules = [
        module for module in model.modules() if isinstance(module, FakeQuantLinear)
    ]
    if not modules:
        raise RuntimeError("QAD skeleton contains no HiF4 inference linears")
    for module in modules:
        module.faquant_hif4_qat_parameters = _placeholder_parameters(module)
    with torch.inference_mode(False):
        stats = enable_hif4_qat(model)
    model.config.use_cache = False
    model.train()
    return stats


def _recorded_down_proj_block_size(checkpoint: Path) -> int | None:
    receipt = checkpoint / "qad_init.json"
    if not receipt.exists():
        return None
    value = json.loads(receipt.read_text(encoding="utf-8")).get(
        "hisq_down_proj_block_size"
    )
    return None if value is None else int(value)


def recorded_quant_exempt_modules(checkpoint: str | Path) -> tuple[str, ...]:
    """Read the per-projection protection list a checkpoint was built with."""

    receipt = Path(checkpoint) / "qad_init.json"
    if not receipt.exists():
        return ()
    value = json.loads(receipt.read_text(encoding="utf-8")).get(
        "quant_exempt_modules"
    )
    return () if not value else tuple(str(entry) for entry in value)


def recorded_quant_exempt_layers(checkpoint: str | Path) -> tuple[int, ...]:
    """Read the whole-layer protection list a checkpoint was built with.

    An exempt layer has plain ``nn.Linear`` modules where every other layer has
    ``FakeQuantLinear``. Rebuilding the skeleton with the wrong list therefore
    fails loudly on missing or unexpected keys rather than silently, but the
    error is obscure, and evaluation would in any case need the same list to
    keep that layer's attention core clean. Both callers take it from here.
    """

    receipt = Path(checkpoint) / "qad_init.json"
    if not receipt.exists():
        return ()
    value = json.loads(receipt.read_text(encoding="utf-8")).get(
        "quant_exempt_layers"
    )
    return () if not value else tuple(int(index) for index in value)


def load_qad_student_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    rotation: str,
    seed: int = 0,
    hisq_block_size: int = 1024,
    hisq_down_proj_block_size: int | None = None,
    hisq_seed: int = 17,
    trainable_input_scales: bool = False,
) -> QADCheckpointLoadStats:
    """Load a sharded ``save_pretrained`` QAD initialization checkpoint."""

    from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict

    # The HiSQ block size decides which Hadamard the runtime applies, but only
    # the permutation and signs are saved, and those have the same shape for
    # every block size. A caller who forgot the setting would therefore load
    # cleanly and silently compute the wrong rotation, so take it from the
    # checkpoint instead of trusting the argument.
    recorded = _recorded_down_proj_block_size(Path(checkpoint))
    if hisq_down_proj_block_size is None:
        hisq_down_proj_block_size = recorded
    elif recorded != hisq_down_proj_block_size:
        raise ValueError(
            "down_proj HiSQ block size disagrees with the checkpoint: "
            f"requested {hisq_down_proj_block_size}, built with {recorded}"
        )
    qat = build_qad_student_skeleton(
        model,
        rotation=rotation,
        seed=seed,
        hisq_block_size=hisq_block_size,
        hisq_down_proj_block_size=hisq_down_proj_block_size,
        hisq_seed=hisq_seed,
        quant_exempt_layers=recorded_quant_exempt_layers(checkpoint),
        quant_exempt_modules=recorded_quant_exempt_modules(checkpoint),
    )
    materialize_training_tensors(model)
    if trainable_input_scales:
        promote_hif4_input_scales(model)
    checkpoint = Path(checkpoint)
    training_receipt = checkpoint / "training_receipt.json"
    has_activation_companding = (checkpoint / "activation_companding.json").exists()
    if training_receipt.exists():
        receipt = json.loads(training_receipt.read_text())
        has_activation_companding = has_activation_companding or (
            receipt.get("args", {}).get("trainable_scope")
            == "activation-companding-only"
        )
        if receipt.get("args", {}).get("train_qk_smooth_scales", False):
            for layer in model.model.layers:
                attention = layer.self_attn
                attention.faquant_qk_smooth_delta = nn.Parameter(
                    torch.zeros(
                        attention.head_dim,
                        device=attention.q_proj.weight.device,
                        dtype=attention.q_proj.weight.dtype,
                    ),
                    requires_grad=True,
                )
    if has_activation_companding:
        group_size = 1
        if training_receipt.exists():
            group_size = int(
                receipt.get("args", {}).get(
                    "activation_companding_group_size", 1
                )
            )
        promote_hif4_activation_companding(model, group_size=group_size)
    residual_scope = (
        receipt.get("args", {}).get("trainable_scope")
        if training_receipt.exists()
        else None
    )
    if residual_scope in (
        "residual-affine-only",
        "residual-affine-fp32-only",
    ):
        from .residual_affine import install_qwen3_residual_affine

        install_qwen3_residual_affine(
            model,
            trainable=True,
            multiply_mode=(
                "fp32"
                if residual_scope == "residual-affine-fp32-only"
                else "legacy"
            ),
        )
    if (checkpoint / "activation_clipping.json").exists():
        promote_hif4_activation_clips(model)
    if (checkpoint / "model.safetensors.index.json").exists():
        result = load_sharded_checkpoint(
            model,
            str(checkpoint),
            strict=True,
            prefer_safe=True,
        )
    else:
        state = load_state_dict(str(checkpoint / "model.safetensors"))
        result = model.load_state_dict(state, strict=True)
    return QADCheckpointLoadStats(
        qat=qat,
        missing_keys=tuple(result.missing_keys),
        unexpected_keys=tuple(result.unexpected_keys),
    )
