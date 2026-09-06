from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ExperimentConfig
from .qad_quantization import QATConversionStats, enable_hif4_qat
from .qwen3 import prepare_model


@dataclass(frozen=True)
class QADPreparationStats:
    rotation: str
    online_hadamard: bool
    qat: QATConversionStats


def qwen3_hif4_qad_config(
    *,
    rotation: str,
    gptq_nsamples: int = 128,
    gptq_seqlen: int = 2048,
    seed: int = 0,
    hisq_block_size: int = 1024,
    hisq_down_proj_block_size: int | None = None,
    hisq_seed: int = 17,
    value_head_rotation: bool = False,
    capture_latent_master: bool = False,
    weight_quant: str = "gptq",
    quant_exempt_layers: tuple[int, ...] = (),
    quant_exempt_modules: tuple[str, ...] = (),
    qk_matmul_exempt_layers: tuple[int, ...] = (),
    pv_matmul_exempt_layers: tuple[int, ...] = (),
    qk_matmul_mxfp8_layers: tuple[int, ...] = (),
    pv_matmul_mxfp8_layers: tuple[int, ...] = (),
    gptq_damp: float = 0.01,
    gptq_mxfp8_hessian_fraction: float = 1.0,
    gptq_mxfp8_hessian_mode: str = "paired",
    gptq_damp_overrides: tuple[tuple[str, float], ...] = (),
    hif4_scale_search_steps: tuple[int, ...] = (),
    sequential_groups: bool = False,
    act_order_within_group: bool = False,
    deployed_attention: bool = False,
    qk_smooth_scales: str | None = None,
) -> ExperimentConfig:
    """Return the exact online-none HiF4 configuration used by QAD.

    ``weight_quant="rtn"`` drops the second-order weight fit and gives the
    no-PTQ starting point the reference recipe distils from; combined with
    ``rotation="none"`` and no value-head rotation it is a student that has
    had nothing done to it but direct-cast quantization.
    """

    if rotation not in ("none", "hadamard", "hisq1024"):
        raise ValueError("QAD rotation must be 'none', 'hadamard', or 'hisq1024'")
    if weight_quant not in ("gptq", "rtn"):
        raise ValueError("QAD weight_quant must be 'gptq' or 'rtn'")
    if capture_latent_master and weight_quant != "gptq":
        raise ValueError("latent-master capture is only implemented for GPTQ")
    if qk_smooth_scales is not None and not deployed_attention:
        raise ValueError(
            "Smooth-QK scales only act on quantized Q/K, so they need "
            "deployed_attention=True during calibration"
        )
    global_rotation = "hadamard" if rotation == "hadamard" else "none"
    hisq_rotation = rotation == "hisq1024"
    # Calibrating under the deployed attention core. Historically GPTQ ran with
    # exact attention, so every Hessian downstream of an attention block -- o_proj
    # first, then every later layer -- was built from activations the deployed
    # model never produces. Whether that matters depends on how far the attention
    # quantizer moves the activations, and this one moves them a long way: it
    # zeroes roughly half of the softmax probabilities and shrinks the output
    # norm by about 5%, which is a systematic shift rather than small noise.
    return ExperimentConfig(
        rotation=global_rotation,
        online_hadamard=rotation == "hadamard",
        attention_kernel="simulated" if deployed_attention else "native",
        attention_input_quant=False,
        attention_output_quant=True,
        qk_matmul_quant=deployed_attention,
        pv_matmul_quant=deployed_attention,
        post_rope_qk_rotation=deployed_attention,
        pv_normalizer_mode="quantized_same" if deployed_attention else "unquantized",
        qk_smooth_scales=qk_smooth_scales,
        quant_target="all",
        quant_format="hif4",
        bits=4,
        weight_group_size=64,
        activation_group_size=64,
        symmetric=True,
        clip_ratio=1.0,
        seed=seed,
        weight_quant=weight_quant,
        gptq_nsamples=gptq_nsamples,
        gptq_seqlen=gptq_seqlen,
        gptq_damp=gptq_damp,
        gptq_mxfp8_hessian_fraction=gptq_mxfp8_hessian_fraction,
        gptq_mxfp8_hessian_mode=gptq_mxfp8_hessian_mode,
        gptq_damp_overrides=tuple(gptq_damp_overrides),
        gptq_hif4_scale_search_steps=tuple(hif4_scale_search_steps),
        gptq_sequential_groups=sequential_groups,
        gptq_act_order_within_group=act_order_within_group,
        gptq_propagate_fake_activations=True,
        qad_capture_weight_metadata=True,
        qad_capture_latent_master=capture_latent_master,
        qwen_hisq_input_rotation=hisq_rotation,
        qwen_hisq_rotation_block_size=hisq_block_size,
        qwen_hisq_rotation_block_size_down_proj=hisq_down_proj_block_size,
        qwen_hisq_rotation_seed=hisq_seed,
        qwen_value_head_rotation=value_head_rotation,
        qwen_quant_exempt_layers=tuple(quant_exempt_layers),
        qwen_quant_exempt_modules=tuple(quant_exempt_modules),
        qwen_qk_matmul_exempt_layers=tuple(qk_matmul_exempt_layers),
        qwen_pv_matmul_exempt_layers=tuple(pv_matmul_exempt_layers),
        qwen_qk_matmul_mxfp8_layers=tuple(qk_matmul_mxfp8_layers),
        qwen_pv_matmul_mxfp8_layers=tuple(pv_matmul_mxfp8_layers),
    )


def prepare_qad_student(
    model: nn.Module,
    *,
    rotation: str,
    calibration_input_ids: torch.Tensor,
    gptq_nsamples: int = 128,
    gptq_seqlen: int = 2048,
    seed: int = 0,
    hisq_block_size: int = 1024,
    hisq_down_proj_block_size: int | None = None,
    hisq_seed: int = 17,
    value_head_rotation: bool = False,
    capture_latent_master: bool = False,
) -> QADPreparationStats:
    config = qwen3_hif4_qad_config(
        rotation=rotation,
        gptq_nsamples=gptq_nsamples,
        gptq_seqlen=gptq_seqlen,
        seed=seed,
        hisq_block_size=hisq_block_size,
        hisq_down_proj_block_size=hisq_down_proj_block_size,
        hisq_seed=hisq_seed,
        value_head_rotation=value_head_rotation,
        capture_latent_master=capture_latent_master,
    )
    prepare_model(
        model,
        config,
        calibration_input_ids=calibration_input_ids,
    )
    with torch.inference_mode(False):
        qat = enable_hif4_qat(model)
    model.train()
    model.config.use_cache = False
    return QADPreparationStats(
        rotation=rotation,
        online_hadamard=config.online_hadamard,
        qat=qat,
    )


def prepare_qad_teacher(
    model: nn.Module,
    *,
    rotation: str,
    seed: int = 0,
) -> nn.Module:
    """Freeze the original BF16 teacher used for logit targets.

    For a rotated student, selected teacher hidden states are transformed into
    the rotated residual basis at LAFD time. Keeping the teacher itself
    unrotated avoids BF16 numerical drift in the authoritative logit target.
    """

    if rotation not in ("none", "hadamard", "hisq1024"):
        raise ValueError("QAD rotation must be 'none', 'hadamard', or 'hisq1024'")
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    return model
