from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ExperimentConfig
from .qad_quantization import QATConversionStats, enable_hif4_qat
from .qwen3 import prepare_model

QAD_ROTATIONS = (
    "none", "hadamard", "hisq1024", "shared_hisq_od",
    "hisq_od", "hisq_norm", "hisq_global", "hisq_global_fused", "fused_od",
    "sbh_readers", "bh_readers", "bh256_readers", "bh64_readers", "bh_all",
    "bh32_readers", "bh16_readers", "bh_all_r64", "bh64_all", "rbh_r64", "rbhn_r64",
)
# Ablations of HiSQ on the residual readers (q/k/v, gate/up); o_proj and down_proj
# keep their per-linear rotations unless noted. hisq_od: readers unrotated.
# hisq_norm: one rotation per RMSNorm and layer. hisq_global: shared_hisq_od's
# network-wide rotation, applied online so RMSNorm's gamma stays in the activation.
# hisq_global_fused: the same with gamma folded into the weights. fused_od: gamma
# folded, readers unrotated. sbh_readers: hisq_global without its permutation, so
# its signs fold into every gamma and only the block Hadamard runs online.
# bh*_readers: the block Hadamard alone (1024, 256, 64, 32 or 16 wide). bh_all: the
# 1024 block Hadamard alone on all seven projections; bh_all_r64 narrows the readers'
# block to 64 and bh64_all every block. rbh_r64: bh_all_r64's online work with the part
# of HiSQ's randomization that folds offline put back -- the readers' network-wide signs
# and permutation (into gamma and the residual stream), o_proj's signs (into v_proj) and
# down_proj's signs and permutation (into the gate/up rows). rbhn_r64: the same with one sign
# and permutation per RMSNorm instead of one per network; the signs fold into each gamma and
# the permutation into the RMSNorm's output order.
_READER_SHARING = {
    "hisq_norm": "norm", "hisq_global": "global", "hisq_global_fused": "global",
    "sbh_readers": "global", "rbh_r64": "global", "rbhn_r64": "norm",
}
_O_CONSTRUCTION = {"rbh_r64": "signs", "rbhn_r64": "signs"}
_READER_CONSTRUCTION = {
    "sbh_readers": "signs", "bh_readers": "plain", "bh256_readers": "plain",
    "bh64_readers": "plain", "bh32_readers": "plain", "bh16_readers": "plain",
    "bh_all": "plain", "bh_all_r64": "plain", "bh64_all": "plain",
}
_READER_BLOCK = {
    "bh256_readers": 256, "bh64_readers": 64, "bh32_readers": 32, "bh16_readers": 16,
    "bh_all_r64": 64, "bh64_all": 64, "rbh_r64": 64, "rbhn_r64": 64,
}
_PLAIN_OD = ("bh_all", "bh_all_r64", "bh64_all")


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
    hisq_global_seed: int | None = None,
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

    if rotation not in QAD_ROTATIONS:
        raise ValueError(f"QAD rotation must be one of {QAD_ROTATIONS}")
    if weight_quant not in ("gptq", "rtn"):
        raise ValueError("QAD weight_quant must be 'gptq' or 'rtn'")
    if capture_latent_master and weight_quant != "gptq":
        raise ValueError("latent-master capture is only implemented for GPTQ")
    if qk_smooth_scales is not None and not deployed_attention:
        raise ValueError(
            "Smooth-QK scales only act on quantized Q/K, so they need "
            "deployed_attention=True during calibration"
        )
    # shared_hisq_od: one residual rotation shared by the network and the V/O head fold, both
    # folded into the weights, plus hisq1024's per-linear rotations of o_proj and down_proj,
    # whose inputs no shared rotation reaches.
    global_rotation = "hadamard" if rotation in ("hadamard", "shared_hisq_od") else "none"
    hisq_rotation = rotation not in ("none", "hadamard")
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
        qwen_hisq_rotation_block_size=64 if rotation == "bh64_all" else hisq_block_size,
        qwen_hisq_rotation_block_size_down_proj=hisq_down_proj_block_size,
        qwen_hisq_rotation_seed=hisq_seed,
        qwen_hisq_rotation_projections=(
            ("o_proj", "down_proj")
            if rotation in ("shared_hisq_od", "hisq_od", "fused_od")
            else ()
        ),
        qwen_hisq_rotation_sharing=_READER_SHARING.get(rotation, "linear"),
        qwen_hisq_global_seed=hisq_global_seed,
        qwen_hisq_reader_construction=_READER_CONSTRUCTION.get(rotation, "full"),
        qwen_hisq_od_construction="plain" if rotation in _PLAIN_OD else "full",
        qwen_hisq_o_construction=_O_CONSTRUCTION.get(rotation),
        qwen_hisq_reader_block_size=_READER_BLOCK.get(rotation),
        qwen_fuse_reader_norms=rotation in ("hisq_global_fused", "fused_od"),
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

    if rotation not in QAD_ROTATIONS:
        raise ValueError(f"QAD rotation must be one of {QAD_ROTATIONS}")
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    return model
