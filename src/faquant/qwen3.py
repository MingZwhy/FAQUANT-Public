from __future__ import annotations

import os
import types
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from typing_extensions import Unpack
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .config import (
    PV_NORMALIZER_MODES,
    ExperimentConfig,
    resolve_attention_matmul_exempt_layers,
    resolve_attention_matmul_mxfp8_layers,
    resolve_quant_exempt_layers,
    resolve_quant_exempt_modules,
)
from .gptq import calibration_tokens, gptq_quantize_qwen3
from .hisq_rotation import apply_qwen_hisq_input_rotation
from .qad_attention import quantized_eager_attention_forward
from .quantization import fake_quantize, quantize_qwen3_linears
from .rotation import (
    apply_qwen3_global_rotation,
    apply_qwen3_value_head_rotation,
    chunked_block_hadamard_transform,
    hadamard_transform,
)
from .simulated_flash_attention import (
    SIMULATED_ATTENTION_IMPLEMENTATION,
    new_simulated_attention_stats,
    register_simulated_attention_backend,
    simulated_flash_attention_forward,
)


DEFAULT_MODEL = os.environ.get("FAQUANT_MODEL", "Qwen/Qwen3-8B")


def _attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3 attention with post-RoPE Q/K rotation and FA-core fake quant."""

    legacy_past_key_value = kwargs.pop("past_key_value", None)
    if past_key_values is None:
        past_key_values = legacy_past_key_value
    elif (
        legacy_past_key_value is not None
        and legacy_past_key_value is not past_key_values
    ):
        raise ValueError("received both past_key_value and past_key_values")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    smooth_scale = self.faquant_qk_smooth_scale
    smooth_delta = getattr(self, "faquant_qk_smooth_delta", None)
    if smooth_delta is not None:
        if smooth_scale is None:
            raise RuntimeError("Smooth-QK delta requires a frozen base scale")
        smooth_scale = smooth_scale * smooth_delta.float().exp()
    key_offset = self.faquant_qk_key_offset
    if smooth_scale is not None or key_offset is not None or self.faquant_qk_rotation:
        # One float32 excursion covering every transform. Keeping a single cast
        # back to bf16 matters: the rotation alone already costs 0.4202 pp of MMLU
        # from that rounding, so these must not add a second round trip.
        query_float = query_states.float()
        key_float = key_states.float()
        if key_offset is not None:
            # Softmax-invariant rather than score-preserving: this shifts every
            # score in a row by the same -q_i.k_bar, which the running max absorbs
            # exactly. It removes the token-shared offset that would otherwise
            # inflate the max of every HiF4 group K lands in.
            key_float = key_float - key_offset
        if smooth_scale is not None:
            # Exact by construction: (Q diag(s)) (K diag(s)^-1)^T = Q K^T.
            query_float = query_float * smooth_scale
            key_float = key_float / smooth_scale
        if self.faquant_qk_rotation:
            rotation_block_size = getattr(
                self, "faquant_qk_rotation_block_size", None
            )
            if rotation_block_size is None or rotation_block_size == self.head_dim:
                query_float = hadamard_transform(query_float)
                key_float = hadamard_transform(key_float)
            else:
                query_float = chunked_block_hadamard_transform(
                    query_float,
                    block_size=rotation_block_size,
                )
                key_float = chunked_block_hadamard_transform(
                    key_float,
                    block_size=rotation_block_size,
                )
        query_states = query_float.to(query_states.dtype)
        key_states = key_float.to(key_states.dtype)

    if self.faquant_attention_input_quant:
        quant_kwargs = self.faquant_core_quant_kwargs
        query_states = fake_quantize(query_states, **quant_kwargs)
        key_states = fake_quantize(key_states, **quant_kwargs)
        value_states = fake_quantize(value_states, **quant_kwargs)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if self.faquant_attention_kernel == "simulated":
        attention_interface: Callable = simulated_flash_attention_forward
        kwargs["cache_position"] = cache_position
    elif self.faquant_attention_kernel == "differentiable":
        attention_interface = quantized_eager_attention_forward
    else:
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    if self.faquant_attention_output_quant and not getattr(
        self.o_proj, "faquant_quantizes_input", False
    ):
        attn_output = fake_quantize(attn_output, **self.faquant_core_quant_kwargs)
    return self.o_proj(attn_output), attn_weights


def install_qwen3_attention_adapter(model: nn.Module, config: ExperimentConfig) -> None:
    """Install the minimum version-pinned attention patch needed by FA-Quant."""

    # Two routes to the same per-head Hadamard on Q/K: the legacy global
    # rotation bundle, and the explicit post-RoPE switch the deployed recipe
    # uses. They set the same attribute, so either one turning it on is enough.
    rotate = (
        config.rotation == "hadamard" and config.online_hadamard
    ) or config.post_rope_qk_rotation
    quantize_attention = config.quant_target in ("attention", "all")
    input_quant = quantize_attention and config.attention_input_quant
    output_quant = quantize_attention and config.attention_output_quant
    depth = len(model.model.layers)
    adapter_exempt = resolve_quant_exempt_layers(config, depth)
    adapter_modules = resolve_quant_exempt_modules(config, depth)
    qk_matmul_exempt, pv_matmul_exempt = (
        resolve_attention_matmul_exempt_layers(config, depth)
    )
    qk_mxfp8, pv_mxfp8 = resolve_attention_matmul_mxfp8_layers(config, depth)
    if qk_matmul_exempt and not config.qk_matmul_quant:
        raise ValueError(
            "QK matmul exemptions require qk_matmul_quant during calibration"
        )
    if pv_matmul_exempt and not config.pv_matmul_quant:
        raise ValueError(
            "PV matmul exemptions require pv_matmul_quant during calibration"
        )
    if qk_mxfp8 and not config.qk_matmul_quant:
        raise ValueError("QK MXFP8 protection requires qk_matmul_quant")
    if pv_mxfp8 and not config.pv_matmul_quant:
        raise ValueError("PV MXFP8 protection requires pv_matmul_quant")
    if qk_matmul_exempt & qk_mxfp8 or pv_matmul_exempt & pv_mxfp8:
        raise ValueError("one attention matmul cannot be both exempt and MXFP8")
    if (
        not rotate
        and not input_quant
        and not output_quant
        and config.attention_kernel == "native"
    ):
        return
    quant_kwargs = {
        "quant_format": config.quant_format,
        "bits": config.bits,
        "group_size": config.activation_group_size,
        "symmetric": config.symmetric,
        "clip_ratio": config.clip_ratio,
    }
    mxfp8_kwargs = {
        "quant_format": "mxfp8e4m3",
        "bits": 8,
        "group_size": 32,
        "symmetric": True,
        "clip_ratio": 1.0,
    }
    for index, layer in enumerate(model.model.layers):
        # A whole-layer exemption means nothing in this layer is quantized, so
        # the attention core has to be clean too.
        layer_exempt = index in adapter_exempt
        # The attention-output QDQ is o_proj's input quantizer, just applied
        # before the module rather than inside it. Protecting o_proj alone and
        # leaving this on would quantize its input anyway.
        layer_output_quant = (
            output_quant
            and not layer_exempt
            and (index, "o_proj") not in adapter_modules
        )
        attention = layer.self_attn
        attention.faquant_qk_rotation = rotate and not layer_exempt
        attention.faquant_qk_rotation_block_size = (
            32 if index in qk_mxfp8 else None
        )
        attention.faquant_attention_kernel = config.attention_kernel
        attention.faquant_attention_input_quant = input_quant and not layer_exempt
        attention.faquant_attention_output_quant = layer_output_quant
        attention.faquant_qk_matmul_quant = (
            config.qk_matmul_quant
            and not layer_exempt
            and index not in qk_matmul_exempt
        )
        attention.faquant_pv_matmul_quant = (
            config.pv_matmul_quant
            and not layer_exempt
            and index not in pv_matmul_exempt
        )
        attention.faquant_pv_normalizer_mode = config.pv_normalizer_mode
        attention.faquant_pv_normalizer_diagnostics = None
        attention.faquant_qk_smooth_scale = None
        attention.faquant_qk_smooth_delta = None
        attention.faquant_qk_key_offset = None
        attention.faquant_attention_query_chunk_size = config.attention_query_chunk_size
        attention.faquant_attention_key_chunk_size = config.attention_key_chunk_size
        attention.faquant_core_quant_kwargs = quant_kwargs
        attention.faquant_matmul_quant_kwargs = quant_kwargs
        attention.faquant_qk_matmul_quant_kwargs = (
            mxfp8_kwargs if index in qk_mxfp8 else quant_kwargs
        )
        attention.faquant_pv_matmul_quant_kwargs = (
            mxfp8_kwargs if index in pv_mxfp8 else quant_kwargs
        )
        # Before o_proj is wrapped, the explicit attention-output boundary
        # supplies its activation QDQ. GPTQ hooks use this marker to avoid
        # quantizing the same physical tensor a second time.
        attention.o_proj.faquant_input_prequantized = layer_output_quant
        attention.faquant_simulated_attention_stats = new_simulated_attention_stats()
        attention.forward = types.MethodType(_attention_forward, attention)


def apply_qwen3_qk_smooth_scales(
    model: nn.Module, artifact: str | Path | dict[str, object]
) -> dict[str, object]:
    """Install calibrated post-RoPE Smooth-QK scales and optional K offsets.

    Neither piece is a mode: they change which operand carries the channel
    imbalance without changing what the softmax computes, so they are installed
    once from a frozen calibration artifact and left alone by runtime mode
    switches. Returns the artifact metadata for provenance.
    """

    if isinstance(artifact, (str, Path)):
        artifact = torch.load(artifact, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or (
        "scales" not in artifact and "key_offsets" not in artifact
    ):
        raise ValueError(
            "Smooth-QK artifact must be a dict with a 'scales' or 'key_offsets' entry"
        )
    scales = artifact.get("scales")
    key_offsets = artifact.get("key_offsets")
    layers = list(model.model.layers)
    for name, table in (("scales", scales), ("key_offsets", key_offsets)):
        if table is not None and len(table) != len(layers):
            raise ValueError(
                f"Smooth-QK artifact {name} covers {len(table)} layers, model has "
                f"{len(layers)}"
            )
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        if not hasattr(attention, "faquant_qk_smooth_scale"):
            raise RuntimeError("Qwen attention adapter is not installed")
        key = f"model.layers.{index}.self_attn"
        device = attention.q_proj.weight.device
        if scales is not None:
            if key not in scales:
                raise ValueError(f"Smooth-QK artifact is missing {key}")
            scale = scales[key].to(dtype=torch.float32)
            if scale.shape != (attention.head_dim,):
                raise ValueError(
                    f"{key}: scale has shape {tuple(scale.shape)}, expected "
                    f"{(attention.head_dim,)}"
                )
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                raise ValueError(f"{key}: scale must be finite and strictly positive")
            attention.faquant_qk_smooth_scale = scale.to(device=device)
        if key_offsets is not None:
            if key not in key_offsets:
                raise ValueError(f"Smooth-QK artifact is missing {key} key offset")
            offset = key_offsets[key].to(dtype=torch.float32)
            expected = (attention.config.num_key_value_heads, attention.head_dim)
            if offset.shape != expected:
                raise ValueError(
                    f"{key}: key offset has shape {tuple(offset.shape)}, expected "
                    f"{expected}"
                )
            if not bool(torch.isfinite(offset).all()):
                raise ValueError(f"{key}: key offset must be finite")
            # Broadcast over the key axis of [batch, kv_heads, seq, head_dim].
            attention.faquant_qk_key_offset = offset[:, None, :].to(device=device)
    return {
        key: value
        for key, value in artifact.items()
        if key not in ("scales", "key_offsets")
    }


def promote_qwen3_qk_smooth_scales(model: nn.Module) -> int:
    """Add a BF16 log-delta while preserving the FP32 calibrated base exactly."""

    promoted = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        scale = getattr(attention, "faquant_qk_smooth_scale", None)
        if scale is None:
            raise RuntimeError("trainable Smooth-QK requires installed scales")
        delta = getattr(attention, "faquant_qk_smooth_delta", None)
        if isinstance(delta, nn.Parameter):
            delta.requires_grad_(True)
            promoted += 1
            continue
        attention.faquant_qk_smooth_delta = nn.Parameter(
            torch.zeros(
                attention.head_dim,
                device=scale.device,
                dtype=attention.q_proj.weight.dtype,
            ),
            requires_grad=True,
        )
        promoted += 1
    return promoted


@torch.no_grad()
def clamp_qwen3_qk_smooth_scales(
    model: nn.Module,
    *,
    minimum: float = 1.0 / 32.0,
    maximum: float = 32.0,
) -> None:
    for layer in model.model.layers:
        attention = layer.self_attn
        scale = getattr(attention, "faquant_qk_smooth_scale", None)
        delta = getattr(attention, "faquant_qk_smooth_delta", None)
        if scale is not None and isinstance(delta, nn.Parameter):
            lower = (minimum / scale).log()
            upper = (maximum / scale).log()
            delta.copy_(
                torch.maximum(
                    torch.minimum(delta.float(), upper),
                    lower,
                ).to(delta.dtype)
            )


def configure_qwen3_attention_runtime(
    model: nn.Module,
    *,
    kernel: str,
    input_quant: bool,
    output_quant: bool,
    qk_matmul_quant: bool,
    pv_matmul_quant: bool,
    post_rope_qk_rotation: bool | None = None,
    pv_normalizer_mode: str | None = None,
    qk_exempt_layers: Iterable[int] = (),
    pv_exempt_layers: Iterable[int] = (),
    qk_mxfp8_layers: Iterable[int] = (),
    pv_mxfp8_layers: Iterable[int] = (),
    exempt_layers: Iterable[int] = (),
    exempt_modules: Iterable[tuple[int, str]] = (),
) -> None:
    """Switch attention-core modes without changing already-quantized weights.

    ``qk_exempt_layers`` and ``pv_exempt_layers`` leave the named layers' QK or
    PV matmul in high precision while every other layer stays quantized. This
    spends the configured protection budget on individual attention matmuls.
    Layer indices are the decoder positions, 0-based.

    ``exempt_layers`` is the whole-layer version and is stronger: those layers
    were built unquantized, so their attention core must also stay clean, and
    the post-RoPE rotation and Smooth-QK scaling are dropped there as well.
    They would be exact transforms with nothing to protect, and would only add
    a bf16 round-trip.

    ``exempt_modules`` names individual protected projections. Only ``o_proj``
    matters here: the attention-output QDQ is its input quantizer applied
    before the module, so protecting o_proj has to switch that off too.
    """

    # "differentiable" is the same arithmetic as "simulated" in a single tile,
    # written out of place so a QAD run can backpropagate through the attention
    # core. It exists for training; evaluation stays on "simulated", which is
    # the instrument every historical number was measured with.
    quantizing_kernels = ("simulated", "differentiable")
    if kernel not in ("native", *quantizing_kernels):
        raise ValueError(f"unsupported attention kernel {kernel!r}")
    if kernel not in quantizing_kernels and (qk_matmul_quant or pv_matmul_quant):
        raise ValueError(
            "QK/PV matmul quantization requires the simulated or "
            "differentiable kernel"
        )
    if pv_normalizer_mode is not None:
        if pv_normalizer_mode not in PV_NORMALIZER_MODES:
            raise ValueError(
                f"pv_normalizer_mode must be one of {PV_NORMALIZER_MODES}"
            )
        if pv_normalizer_mode == "quantized_same" and kernel not in quantizing_kernels:
            raise ValueError(
                "pv_normalizer_mode='quantized_same' requires the simulated "
                "or differentiable kernel"
            )
    depth = len(model.model.layers)
    whole_exempt = frozenset(exempt_layers)
    protected_modules = frozenset(exempt_modules)
    # A whole-layer exemption implies both matmul exemptions, so callers only
    # have to pass the one list.
    qk_exempt = frozenset(qk_exempt_layers) | whole_exempt
    pv_exempt = frozenset(pv_exempt_layers) | whole_exempt
    qk_mxfp8 = frozenset(qk_mxfp8_layers)
    pv_mxfp8 = frozenset(pv_mxfp8_layers)
    for name, exempt in (
        ("qk", qk_exempt),
        ("pv", pv_exempt),
        ("qk_mxfp8", qk_mxfp8),
        ("pv_mxfp8", pv_mxfp8),
        ("layer", whole_exempt),
    ):
        out_of_range = sorted(index for index in exempt if not 0 <= index < depth)
        if out_of_range:
            raise ValueError(
                f"{name} exempt layers {out_of_range} outside 0..{depth - 1}"
            )
    if qk_exempt and not qk_matmul_quant:
        raise ValueError("qk_exempt_layers is meaningless with qk_matmul_quant off")
    if pv_exempt and not pv_matmul_quant:
        raise ValueError("pv_exempt_layers is meaningless with pv_matmul_quant off")
    if qk_mxfp8 and not qk_matmul_quant:
        raise ValueError("qk_mxfp8_layers is meaningless with qk_matmul_quant off")
    if pv_mxfp8 and not pv_matmul_quant:
        raise ValueError("pv_mxfp8_layers is meaningless with pv_matmul_quant off")
    if qk_exempt & qk_mxfp8 or pv_exempt & pv_mxfp8:
        raise ValueError("one attention matmul cannot be exempt and MXFP8")
    mxfp8_kwargs = {
        "quant_format": "mxfp8e4m3",
        "bits": 8,
        "group_size": 32,
        "symmetric": True,
        "clip_ratio": 1.0,
    }
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if not hasattr(attention, "faquant_attention_kernel"):
            raise RuntimeError("Qwen attention adapter is not installed")
        layer_exempt = index in whole_exempt
        attention.faquant_attention_kernel = kernel
        attention.faquant_attention_input_quant = input_quant and not layer_exempt
        attention.faquant_attention_output_quant = (
            output_quant
            and not layer_exempt
            and (index, "o_proj") not in protected_modules
        )
        attention.faquant_qk_matmul_quant = qk_matmul_quant and index not in qk_exempt
        attention.faquant_pv_matmul_quant = pv_matmul_quant and index not in pv_exempt
        base_quant_kwargs = attention.faquant_matmul_quant_kwargs
        attention.faquant_qk_matmul_quant_kwargs = (
            mxfp8_kwargs if index in qk_mxfp8 else base_quant_kwargs
        )
        attention.faquant_pv_matmul_quant_kwargs = (
            mxfp8_kwargs if index in pv_mxfp8 else base_quant_kwargs
        )
        if layer_exempt:
            # Exact transforms with nothing left to protect here, and each one
            # costs a bf16 round-trip.
            attention.faquant_qk_rotation = False
            attention.faquant_qk_smooth_scale = None
            attention.faquant_qk_key_offset = None
        elif post_rope_qk_rotation is not None:
            attention.faquant_qk_rotation = post_rope_qk_rotation
        attention.faquant_qk_rotation_block_size = (
            32 if index in qk_mxfp8 else None
        )
        if pv_normalizer_mode is not None:
            attention.faquant_pv_normalizer_mode = pv_normalizer_mode
        attention.faquant_k_matmul_qdq_cache = None
        attention.faquant_v_matmul_qdq_cache = None
        for key in attention.faquant_simulated_attention_stats:
            attention.faquant_simulated_attention_stats[key] = 0


def collect_qwen3_simulated_attention_stats(model: nn.Module) -> dict[str, int]:
    totals = new_simulated_attention_stats()
    for layer in model.model.layers:
        stats = getattr(layer.self_attn, "faquant_simulated_attention_stats", {})
        for key in totals:
            totals[key] += int(stats.get(key, 0))
    return totals


def _resolve_attention_implementation(
    config: ExperimentConfig, attn_implementation: str
) -> str:
    if (
        config.attention_kernel != "simulated"
        or attn_implementation != "flash_attention_2"
    ):
        return attn_implementation
    try:
        register_simulated_attention_backend()
    except (ImportError, AttributeError) as error:
        warnings.warn(
            "This Transformers version cannot register a compact custom attention "
            "mask backend; falling back to eager mask preparation while the "
            f"attention computation still uses the tiled simulator ({error}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return "eager"
    return SIMULATED_ATTENTION_IMPLEMENTATION


@torch.inference_mode()
def prepare_model(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    calibration_input_ids: torch.Tensor | None = None,
) -> nn.Module:
    """Apply mathematically invariant rotations, then requested fake quantization."""

    if config.rotation == "hadamard":
        apply_qwen3_global_rotation(
            model, seed=config.seed, online_hadamard=config.online_hadamard
        )
    exempt_modules = resolve_quant_exempt_modules(config, len(model.model.layers))
    if config.qwen_value_head_rotation:
        # Before HiSQ and GPTQ: both must see the basis the model will run in,
        # or they optimize their grids for a distribution that never occurs.
        _, pv_mxfp8_layers = resolve_attention_matmul_mxfp8_layers(
            config, len(model.model.layers)
        )
        model.faquant_value_head_rotation_stats = apply_qwen3_value_head_rotation(
            model,
            exempt_modules=exempt_modules,
            block_sizes={index: 32 for index in pv_mxfp8_layers},
        )
    if config.qwen_hisq_input_rotation and config.weight_quant == "rtn":
        model.faquant_hisq_rotation_stats = apply_qwen_hisq_input_rotation(
            model,
            block_size=config.qwen_hisq_rotation_block_size,
            down_proj_block_size=config.qwen_hisq_rotation_block_size_down_proj,
            seed=config.qwen_hisq_rotation_seed,
            exempt_modules=exempt_modules,
        )
    install_qwen3_attention_adapter(model, config)
    if config.qk_smooth_scales is not None:
        # Installed before GPTQ so calibration sees the model that will run. Safe
        # because the transform leaves the scores, and therefore every Linear's
        # input, unchanged up to one bf16 rounding.
        model.faquant_qk_smooth_metadata = apply_qwen3_qk_smooth_scales(
            model, config.qk_smooth_scales
        )
    if config.quant_target != "none" and config.weight_quant == "gptq":
        if calibration_input_ids is None:
            raise ValueError("GPTQ requires calibration_input_ids")
        model.faquant_gptq_stats = gptq_quantize_qwen3(
            model, calibration_input_ids, config
        )
        quantize_qwen3_linears(model, config, weights_prequantized=True)
    else:
        quantize_qwen3_linears(model, config)
    model.eval()
    return model


def load_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL,
    *,
    config: ExperimentConfig | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
) -> tuple[nn.Module, object]:
    """Load Qwen3 from a local path and prepare one FA-Quant experiment variant."""

    config = config or ExperimentConfig()
    attn_implementation = _resolve_attention_implementation(config, attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    if model.config.model_type != "qwen3":
        raise TypeError(
            f"FA-Quant currently expects Qwen3, got {model.config.model_type}"
        )
    calibration_input_ids = None
    if config.quant_target != "none" and config.weight_quant == "gptq":
        calibration_input_ids = calibration_tokens(
            tokenizer,
            nsamples=config.gptq_nsamples,
            seqlen=config.gptq_seqlen,
            seed=config.seed,
        )
    return (
        prepare_model(model, config, calibration_input_ids=calibration_input_ids),
        tokenizer,
    )
