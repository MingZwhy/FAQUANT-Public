from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QuantTarget = Literal["none", "attention", "ffn", "all"]
RotationMode = Literal["none", "hadamard"]
WeightQuantMethod = Literal["rtn", "gptq"]
AttentionKernel = Literal["native", "simulated"]
QuantFormat = Literal["int", "hif4", "mxfp8e4m3"]
PVNormalizerMode = Literal["unquantized", "quantized_same"]
MXFP8HessianMode = Literal["paired", "prefix_split"]

PV_NORMALIZER_MODES = ("unquantized", "quantized_same")
"""Where the simulated attention online-softmax denominator comes from.

``unquantized`` accumulates it from the raw tile probabilities while the PV
numerator uses the quantized ones. That mismatch is what every recorded result
was produced with, so it stays the default and the control. ``quantized_same``
is P-Reordering: the denominator is the row sum of the SAME quantized P that
feeds the numerator, exactly what appending a ones column to V computes in one
PV GEMM.
"""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by smoke tests and benchmark runs."""

    rotation: RotationMode = "none"
    online_hadamard: bool = True
    attention_kernel: AttentionKernel = "native"
    attention_input_quant: bool = True
    attention_output_quant: bool = True
    qk_matmul_quant: bool = False
    pv_matmul_quant: bool = False
    # Post-RoPE Q/K Hadamard. Until now this was only reachable at evaluation
    # time, which meant GPTQ calibrated against an attention core that did not
    # match deployment. Exposing it here lets the Hessians be built in the
    # geometry the model will actually run in.
    post_rope_qk_rotation: bool = False
    # "unquantized" reproduces every recorded result; "quantized_same" is
    # P-Reordering. Kept as an explicit control rather than a silent fix.
    pv_normalizer_mode: PVNormalizerMode = "unquantized"
    # Path to a frozen Smooth-QK calibration artifact. The scale is an exact
    # equivalence transform on post-RoPE Q/K, so it only matters when Q/K are
    # quantized; it is rejected otherwise so a run cannot silently claim it.
    qk_smooth_scales: str | None = None
    attention_query_chunk_size: int = 256
    attention_key_chunk_size: int = 128
    quant_target: QuantTarget = "none"
    quant_format: QuantFormat = "int"
    bits: int = 4
    weight_group_size: int | None = None
    activation_group_size: int | None = None
    symmetric: bool = True
    clip_ratio: float = 1.0
    seed: int = 0
    weight_quant: WeightQuantMethod = "rtn"
    gptq_nsamples: int = 128
    gptq_seqlen: int = 2048
    gptq_damp: float = 0.01
    # Fraction of calibration samples that use MXFP8 in selected attention
    # matmuls while their Hessians are accumulated. Final propagation always
    # uses the configured deployment format. Values below one shrink a noisy
    # MXFP8 Hessian toward the incumbent HiF4 estimate.
    gptq_mxfp8_hessian_fraction: float = 1.0
    # "paired" evaluates both formats on every calibration sample and forms a
    # low-variance weighted Hessian. "prefix_split" assigns an initial sample
    # fraction to MXFP8 and the remainder to HiF4.
    gptq_mxfp8_hessian_mode: MXFP8HessianMode = "paired"
    # Optional per-projection damping overrides, represented as
    # (projection_name, damp) pairs. Empty preserves the historical single
    # global value.
    gptq_damp_overrides: tuple[tuple[str, float], ...] = ()
    # Candidate multipliers on each HiF4 block maximum, searched per output row
    # by GPTQ's own error metric. Empty keeps the reference max-covering rule
    # that every existing checkpoint used.
    # Quantize columns in descending Hessian-diagonal order inside each 64-value
    # HiF4 group. Group membership, and therefore the deployed arithmetic, is
    # unchanged; only the order the solver works through a group changes.
    gptq_act_order_within_group: bool = False
    # Candidate offsets, in whole E6M2 grid steps, for the per-block HiF4 scale
    # GPTQ fits. Include 0 so the search can keep the reference scale, and both
    # signs: HiF4 rounds the scale to nearest rather than up, so roughly half
    # the blocks already clip, and a minority of those are better off with a
    # larger scale than any amount of shrinking can give them.
    gptq_hif4_scale_search_steps: tuple[int, ...] = ()
    # Build each projection group's Hessian only after the earlier groups in
    # the same layer are quantized. Layers are already sequential with respect
    # to each other; this makes the four groups inside a layer sequential too,
    # so gate/up see a quantized o_proj and down_proj sees quantized gate/up.
    # Costs one calibration forward pass per group instead of one per layer.
    gptq_sequential_groups: bool = False
    gptq_propagate_fake_activations: bool = True
    qad_capture_weight_metadata: bool = False
    qad_capture_latent_master: bool = False
    qwen_smoothquant: bool = False
    qwen_smoothquant_alpha: float = 0.9
    qwen_hisq_input_rotation: bool = False
    qwen_hisq_rotation_block_size: int = 1024
    # Widen the HiSQ block for down_proj only. Its input is 12288 wide, so a
    # 1024 block mixes a twelfth of it at a time, and the SwiGLU massive
    # activations survive that: the Hessian diagonal still spans 71x from max
    # to median where every other family sits between 3x and 9x. A 4096 block
    # takes it to 2.7x (results/act_order_potential/probe.json). Note that this
    # is not the same as the global block-4096 setting already ablated, which
    # also made the 4096-wide inputs full width and scored slightly worse.
    qwen_hisq_rotation_block_size_down_proj: int | None = None
    qwen_hisq_rotation_seed: int = 17
    # Fold a per-head Hadamard through v_proj rows and o_proj columns. Free at
    # runtime, and the only thing in the HiSQ recipe that touches the
    # head-internal geometry of V and of the attention output.
    qwen_value_head_rotation: bool = False
    # Decoder indices left entirely in BF16: no weight or activation
    # quantization, no HiSQ or value-head rotation, and no attention-core
    # quantization. This is whole-layer protection rather than individual
    # matmul protection, and one layer is 1/36 of the
    # matmul work at any sequence length.
    #
    # It has to be set before the checkpoint is built, not at evaluation time.
    # GPTQ walks the stack in order and feeds each layer the *quantized* output
    # of the one before it, so exempting a layer afterwards would leave every
    # later layer compensating for an error that no longer occurs.
    qwen_quant_exempt_layers: tuple[int, ...] = ()
    # Finer-grained protection: individual projections, written "<layer>.<proj>"
    # as in "5.down_proj". Whole-layer entries above expand into all seven of
    # these. Unlike a whole-layer exemption, this leaves the attention core
    # alone, because QK/PV quantization acts on a projection's output and is a
    # separate decision from how the projection itself is stored.
    qwen_quant_exempt_modules: tuple[str, ...] = ()
    # Attention-matmul protection used while GPTQ builds downstream Hessians.
    # These do not exempt q/k/v projection weights; they only leave selected
    # QK or PV matmuls in high precision. Empty preserves every existing build.
    qwen_qk_matmul_exempt_layers: tuple[int, ...] = ()
    qwen_pv_matmul_exempt_layers: tuple[int, ...] = ()
    # Selected attention matmuls use MXFP8 E4M3 instead of HiF4. These are the
    # deployable 8-bit protection knobs; they remain quantized and therefore
    # preserve the error structure better than a BF16 exemption.
    qwen_qk_matmul_mxfp8_layers: tuple[int, ...] = ()
    qwen_pv_matmul_mxfp8_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        default_group_size = 64 if self.quant_format == "hif4" else 128
        if self.weight_group_size is None:
            object.__setattr__(self, "weight_group_size", default_group_size)
        if self.activation_group_size is None:
            object.__setattr__(self, "activation_group_size", default_group_size)
        if not 2 <= self.bits <= 16:
            raise ValueError("bits must be in [2, 16]")
        if self.pv_normalizer_mode not in PV_NORMALIZER_MODES:
            raise ValueError(
                f"pv_normalizer_mode must be one of {PV_NORMALIZER_MODES}"
            )
        if self.pv_normalizer_mode == "quantized_same" and (
            self.attention_kernel != "simulated"
        ):
            raise ValueError(
                "pv_normalizer_mode='quantized_same' requires the simulated kernel"
            )
        if self.qk_smooth_scales is not None and not (
            self.qk_matmul_quant or self.attention_input_quant
        ):
            raise ValueError(
                "qk_smooth_scales only affects quantized Q/K; enable "
                "qk_matmul_quant or attention_input_quant"
            )
        if self.qwen_value_head_rotation and self.rotation == "hadamard":
            raise ValueError(
                "rotation='hadamard' already folds the per-head V/output "
                "Hadamard; applying it twice is the identity"
            )
        if self.weight_group_size == 0 or self.activation_group_size == 0:
            raise ValueError("group sizes must be positive or -1")
        if not 0.0 < self.clip_ratio <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        if self.gptq_nsamples <= 0 or self.gptq_seqlen <= 0:
            raise ValueError("GPTQ calibration dimensions must be positive")
        if self.gptq_damp <= 0:
            raise ValueError("gptq_damp must be positive")
        if not 0.0 <= self.gptq_mxfp8_hessian_fraction <= 1.0:
            raise ValueError("gptq_mxfp8_hessian_fraction must be in [0, 1]")
        if self.gptq_mxfp8_hessian_mode not in ("paired", "prefix_split"):
            raise ValueError(
                "gptq_mxfp8_hessian_mode must be 'paired' or 'prefix_split'"
            )
        valid_damp_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        seen_damp_modules: set[str] = set()
        for projection, damp in self.gptq_damp_overrides:
            if projection not in valid_damp_modules:
                raise ValueError(
                    f"gptq_damp_overrides names unknown projection {projection!r}"
                )
            if projection in seen_damp_modules:
                raise ValueError(
                    f"gptq_damp_overrides lists {projection!r} twice"
                )
            if damp <= 0:
                raise ValueError("GPTQ damping overrides must be positive")
            seen_damp_modules.add(projection)
        # The metadata is the HiF4 grid a QAT student re-quantizes against, so
        # it only means anything in HiF4.  It used to be restricted to GPTQ as
        # well, because GPTQ was the only path that recorded it; RTN now fits
        # the same grid in replace_linear_with_fake_quant, which is what lets a
        # no-PTQ student exist at all.
        if self.qad_capture_weight_metadata and self.quant_format != "hif4":
            raise ValueError(
                "QAD weight metadata capture requires HiF4"
            )
        if self.qad_capture_latent_master and not self.qad_capture_weight_metadata:
            raise ValueError(
                "QAD latent-master capture requires fixed HiF4 metadata"
            )
        if not 0.0 <= self.qwen_smoothquant_alpha <= 1.0:
            raise ValueError("qwen_smoothquant_alpha must be in [0, 1]")
        if self.qwen_smoothquant:
            if (
                self.quant_format != "hif4"
                or self.quant_target != "all"
                or self.weight_quant != "gptq"
            ):
                raise ValueError(
                    "Qwen SmoothQuant ablation requires all-linear HiF4 GPTQ"
                )
            if self.rotation != "none":
                raise ValueError(
                    "Qwen SmoothQuant is currently isolated to no-rotation "
                    "experiments"
                )
        if self.qwen_hisq_input_rotation:
            if (
                self.quant_format != "hif4"
                or self.quant_target != "all"
            ):
                raise ValueError(
                    "Qwen HiSQ rotation ablation requires all-linear HiF4 quantization"
                )
            if self.rotation != "none":
                raise ValueError(
                    "Qwen per-linear HiSQ rotation and global residual rotation "
                    "are separate ablations"
                )
            if self.qwen_smoothquant:
                raise ValueError(
                    "Qwen HiSQ rotation is currently isolated from SmoothQuant"
                )
            block_size = self.qwen_hisq_rotation_block_size
            if block_size <= 0 or block_size & (block_size - 1):
                raise ValueError(
                    "qwen_hisq_rotation_block_size must be a positive power of two"
                )
            override = self.qwen_hisq_rotation_block_size_down_proj
            if override is not None and (override <= 0 or override & (override - 1)):
                raise ValueError(
                    "qwen_hisq_rotation_block_size_down_proj must be a positive "
                    "power of two"
                )
        if self.attention_query_chunk_size <= 0 or self.attention_key_chunk_size <= 0:
            raise ValueError("attention chunk sizes must be positive")
        if self.attention_kernel != "simulated" and (
            self.qk_matmul_quant or self.pv_matmul_quant
        ):
            raise ValueError(
                "QK/PV matmul quantization requires attention_kernel='simulated'"
            )
        if self.quant_format == "hif4":
            if self.bits != 4:
                raise ValueError("HiF4 requires bits=4")
            if self.weight_group_size != 64 or self.activation_group_size != 64:
                raise ValueError("HiF4 requires 64-value weight and activation groups")
            if not self.symmetric:
                raise ValueError(
                    "HiF4 uses sign-magnitude encoding and cannot be asymmetric"
                )
            if self.clip_ratio != 1.0:
                raise ValueError("HiF4 direct-cast does not support clip_ratio != 1")
QWEN_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
QWEN_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
QWEN_PROJECTIONS = QWEN_ATTENTION_PROJECTIONS + QWEN_MLP_PROJECTIONS


def resolve_quant_exempt_layers(config: object, depth: int) -> frozenset[int]:
    """Validate ``qwen_quant_exempt_layers`` against an actual model depth.

    Every consumer resolves through here so that a typo becomes an error at the
    point the model is built rather than a layer that quietly stays quantized.
    """

    raw = tuple(getattr(config, "qwen_quant_exempt_layers", ()) or ())
    exempt = frozenset(int(index) for index in raw)
    out_of_range = sorted(index for index in exempt if not 0 <= index < depth)
    if out_of_range:
        raise ValueError(
            f"quant exempt layers {out_of_range} outside 0..{depth - 1}"
        )
    return exempt


def resolve_quant_exempt_modules(
    config: object, depth: int
) -> frozenset[tuple[int, str]]:
    """Expand both exemption settings into explicit ``(layer, projection)`` pairs.

    Whole-layer entries become all seven projections of that layer, so the two
    settings compose and every consumer only has to understand one shape.
    """

    exempt: set[tuple[int, str]] = set()
    for index in resolve_quant_exempt_layers(config, depth):
        exempt.update((index, name) for name in QWEN_PROJECTIONS)

    for spec in tuple(getattr(config, "qwen_quant_exempt_modules", ()) or ()):
        text = str(spec).strip()
        layer_text, _, projection = text.partition(".")
        if not projection:
            raise ValueError(
                f"quant exempt module {text!r} must be written '<layer>.<projection>'"
            )
        try:
            index = int(layer_text)
        except ValueError as error:
            raise ValueError(
                f"quant exempt module {text!r} has a non-integer layer"
            ) from error
        if not 0 <= index < depth:
            raise ValueError(
                f"quant exempt module {text!r} is outside layers 0..{depth - 1}"
            )
        if projection not in QWEN_PROJECTIONS:
            raise ValueError(
                f"quant exempt module {text!r} names no Qwen projection; "
                f"expected one of {', '.join(QWEN_PROJECTIONS)}"
            )
        exempt.add((index, projection))
    return frozenset(exempt)


def resolve_attention_matmul_exempt_layers(
    config: object,
    depth: int,
) -> tuple[frozenset[int], frozenset[int]]:
    """Validate build-time QK/PV matmul exemptions against model depth."""

    resolved: list[frozenset[int]] = []
    for label, attribute in (
        ("qk", "qwen_qk_matmul_exempt_layers"),
        ("pv", "qwen_pv_matmul_exempt_layers"),
    ):
        raw = tuple(getattr(config, attribute, ()) or ())
        values = frozenset(int(index) for index in raw)
        out_of_range = sorted(index for index in values if not 0 <= index < depth)
        if out_of_range:
            raise ValueError(
                f"{label} matmul exempt layers {out_of_range} outside "
                f"0..{depth - 1}"
            )
        resolved.append(values)
    return resolved[0], resolved[1]


def resolve_attention_matmul_mxfp8_layers(
    config: object,
    depth: int,
) -> tuple[frozenset[int], frozenset[int]]:
    """Validate QK/PV matmuls assigned to MXFP8 protection."""

    resolved: list[frozenset[int]] = []
    for label, attribute in (
        ("qk", "qwen_qk_matmul_mxfp8_layers"),
        ("pv", "qwen_pv_matmul_mxfp8_layers"),
    ):
        raw = tuple(getattr(config, attribute, ()) or ())
        values = frozenset(int(index) for index in raw)
        out_of_range = sorted(index for index in values if not 0 <= index < depth)
        if out_of_range:
            raise ValueError(
                f"{label} MXFP8 layers {out_of_range} outside 0..{depth - 1}"
            )
        resolved.append(values)
    return resolved[0], resolved[1]


def qwen_projection_path(projection: str) -> str:
    """Return the dotted path of a projection inside one decoder layer."""

    if projection in QWEN_ATTENTION_PROJECTIONS:
        return f"self_attn.{projection}"
    if projection in QWEN_MLP_PROJECTIONS:
        return f"mlp.{projection}"
    raise ValueError(f"unknown Qwen projection {projection!r}")
