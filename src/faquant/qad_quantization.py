from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F

from .hif4 import (
    CompactHiF4Parameters,
    HIF4_BLOCK_SIZE,
    fit_hif4_compact_parameters,
    quantize_hif4_with_compact_parameters,
)
from .quantization import FakeQuantLinear, fake_quantize


def ste_fake_quantize_hif4(x: torch.Tensor) -> torch.Tensor:
    """Apply the exact HiF4 direct cast with an identity STE backward."""

    quantized = fake_quantize(
        x,
        quant_format="hif4",
        bits=4,
        group_size=HIF4_BLOCK_SIZE,
        symmetric=True,
        clip_ratio=1.0,
    )
    return x + (quantized - x).detach()


@dataclass(frozen=True)
class QATConversionStats:
    converted: int
    quantized_forward_calls: int


@dataclass(frozen=True)
class TrainingTensorStats:
    parameters_materialized: int
    buffers_materialized: int


@dataclass(frozen=True)
class TrainableScopeStats:
    trainable_tensors: int
    trainable_parameters: int
    frozen_parameters: int


@dataclass(frozen=True)
class QATMasterGridStats:
    linears: int
    elements: int
    master_off_grid_elements: int
    master_off_grid_fraction: float
    master_to_grid_relative_l2: float


@dataclass(frozen=True)
class QATLearningRateScaleStats:
    linears: int
    reference_code_step: float
    minimum_factor: float
    median_factor: float
    maximum_factor: float


@dataclass(frozen=True)
class QATInputScaleStats:
    linears: int
    elements: int


@dataclass(frozen=True)
class QATActivationClipStats:
    linears: int
    elements: int


@dataclass(frozen=True)
class QATActivationCompandStats:
    linears: int
    elements: int


class HiF4QATLinear(nn.Module):
    """Trainable HiF4 W4A4 linear with exact-reference QDQ and STE gradients.

    The module is deliberately separate from :class:`FakeQuantLinear`, whose
    frozen, pre-quantized weights are the inference/PTQ contract. QAD starts
    from such a module (normally after GPTQ), promotes its dequantized weight
    to a trainable master parameter, and re-applies the same HiF4 direct cast
    on every forward.
    """

    faquant_quantizes_input = True

    def __init__(self, module: FakeQuantLinear) -> None:
        super().__init__()
        if module.quant_format != "hif4" or module.bits != 4:
            raise ValueError("HiF4QATLinear requires a 4-bit HiF4 FakeQuantLinear")
        if module.activation_group_size != HIF4_BLOCK_SIZE:
            raise ValueError("HiF4QATLinear requires 64-value activation groups")

        self.in_features = module.in_features
        self.out_features = module.out_features
        self.quant_format = module.quant_format
        self.bits = module.bits
        self.weight_group_size = HIF4_BLOCK_SIZE
        self.activation_group_size = module.activation_group_size
        self.symmetric = module.symmetric
        self.clip_ratio = module.clip_ratio
        self.online_hadamard = module.online_hadamard
        self.online_hadamard_block_size = module.online_hadamard_block_size
        self.input_rotation_block_size = module.input_rotation_block_size
        self.metadata_mode = "fixed"

        latent_master = getattr(module, "faquant_hif4_qat_master", None)
        initial_weight = module.weight if latent_master is None else latent_master
        self.weight = nn.Parameter(
            initial_weight.detach().to(module.weight.device).clone(),
            requires_grad=True,
        )
        if module.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(module.bias.detach().clone(), requires_grad=True)

        for name in (
            "input_scale",
            "input_rotation_signs",
            "input_rotation_permutation",
            "activation_min",
            "activation_max",
            "activation_compand_log_scale",
        ):
            value = getattr(module, name, None)
            self.register_buffer(
                name,
                None if value is None else value.detach().clone(),
                persistent=True,
            )
        self.register_buffer(
            "quantized_forward_calls",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        parameters = getattr(module, "faquant_hif4_qat_parameters", None)
        if parameters is None:
            raise ValueError(
                "HiF4QATLinear requires fixed GPTQ HiF4 metadata; prepare the "
                "student with qad_capture_weight_metadata=True"
            )
        for name, value in (
            ("weight_hif4_scale", parameters.scale),
            ("weight_hif4_reciprocal", parameters.reciprocal),
            (
                "weight_hif4_scale_lv2_exponent",
                parameters.scale_lv2_exponent,
            ),
            (
                "weight_hif4_scale_lv3_exponent",
                parameters.scale_lv3_exponent,
            ),
        ):
            self.register_buffer(name, value.detach().clone(), persistent=True)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            x = x * self.input_scale.to(device=x.device, dtype=x.dtype)
        if self.input_rotation_signs is not None:
            from .rotation import chunked_block_hadamard_transform

            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            chunks: list[torch.Tensor] = []
            signs = self.input_rotation_signs.to(
                device=x.device, dtype=torch.float32
            )
            permutation = self.input_rotation_permutation.to(device=x.device)
            for start in range(0, flat.shape[0], 256):
                stop = min(start + 256, flat.shape[0])
                work = flat[start:stop].float() * signs
                work = work.index_select(-1, permutation)
                if self.input_rotation_block_size is not None:
                    work = chunked_block_hadamard_transform(
                        work,
                        block_size=self.input_rotation_block_size,
                        chunk_rows=256,
                    )
                chunks.append(work.to(x.dtype))
            x = torch.cat(chunks, dim=0).reshape(shape)
        if self.activation_min is not None or self.activation_max is not None:
            x = x.clamp(min=self.activation_min, max=self.activation_max)
        return x

    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.online_hadamard:
            return ste_fake_quantize_hif4(x)

        from .rotation import (
            chunked_block_hadamard_transform,
            generalized_hadamard_transform,
        )

        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        chunks: list[torch.Tensor] = []
        for start in range(0, flat.shape[0], 256):
            stop = min(start + 256, flat.shape[0])
            if self.online_hadamard_block_size is None:
                rotated = generalized_hadamard_transform(flat[start:stop].float())
            else:
                rotated = chunked_block_hadamard_transform(
                    flat[start:stop].float(),
                    block_size=self.online_hadamard_block_size,
                    chunk_rows=256,
                )
            chunks.append(ste_fake_quantize_hif4(rotated.to(x.dtype)))
        return torch.cat(chunks, dim=0).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.quantized_forward_calls.add_(1)
        x = self._preprocess_input(x)
        compand_scale = None
        if self.activation_compand_log_scale is not None:
            if self.online_hadamard:
                raise RuntimeError(
                    "activation companding is only defined without online Hadamard"
                )
            log_scale = self.activation_compand_log_scale
            if log_scale.numel() != self.in_features:
                if self.in_features % log_scale.numel():
                    raise RuntimeError("invalid activation companding shape")
                log_scale = log_scale.repeat_interleave(
                    self.in_features // log_scale.numel()
                )
            compand_scale = log_scale.clamp(-2.079, 2.079).exp().to(
                device=x.device, dtype=x.dtype
            )
            x = x * compand_scale
        x_quantized = self._quantize_input(x)
        if compand_scale is not None:
            x_quantized = x_quantized / compand_scale
        quantized = self._quantize_weight()
        # Both forms pass gradient straight through to the master weight, but only
        # this one is forward-exact: `self.weight - self.weight.detach()` is
        # elementwise zero, so the sum is `quantized` bit for bit, whereas
        # `self.weight + (quantized - self.weight)` relies on the subtraction
        # being exact.  In bfloat16 that holds only while the two share an
        # exponent, and QAD pushes master weights off their grid point until it
        # stops holding: at step 750 of the 1250-step seed-2718 run, 44 elements
        # across three early-layer projections -- the ones where HiF4's code
        # spacing is finest -- ended up one ULP away.  Amplified through 36
        # layers that showed up as a 2.5 logit gap between the QAT forward and
        # the module `to_inference` builds, tripping the exact-parity check in
        # scripts/eval_qwen_hif4_qad.py and costing that checkpoint both
        # readouts.  The deployed side was always the correct one; it is the
        # training forward that was off.
        weight_quantized = quantized + (self.weight - self.weight.detach())
        return F.linear(x_quantized, weight_quantized, self.bias)

    def _weight_parameters(self) -> CompactHiF4Parameters:
        return CompactHiF4Parameters(
            scale=self.weight_hif4_scale,
            reciprocal=self.weight_hif4_reciprocal,
            scale_lv2_exponent=self.weight_hif4_scale_lv2_exponent,
            scale_lv3_exponent=self.weight_hif4_scale_lv3_exponent,
        )

    def _quantize_weight(self) -> torch.Tensor:
        shape = self.weight.shape
        grouped = self.weight.reshape(
            shape[0], shape[1] // HIF4_BLOCK_SIZE, HIF4_BLOCK_SIZE
        )
        parameters = self._weight_parameters()
        if self.metadata_mode == "dynamic":
            parameters = fit_hif4_compact_parameters(grouped)
        elif self.metadata_mode != "fixed":
            raise ValueError(f"unsupported HiF4 QAT metadata mode={self.metadata_mode!r}")
        return quantize_hif4_with_compact_parameters(
            grouped,
            parameters,
        ).reshape_as(self.weight)

    def to_inference(self) -> FakeQuantLinear:
        """Materialize the current master weights as a frozen inference module."""

        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self._quantize_weight())
            if self.bias is not None:
                linear.bias.copy_(self.bias)

        # Recreate the metadata contract consumed by FakeQuantLinear.
        if self.input_scale is not None:
            linear.faquant_input_scale = self.input_scale
        if self.input_rotation_signs is not None:
            linear.faquant_input_rotation_signs = self.input_rotation_signs
            linear.faquant_input_rotation_permutation = (
                self.input_rotation_permutation
            )
            linear.faquant_input_rotation_block_size = (
                self.input_rotation_block_size
            )
        if self.activation_min is not None:
            linear.faquant_activation_min = self.activation_min
        if self.activation_max is not None:
            linear.faquant_activation_max = self.activation_max
        if self.activation_compand_log_scale is not None:
            linear.faquant_activation_compand_log_scale = (
                self.activation_compand_log_scale
            )

        return FakeQuantLinear(
            linear,
            quant_format="hif4",
            bits=4,
            weight_group_size=HIF4_BLOCK_SIZE,
            activation_group_size=HIF4_BLOCK_SIZE,
            symmetric=True,
            clip_ratio=1.0,
            quantize_weight=False,
            online_hadamard=self.online_hadamard,
            online_hadamard_block_size=self.online_hadamard_block_size,
            input_scale=self.input_scale,
            input_rotation_signs=self.input_rotation_signs,
            input_rotation_permutation=self.input_rotation_permutation,
            input_rotation_block_size=self.input_rotation_block_size,
            activation_min=self.activation_min,
            activation_max=self.activation_max,
            activation_compand_log_scale=self.activation_compand_log_scale,
        )

    def to_float_master(self) -> nn.Linear:
        """Recover the trainable floating-point master linear.

        This is intentionally narrower than ``to_inference``: it is used to
        take an *unrotated* QAD student back to an ordinary Qwen graph before a
        new global rotation and GPTQ pass. Quantization-time activation
        transforms cannot be silently retained in a plain ``nn.Linear``.
        """

        if self.online_hadamard:
            raise ValueError(
                "floating-master export requires an unrotated QAD checkpoint"
            )
        if any(
            value is not None
            for value in (
                self.input_scale,
                self.input_rotation_signs,
                self.activation_min,
                self.activation_max,
                self.activation_compand_log_scale,
            )
        ):
            raise ValueError(
                "floating-master export does not support activation preprocessing"
            )
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def to_float_deployed(self) -> nn.Linear:
        """Recover an ordinary linear containing the deployed QAD weight.

        Unlike ``to_float_master``, this first applies the checkpoint's fixed
        HiF4 metadata. It is the faithful source for changing the quantization
        basis after QAD, because it preserves the weight values that produced
        the evaluated QAD checkpoint.
        """

        if self.online_hadamard:
            raise ValueError(
                "floating deployed export requires an unrotated QAD checkpoint"
            )
        if any(
            value is not None
            for value in (
                self.input_scale,
                self.input_rotation_signs,
                self.activation_min,
                self.activation_max,
                self.activation_compand_log_scale,
            )
        ):
            raise ValueError(
                "floating deployed export does not support activation preprocessing"
            )
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self._quantize_weight())
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear


def _replace_modules(
    module: nn.Module,
    *,
    source_type: type[nn.Module],
    convert: object,
) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, source_type):
            setattr(module, name, convert(child))
            count += 1
        else:
            count += _replace_modules(
                child,
                source_type=source_type,
                convert=convert,
            )
    return count


def enable_hif4_qat(model: nn.Module) -> QATConversionStats:
    """Replace every inference HiF4 linear with a trainable STE-QAT linear."""

    converted = _replace_modules(
        model,
        source_type=FakeQuantLinear,
        convert=HiF4QATLinear,
    )
    if converted == 0:
        raise ValueError("model contains no FakeQuantLinear modules to convert")
    return QATConversionStats(converted=converted, quantized_forward_calls=0)


def convert_hif4_qat_to_inference(model: nn.Module) -> QATConversionStats:
    """Freeze QAT master weights into the existing inference fake-quant path."""

    calls = sum(
        int(module.quantized_forward_calls.item())
        for module in model.modules()
        if isinstance(module, HiF4QATLinear)
    )
    converted = _replace_modules(
        model,
        source_type=HiF4QATLinear,
        convert=lambda module: module.to_inference(),
    )
    if converted == 0:
        raise ValueError("model contains no HiF4QATLinear modules to convert")
    return QATConversionStats(
        converted=converted,
        quantized_forward_calls=calls,
    )


def convert_hif4_qat_to_float_master(model: nn.Module) -> QATConversionStats:
    """Replace unrotated QAT linears with their unquantized master weights."""

    calls = sum(
        int(module.quantized_forward_calls.item())
        for module in model.modules()
        if isinstance(module, HiF4QATLinear)
    )
    converted = _replace_modules(
        model,
        source_type=HiF4QATLinear,
        convert=lambda module: module.to_float_master(),
    )
    if converted == 0:
        raise ValueError("model contains no HiF4QATLinear modules to convert")
    return QATConversionStats(
        converted=converted,
        quantized_forward_calls=calls,
    )


def convert_hif4_qat_to_float_deployed(model: nn.Module) -> QATConversionStats:
    """Replace unrotated QAT linears with deployed dequantized HiF4 weights."""

    calls = sum(
        int(module.quantized_forward_calls.item())
        for module in model.modules()
        if isinstance(module, HiF4QATLinear)
    )
    converted = _replace_modules(
        model,
        source_type=HiF4QATLinear,
        convert=lambda module: module.to_float_deployed(),
    )
    if converted == 0:
        raise ValueError("model contains no HiF4QATLinear modules to convert")
    return QATConversionStats(
        converted=converted,
        quantized_forward_calls=calls,
    )


def collect_hif4_qat_stats(model: nn.Module) -> QATConversionStats:
    modules = [
        module for module in model.modules() if isinstance(module, HiF4QATLinear)
    ]
    return QATConversionStats(
        converted=len(modules),
        quantized_forward_calls=sum(
            int(module.quantized_forward_calls.item()) for module in modules
        ),
    )


def promote_hif4_activation_clips(model: nn.Module) -> QATActivationClipStats:
    """Materialize no-op per-channel clipping buffers on every QAT Linear.

    ``None`` buffers are absent from a PyTorch state dict.  A checkpoint that
    calibrates only a subset of linears still needs a fixed, strictly
    reloadable structure, so unselected modules receive ``[-inf, +inf]``.
    """

    modules = [
        module for module in model.modules() if isinstance(module, HiF4QATLinear)
    ]
    if not modules:
        raise ValueError("model contains no HiF4QATLinear modules")
    elements = 0
    for module in modules:
        shape = (module.in_features,)
        for name, fill in (("activation_min", -float("inf")), ("activation_max", float("inf"))):
            value = getattr(module, name)
            if value is None:
                value = torch.full(
                    shape,
                    fill,
                    device=module.weight.device,
                    dtype=torch.float32,
                )
                setattr(module, name, value)
            elif tuple(value.shape) not in ((), shape):
                raise ValueError(
                    f"{name} must be scalar or have shape {shape}, got "
                    f"{tuple(value.shape)}"
                )
        elements += module.in_features
    return QATActivationClipStats(linears=len(modules), elements=elements)


def set_hif4_qat_metadata_mode(model: nn.Module, mode: str) -> int:
    """Select fixed-GPTQ or dynamically re-fitted HiF4 metadata for QAT."""

    if mode not in ("fixed", "dynamic"):
        raise ValueError("HiF4 QAT metadata mode must be 'fixed' or 'dynamic'")
    modules = [
        module for module in model.modules() if isinstance(module, HiF4QATLinear)
    ]
    if not modules:
        raise ValueError("model contains no HiF4QATLinear modules")
    for module in modules:
        module.metadata_mode = mode
    return len(modules)


def configure_qat_trainable_scope(
    model: nn.Module,
    scope: str,
    *,
    minimum_transformer_layer: int = 0,
    activation_companding_group_size: int = 1,
) -> TrainableScopeStats:
    """Configure either legacy all-parameter or quantized-linear-only QAD."""

    if scope not in (
        "all",
        "qat-linears",
        "bf16-only",
        "norms-only",
        "input-scales-only",
        "activation-companding-only",
        "residual-affine-only",
        "residual-affine-fp32-only",
    ):
        raise ValueError(
            "QAT trainable scope must be 'all', 'qat-linears', 'bf16-only', "
            "'norms-only', 'input-scales-only', or "
            "'activation-companding-only', 'residual-affine-only', or "
            "'residual-affine-fp32-only'"
        )
    if minimum_transformer_layer < 0:
        raise ValueError("minimum transformer layer must be non-negative")
    if scope != "qat-linears" and minimum_transformer_layer:
        raise ValueError("minimum transformer layer requires qat-linears scope")
    if scope == "qat-linears":
        model.requires_grad_(False)
        modules = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, HiF4QATLinear)
        ]
        if not modules:
            raise ValueError("model contains no HiF4QATLinear modules")
        for name, module in modules:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            layer = None if match is None else int(match.group(1))
            if minimum_transformer_layer and layer is None:
                raise ValueError(
                    f"cannot identify transformer layer for QAT module {name!r}"
                )
            module.weight.requires_grad_(
                layer is None or layer >= minimum_transformer_layer
            )
    elif scope == "bf16-only":
        model.requires_grad_(True)
        modules = [
            module for module in model.modules() if isinstance(module, HiF4QATLinear)
        ]
        if not modules:
            raise ValueError("model contains no HiF4QATLinear modules")
        for module in modules:
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
    elif scope == "norms-only":
        model.requires_grad_(False)
        norm_modules = [
            module
            for module in model.modules()
            if isinstance(module, nn.LayerNorm)
            or module.__class__.__name__.lower().endswith("rmsnorm")
        ]
        if not norm_modules:
            raise ValueError("model contains no LayerNorm or RMSNorm modules")
        for module in norm_modules:
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
    elif scope == "input-scales-only":
        model.requires_grad_(False)
        promote_hif4_input_scales(model)
        for module in model.modules():
            if isinstance(module, HiF4QATLinear):
                module.input_scale.requires_grad_(True)
    elif scope == "activation-companding-only":
        model.requires_grad_(False)
        promote_hif4_activation_companding(
            model, group_size=activation_companding_group_size
        )
    elif scope in ("residual-affine-only", "residual-affine-fp32-only"):
        model.requires_grad_(False)
        from .residual_affine import install_qwen3_residual_affine

        install_qwen3_residual_affine(
            model,
            trainable=True,
            multiply_mode=(
                "fp32" if scope == "residual-affine-fp32-only" else "legacy"
            ),
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return TrainableScopeStats(
        trainable_tensors=len(trainable),
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        frozen_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    )


def promote_hif4_input_scales(model: nn.Module) -> QATInputScaleStats:
    """Promote per-channel Linear input gains to trainable FP32 parameters.

    Missing gains are initialized to one, so promotion has exact step-0
    parity. Existing static SmoothQuant gains are retained.
    """

    linears = elements = 0
    for module in model.modules():
        if not isinstance(module, HiF4QATLinear):
            continue
        existing_parameter = module._parameters.get("input_scale")
        if existing_parameter is not None:
            existing_parameter.requires_grad_(True)
            linears += 1
            elements += existing_parameter.numel()
            continue
        current = module._buffers.pop("input_scale", None)
        if current is None:
            current = torch.ones(
                module.in_features,
                device=module.weight.device,
                dtype=torch.float32,
            )
        module.register_parameter(
            "input_scale",
            nn.Parameter(current.detach().float().clone(), requires_grad=True),
        )
        linears += 1
        elements += current.numel()
    if not linears:
        raise ValueError("model contains no HiF4QATLinear modules")
    return QATInputScaleStats(linears=linears, elements=elements)


def promote_hif4_activation_companding(
    model: nn.Module,
    *,
    group_size: int = 1,
) -> QATActivationCompandStats:
    """Promote reversible post-rotation A4 companders to FP32 parameters.

    The deployed operation is ``Q(x * exp(log_scale)) / exp(log_scale)``.
    Zero initialization is therefore exact step-0 parity, while allowing the
    optimizer to redistribute dynamic range inside each 64-value HiF4 block
    without changing any weight or the unquantized Linear function.
    """

    if group_size <= 0:
        raise ValueError("activation companding group size must be positive")
    linears = elements = 0
    for module in model.modules():
        if not isinstance(module, HiF4QATLinear):
            continue
        if module.online_hadamard:
            raise ValueError(
                "activation companding requires post-HiSQ direct A4 quantization"
            )
        if module.in_features % group_size:
            raise ValueError(
                f"in_features={module.in_features} is not divisible by "
                f"activation companding group_size={group_size}"
            )
        parameter_elements = module.in_features // group_size
        existing = module._parameters.get("activation_compand_log_scale")
        if existing is not None:
            if existing.numel() != parameter_elements:
                raise ValueError("existing activation companding shape mismatch")
            existing.requires_grad_(True)
        else:
            current = module._buffers.pop("activation_compand_log_scale", None)
            if current is None:
                current = torch.zeros(
                    parameter_elements,
                    device=module.weight.device,
                    dtype=torch.float32,
                )
            module.register_parameter(
                "activation_compand_log_scale",
                nn.Parameter(current.detach().float().clone(), requires_grad=True),
            )
        linears += 1
        elements += parameter_elements
    if not linears:
        raise ValueError("model contains no HiF4QATLinear modules")
    return QATActivationCompandStats(linears=linears, elements=elements)


@torch.no_grad()
def hif4_code_step_parameter_groups(
    model: nn.Module,
    *,
    base_lr: float,
    minimum_factor: float = 0.125,
    maximum_factor: float = 8.0,
) -> tuple[list[dict[str, object]], QATLearningRateScaleStats]:
    """Build per-linear LR groups normalized by each HiF4 code step.

    Adam's early update magnitude is nearly independent of gradient scale.  A
    single absolute LR therefore makes linears with fine HiF4 grids cross code
    boundaries much earlier than linears with coarse grids.  Scaling each
    parameter group's LR by its median fixed-metadata dequantization step makes
    the optimizer displacement closer to uniform in deployed code space.
    """

    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if not 0 < minimum_factor <= maximum_factor:
        raise ValueError("invalid HiF4 learning-rate factor bounds")
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, HiF4QATLinear) and module.weight.requires_grad
    ]
    if not modules:
        raise ValueError("model contains no trainable HiF4QATLinear modules")

    code_steps: list[torch.Tensor] = []
    for _, module in modules:
        parameters = module._weight_parameters()
        dequant_scale = (
            parameters.scale.float()
            * torch.exp2(parameters.scale_lv2_exponent.float())
            * torch.exp2(parameters.scale_lv3_exponent.float())
        )
        # Adjacent HiF4 mantissas differ by 0.25. The common factor cancels in
        # normalization, but retaining it makes the receipt physically clear.
        code_steps.append(0.25 * dequant_scale.median())
    reference = torch.stack(
        [value.to(device=code_steps[0].device) for value in code_steps]
    ).median()
    if not torch.isfinite(reference) or reference <= 0:
        raise RuntimeError("invalid median HiF4 code step")

    groups: list[dict[str, object]] = []
    factors: list[float] = []
    for (name, module), code_step in zip(modules, code_steps, strict=True):
        factor = float((code_step / reference.to(code_step.device)).item())
        factor = min(max(factor, minimum_factor), maximum_factor)
        factors.append(factor)
        groups.append(
            {
                "params": [module.weight],
                "lr": base_lr * factor,
                "faquant_module": name,
                "faquant_lr_factor": factor,
            }
        )
    sorted_factors = sorted(factors)
    return groups, QATLearningRateScaleStats(
        linears=len(groups),
        reference_code_step=float(reference.item()),
        minimum_factor=sorted_factors[0],
        median_factor=sorted_factors[len(sorted_factors) // 2],
        maximum_factor=sorted_factors[-1],
    )


@torch.no_grad()
def collect_hif4_qat_master_grid_stats(model: nn.Module) -> QATMasterGridStats:
    """Summarize latent-master residuals without retaining full weight copies."""

    linears = elements = off_grid = 0
    residual_sq = grid_sq = 0.0
    for module in model.modules():
        if not isinstance(module, HiF4QATLinear):
            continue
        quantized = module._quantize_weight()
        difference = module.weight.float() - quantized.float()
        linears += 1
        elements += module.weight.numel()
        off_grid += int(module.weight.ne(quantized).sum().item())
        residual_sq += float(difference.square().sum().item())
        grid_sq += float(quantized.float().square().sum().item())
        del quantized, difference
    if not linears:
        raise ValueError("model contains no HiF4QATLinear modules")
    return QATMasterGridStats(
        linears=linears,
        elements=elements,
        master_off_grid_elements=off_grid,
        master_off_grid_fraction=off_grid / elements,
        master_to_grid_relative_l2=(residual_sq / max(grid_sq, 1e-30)) ** 0.5,
    )


@torch.no_grad()
def scale_hif4_qat_master_residual(model: nn.Module, scale: float) -> int:
    """Scale latent residuals toward deployed codepoints with exact parity."""

    if not 0.0 <= scale <= 1.0:
        raise ValueError("latent residual scale must be in [0, 1]")
    converted = 0
    for module in model.modules():
        if not isinstance(module, HiF4QATLinear):
            continue
        if module.metadata_mode != "fixed":
            raise ValueError("latent residual scaling requires fixed metadata")
        deployed = module._quantize_weight()
        module.weight.copy_(deployed + scale * (module.weight - deployed))
        actual = module._quantize_weight()
        if not torch.equal(actual, deployed):
            mismatches = int(actual.ne(deployed).sum().item())
            raise RuntimeError(
                f"latent residual scaling changed {mismatches} deployed values"
            )
        converted += 1
    if not converted:
        raise ValueError("model contains no HiF4QATLinear modules")
    return converted


def materialize_training_tensors(model: nn.Module) -> TrainingTensorStats:
    """Replace inference-mode tensors before FSDP synchronizes model state.

    Rotation setup intentionally runs in ``torch.inference_mode`` and registers
    a small number of derived buffers there. Such tensors work for inference
    but have no version counter, so FSDP cannot broadcast them. Parameters and
    buffers are cloned in ordinary mode without changing their values.
    """

    parameters_materialized = 0
    buffers_materialized = 0
    with torch.inference_mode(False):
        for module in model.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None or not parameter.is_inference():
                    continue
                replacement = nn.Parameter(
                    parameter.detach().clone(),
                    requires_grad=parameter.requires_grad,
                )
                module._parameters[name] = replacement
                parameters_materialized += 1
            for name, buffer in list(module._buffers.items()):
                if buffer is None or not buffer.is_inference():
                    continue
                module._buffers[name] = buffer.detach().clone()
                buffers_materialized += 1
    return TrainingTensorStats(
        parameters_materialized=parameters_materialized,
        buffers_materialized=buffers_materialized,
    )
