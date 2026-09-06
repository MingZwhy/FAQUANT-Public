import copy

import pytest
import torch
from torch import nn

from faquant.qad_quantization import (
    HiF4QATLinear,
    hif4_code_step_parameter_groups,
    collect_hif4_qat_stats,
    collect_hif4_qat_master_grid_stats,
    configure_qat_trainable_scope,
    convert_hif4_qat_to_float_deployed,
    convert_hif4_qat_to_float_master,
    convert_hif4_qat_to_inference,
    enable_hif4_qat,
    materialize_training_tensors,
    promote_hif4_activation_clips,
    promote_hif4_activation_companding,
    scale_hif4_qat_master_residual,
    set_hif4_qat_metadata_mode,
    ste_fake_quantize_hif4,
)


def test_hif4_code_step_parameter_groups_scale_linear_learning_rates() -> None:
    first = HiF4QATLinear(_fake_quant_linear(out_features=16))
    second = HiF4QATLinear(_fake_quant_linear(out_features=16))
    second.weight_hif4_scale.mul_(4.0)
    model = nn.Sequential(first, second)

    groups, stats = hif4_code_step_parameter_groups(model, base_lr=1e-5)

    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-5)
    assert groups[1]["lr"] == pytest.approx(4e-5)
    assert stats.linears == 2
    assert stats.minimum_factor == pytest.approx(1.0)
    assert stats.maximum_factor == pytest.approx(4.0)


def test_qat_trainable_scope_can_freeze_early_transformer_layers() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            nn.Sequential(HiF4QATLinear(_fake_quant_linear())),
            nn.Sequential(HiF4QATLinear(_fake_quant_linear())),
        ]
    )

    stats = configure_qat_trainable_scope(
        model, "qat-linears", minimum_transformer_layer=1
    )

    assert not model.layers[0][0].weight.requires_grad
    assert model.layers[1][0].weight.requires_grad
    assert stats.trainable_tensors == 1


def test_qat_bf16_only_scope_freezes_quantized_linears() -> None:
    model = nn.Sequential(
        HiF4QATLinear(_fake_quant_linear()),
        nn.LayerNorm(16),
    )

    stats = configure_qat_trainable_scope(model, "bf16-only")

    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad
    assert stats.trainable_parameters == 32


def test_qat_norms_only_scope_freezes_everything_else() -> None:
    model = nn.Sequential(
        HiF4QATLinear(_fake_quant_linear()),
        nn.LayerNorm(16),
        nn.Linear(16, 16),
    )

    stats = configure_qat_trainable_scope(model, "norms-only")

    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad
    assert model[1].bias.requires_grad
    assert not model[2].weight.requires_grad
    assert stats.trainable_parameters == 32


def test_qat_input_scales_only_has_exact_unit_initialization() -> None:
    module = HiF4QATLinear(_fake_quant_linear())
    model = nn.Sequential(module)
    inputs = torch.randn(2, 64)
    expected = model(inputs)

    stats = configure_qat_trainable_scope(model, "input-scales-only")
    actual = model(inputs)

    torch.testing.assert_close(module.input_scale, torch.ones(64))
    torch.testing.assert_close(actual, expected)
    assert stats.trainable_tensors == 1
    assert stats.trainable_parameters == 64


def test_qat_activation_clips_are_noop_persistent_buffers() -> None:
    module = HiF4QATLinear(_fake_quant_linear())
    model = nn.Sequential(module)
    inputs = torch.randn(3, 64)
    with torch.no_grad():
        expected = model(inputs)

    stats = promote_hif4_activation_clips(model)
    with torch.no_grad():
        actual = model(inputs)

    assert stats.linears == 1
    assert stats.elements == 64
    assert torch.isneginf(module.activation_min).all()
    assert torch.isposinf(module.activation_max).all()
    assert "0.activation_min" in model.state_dict()
    assert "0.activation_max" in model.state_dict()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qat_activation_companding_has_parity_and_trainable_gradient() -> None:
    module = HiF4QATLinear(_fake_quant_linear())
    model = nn.Sequential(module)
    inputs = torch.randn(3, 64)
    with torch.no_grad():
        expected = model(inputs)

    scope = configure_qat_trainable_scope(model, "activation-companding-only")
    actual = model(inputs)

    assert scope.trainable_tensors == 1
    assert scope.trainable_parameters == 64
    torch.testing.assert_close(actual.detach(), expected, atol=0, rtol=0)
    actual.float().square().mean().backward()
    assert module.activation_compand_log_scale.grad is not None
    assert torch.isfinite(module.activation_compand_log_scale.grad).all()
    assert module.activation_compand_log_scale.grad.abs().sum() > 0
    assert module.weight.grad is None


def test_qat_activation_companding_inference_conversion_preserves_forward() -> None:
    module = HiF4QATLinear(_fake_quant_linear())
    model = nn.Sequential(module)
    promote_hif4_activation_companding(model)
    with torch.no_grad():
        module.activation_compand_log_scale.copy_(
            torch.linspace(-0.2, 0.2, module.in_features)
        )
    inputs = torch.randn(3, 64)
    with torch.no_grad():
        expected = model(inputs)
    convert_hif4_qat_to_inference(model)
    with torch.no_grad():
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qat_activation_companding_can_tie_hif4_lv2_subblocks() -> None:
    module = HiF4QATLinear(_fake_quant_linear())
    model = nn.Sequential(module)
    inputs = torch.randn(3, 64)
    with torch.no_grad():
        expected = model(inputs)
    scope = configure_qat_trainable_scope(
        model,
        "activation-companding-only",
        activation_companding_group_size=8,
    )
    with torch.no_grad():
        actual = model(inputs)
    assert module.activation_compand_log_scale.shape == (8,)
    assert scope.trainable_parameters == 8
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
from faquant.hif4 import (
    fit_hif4_compact_parameters,
    project_hif4_source_to_target_cells,
    quantize_hif4_with_compact_parameters,
)
from faquant.quantization import FakeQuantLinear, fake_quantize


def _fake_quant_linear(
    in_features: int = 64,
    out_features: int = 16,
    *,
    online_hadamard: bool = False,
) -> FakeQuantLinear:
    torch.manual_seed(41)
    linear = nn.Linear(in_features, out_features, bias=True)
    grouped = linear.weight.detach().reshape(out_features, in_features // 64, 64)
    parameters = fit_hif4_compact_parameters(grouped)
    module = FakeQuantLinear(
        linear,
        quant_format="hif4",
        bits=4,
        weight_group_size=64,
        activation_group_size=64,
        symmetric=True,
        clip_ratio=1.0,
        online_hadamard=online_hadamard,
    )
    module.faquant_hif4_qat_parameters = parameters
    return module


def test_hif4_ste_forward_is_exact_and_backward_is_identity() -> None:
    torch.manual_seed(42)
    x = torch.randn(3, 64, requires_grad=True)
    actual = ste_fake_quantize_hif4(x)
    expected = fake_quantize(
        x.detach(),
        quant_format="hif4",
        bits=4,
        group_size=64,
    )
    torch.testing.assert_close(actual.detach(), expected, atol=0, rtol=0)

    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_latent_master_projection_preserves_target_codes() -> None:
    torch.manual_seed(43)
    source = torch.randn(4, 2, 64)
    fitted = fit_hif4_compact_parameters(source)
    target = quantize_hif4_with_compact_parameters(source * 0.83, fitted)
    latent = project_hif4_source_to_target_cells(
        source,
        target,
        fitted,
        output_dtype=torch.bfloat16,
    )
    actual = quantize_hif4_with_compact_parameters(latent, fitted)
    torch.testing.assert_close(actual, target.to(actual.dtype), atol=0, rtol=0)
    assert torch.any(latent.float().ne(target))


def test_qat_uses_optional_latent_master_without_changing_step_zero() -> None:
    frozen = _fake_quant_linear()
    grouped = frozen.weight.detach().reshape(16, 1, 64)
    parameters = frozen.faquant_hif4_qat_parameters
    source = grouped.float() + torch.randn_like(grouped.float()) * 0.01
    latent = project_hif4_source_to_target_cells(
        source,
        grouped,
        parameters,
        output_dtype=frozen.weight.dtype,
    ).reshape_as(frozen.weight)
    frozen.faquant_hif4_qat_master = latent.cpu()
    qat = HiF4QATLinear(frozen)
    torch.testing.assert_close(qat.weight, latent, atol=0, rtol=0)
    torch.testing.assert_close(qat._quantize_weight(), frozen.weight, atol=0, rtol=0)
    stats = collect_hif4_qat_master_grid_stats(nn.Sequential(qat))
    assert stats.linears == 1
    assert stats.master_off_grid_elements > 0
    assert stats.master_to_grid_relative_l2 > 0


@pytest.mark.parametrize("online_hadamard", [False, True])
def test_qat_step_zero_matches_inference_and_has_gradients(
    online_hadamard: bool,
) -> None:
    frozen = _fake_quant_linear(online_hadamard=online_hadamard)
    qat = HiF4QATLinear(copy.deepcopy(frozen)).train()
    x_frozen = torch.randn(2, 5, 64)
    x_qat = x_frozen.detach().clone().requires_grad_(True)

    with torch.no_grad():
        expected = frozen(x_frozen)
    actual = qat(x_qat)
    torch.testing.assert_close(actual.detach(), expected, atol=0, rtol=0)

    actual.float().square().mean().backward()
    assert qat.weight.grad is not None
    assert torch.isfinite(qat.weight.grad).all()
    assert qat.weight.grad.abs().sum() > 0
    assert x_qat.grad is not None
    assert torch.isfinite(x_qat.grad).all()


def test_model_conversion_round_trip_preserves_forward() -> None:
    model = nn.Sequential(
        _fake_quant_linear(out_features=64),
        nn.GELU(),
        _fake_quant_linear(),
    )
    x = torch.randn(2, 64)
    with torch.no_grad():
        expected = model(x)

    enabled = enable_hif4_qat(model)
    assert enabled.converted == 2
    assert all(
        isinstance(model[index], HiF4QATLinear) for index in (0, 2)
    )
    with torch.no_grad():
        qat_output = model(x)
    torch.testing.assert_close(qat_output, expected, atol=0, rtol=0)
    assert collect_hif4_qat_stats(model).quantized_forward_calls == 2

    converted = convert_hif4_qat_to_inference(model)
    assert converted.converted == 2
    assert converted.quantized_forward_calls == 2
    assert all(isinstance(model[index], FakeQuantLinear) for index in (0, 2))
    with torch.no_grad():
        actual = model(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_float_master_conversion_preserves_master_weights() -> None:
    frozen = _fake_quant_linear()
    model = nn.Sequential(HiF4QATLinear(frozen))
    with torch.no_grad():
        model[0].weight.add_(0.03125)
        expected_weight = model[0].weight.detach().clone()
        expected_bias = model[0].bias.detach().clone()

    converted = convert_hif4_qat_to_float_master(model)

    assert converted.converted == 1
    assert isinstance(model[0], nn.Linear)
    torch.testing.assert_close(model[0].weight, expected_weight, atol=0, rtol=0)
    torch.testing.assert_close(model[0].bias, expected_bias, atol=0, rtol=0)


def test_qat_scope_freezes_non_quantized_parameters() -> None:
    model = nn.Sequential(
        nn.Embedding(8, 64),
        HiF4QATLinear(_fake_quant_linear(out_features=64)),
        nn.LayerNorm(64),
    )
    stats = configure_qat_trainable_scope(model, "qat-linears")
    assert model[1].weight.requires_grad
    assert not model[1].bias.requires_grad
    assert not model[0].weight.requires_grad
    assert not model[2].weight.requires_grad
    assert stats.trainable_tensors == 1
    assert stats.trainable_parameters == model[1].weight.numel()


def test_weight_ste_is_forward_exact_when_master_drifts_far_off_grid() -> None:
    """The straight-through weight must reach the matmul bit-exactly.

    `self.weight + (quantized - self.weight).detach()` is the natural way to
    write this and it is what shipped, but in floating point it equals
    `quantized` only while the subtraction is exact -- Sterbenz's condition,
    which needs the two within a factor of two of each other.  QAD walks master
    weights away from their grid point until that stops holding for a few
    elements, and then the training forward and the module `to_inference`
    materializes disagree by an ULP.  That is a train/deploy mismatch, and
    scripts/eval_qwen_hif4_qad.py refuses such a checkpoint outright: step 750 of
    the 1250-step seed-2718 run lost both of its readouts to a 2.5 logit gap
    grown from 44 elements in three early-layer projections.

    Rather than reverse-engineer that weight configuration, this pins the
    contract directly.  `_quantize_weight` is stubbed to hand back grid values
    that violate Sterbenz against the master -- the ratios below are outside
    [0.5, 2], which is the only way the old form can round -- and both the
    training forward and the converted module read the stub, so any gap between
    them is the estimator's own arithmetic.
    """

    frozen = _fake_quant_linear()
    qat = HiF4QATLinear(frozen).to(torch.bfloat16)

    # Pairs where bf16 `w + (q - w)` rounds away from `q`, found by search.
    pairs = [
        (-0.1127930, 0.0212402),
        (0.1777344, 0.0366211),
        (-0.0593262, 0.0854492),
        (-0.0490723, 0.0893555),
        (0.1845703, -0.0314941),
        (-0.0158691, 0.0541992),
    ]
    numel = qat.weight.numel()
    masters = torch.tensor(
        [pairs[i % len(pairs)][0] for i in range(numel)], dtype=torch.bfloat16
    ).reshape_as(qat.weight)
    grid = torch.tensor(
        [pairs[i % len(pairs)][1] for i in range(numel)], dtype=torch.bfloat16
    ).reshape_as(qat.weight)

    with torch.no_grad():
        qat.weight.copy_(masters)
    qat._quantize_weight = lambda: grid.clone()  # type: ignore[method-assign]

    # The premise: this configuration does break the old formulation.
    assert bool(((qat.weight + (grid - qat.weight)) != grid).any())

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        training_forward = qat(x)
        deployed_forward = copy.deepcopy(qat).to_inference()(x)
    torch.testing.assert_close(training_forward, deployed_forward, atol=0, rtol=0)

    # And the estimator still hands the master weight an identity gradient: the
    # weight sees the quantized activations, not the raw ones.
    x_grad = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    qat(x_grad).float().sum().backward()
    assert qat.weight.grad is not None
    with torch.no_grad():
        expected_grad = qat._quantize_input(x_grad.detach()).float().sum(dim=0)
    torch.testing.assert_close(
        qat.weight.grad.float(),
        expected_grad.expand_as(qat.weight),
        atol=0,
        rtol=0,
    )


def test_dynamic_metadata_mode_refits_current_master() -> None:
    model = nn.Sequential(HiF4QATLinear(_fake_quant_linear()))
    assert set_hif4_qat_metadata_mode(model, "dynamic") == 1
    assert model[0].metadata_mode == "dynamic"
    with torch.no_grad():
        model[0].weight.mul_(1.5)
        grouped = model[0].weight.reshape(16, 1, 64)
        expected = quantize_hif4_with_compact_parameters(
            grouped,
            fit_hif4_compact_parameters(grouped),
        ).reshape_as(model[0].weight)
    torch.testing.assert_close(model[0]._quantize_weight(), expected, atol=0, rtol=0)


def test_latent_residual_scaling_preserves_codes_and_reduces_distance() -> None:
    frozen = _fake_quant_linear()
    qat = HiF4QATLinear(frozen)
    with torch.no_grad():
        deployed = qat._quantize_weight()
        qat.weight.add_(0.001)
        before = (qat.weight.float() - deployed.float()).norm()
    assert scale_hif4_qat_master_residual(nn.Sequential(qat), 0.25) == 1
    torch.testing.assert_close(qat._quantize_weight(), deployed, atol=0, rtol=0)
    after = (qat.weight.float() - deployed.float()).norm()
    torch.testing.assert_close(after, before * 0.25, rtol=2e-3, atol=1e-5)


def test_float_master_conversion_rejects_online_rotation() -> None:
    model = nn.Sequential(HiF4QATLinear(_fake_quant_linear(online_hadamard=True)))
    with pytest.raises(ValueError, match="unrotated"):
        convert_hif4_qat_to_float_master(model)


def test_float_deployed_conversion_materializes_quantized_weight() -> None:
    frozen = _fake_quant_linear()
    model = nn.Sequential(HiF4QATLinear(frozen))
    with torch.no_grad():
        model[0].weight.add_(0.03125)
        expected = model[0]._quantize_weight().detach().clone()
        master = model[0].weight.detach().clone()

    converted = convert_hif4_qat_to_float_deployed(model)

    assert converted.converted == 1
    assert isinstance(model[0], nn.Linear)
    torch.testing.assert_close(model[0].weight, expected, atol=0, rtol=0)
    assert not torch.equal(model[0].weight, master)


def test_qat_rejects_non_hif4_module() -> None:
    module = FakeQuantLinear(
        nn.Linear(64, 8, bias=False),
        quant_format="int",
        bits=4,
        weight_group_size=64,
        activation_group_size=64,
        symmetric=True,
        clip_ratio=1.0,
    )
    with pytest.raises(ValueError, match="HiF4"):
        HiF4QATLinear(module)


def test_materialize_training_tensors_replaces_inference_buffers() -> None:
    module = nn.Linear(4, 4)
    with torch.inference_mode():
        module.register_buffer("derived", torch.arange(4.0))
    assert module.derived.is_inference()
    before = module.derived.clone()
    stats = materialize_training_tensors(module)
    assert stats.buffers_materialized == 1
    assert stats.parameters_materialized == 0
    assert not module.derived.is_inference()
    torch.testing.assert_close(module.derived, before)
