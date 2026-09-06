import pytest
import torch
from torch import nn
from torch.nn import functional as F

from faquant.config import ExperimentConfig
from faquant.hif4 import (
    HiF4Parameters,
    fit_hif4_parameters,
    quantize_hif4_with_parameters,
)
from faquant.quantization import FakeQuantLinear, fake_quantize
from faquant.rotation import generalized_hadamard_transform


def test_fake_int4_is_finite_and_group_local() -> None:
    x = torch.tensor([[0.0, 1.0, -2.0, 3.0, 100.0, -50.0, 1.0, 2.0]])
    quantized = fake_quantize(x, bits=4, group_size=4, symmetric=True)
    assert quantized.shape == x.shape
    assert torch.isfinite(quantized).all()
    # The outlier in the second group must not change the first group's result.
    changed = x.clone()
    changed[0, 4] = 1000
    changed_quantized = fake_quantize(changed, bits=4, group_size=4, symmetric=True)
    torch.testing.assert_close(quantized[:, :4], changed_quantized[:, :4])


def test_invalid_group_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        fake_quantize(torch.randn(2, 7), group_size=4)


def test_chunked_online_hadamard_fake_quant_matches_reference() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(24, 7, bias=False)
    module = FakeQuantLinear(
        linear,
        bits=4,
        weight_group_size=8,
        activation_group_size=8,
        symmetric=True,
        clip_ratio=1.0,
        online_hadamard=True,
    )
    x = torch.randn(2, 151, 24)
    rotated = generalized_hadamard_transform(x.float()).to(x.dtype)
    expected = F.linear(fake_quantize(rotated, bits=4, group_size=8), module.weight)
    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hif4_matches_pinned_upstream_reference(dtype: torch.dtype) -> None:
    x = torch.linspace(-17.25, 18.75, 70, dtype=torch.float32).to(dtype)
    expected_values = [
        -17.5, -17.5, -15.0, -15.0, -15.0, -15.0, -15.0, -12.5,
        -12.5, -12.5, -12.5, -12.5, -10.0, -10.0, -10.0, -10.0,
        -8.75, -8.75, -7.5, -7.5, -6.25, -6.25, -6.25, -5.0,
        -4.375, -4.375, -3.75, -3.125, -2.5, -1.875, -1.875, -1.25,
        -0.625, -0.0, 0.625, 1.25, 1.25, 1.875, 2.5, 3.125,
        3.75, 3.75, 5.0, 5.0, 6.25, 6.25, 6.25, 7.5,
        7.5, 8.75, 8.75, 8.75, 10.0, 10.0, 10.0, 12.5,
        12.5, 12.5, 12.5, 12.5, 15.0, 15.0, 15.0, 15.0,
        15.0, 17.5, 17.5, 17.5, 17.5, 17.5,
    ]
    if dtype == torch.bfloat16:
        expected_values[2] = -17.5
    expected = torch.tensor(expected_values, dtype=dtype)
    actual = fake_quantize(
        x,
        quant_format="hif4",
        bits=4,
        group_size=64,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_hif4_blocks_are_local_and_tail_is_padded() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 70)
    expected = fake_quantize(x, quant_format="hif4", group_size=64)
    changed = x.clone()
    changed[:, 64:] *= 1000
    actual = fake_quantize(changed, quant_format="hif4", group_size=64)
    torch.testing.assert_close(actual[:, :64], expected[:, :64], atol=0, rtol=0)
    assert torch.isfinite(actual).all()


def test_online_hadamard_hif4_matches_explicit_reference() -> None:
    torch.manual_seed(9)
    linear = nn.Linear(128, 7, bias=False)
    module = FakeQuantLinear(
        linear,
        quant_format="hif4",
        bits=4,
        weight_group_size=64,
        activation_group_size=64,
        symmetric=True,
        clip_ratio=1.0,
        online_hadamard=True,
    )
    x = torch.randn(2, 5, 128)
    rotated = generalized_hadamard_transform(x.float()).to(x.dtype)
    expected = F.linear(
        fake_quantize(
            rotated,
            quant_format="hif4",
            bits=4,
            group_size=64,
        ),
        module.weight,
    )
    torch.testing.assert_close(module(x), expected, atol=0, rtol=0)


def test_hif4_config_resolves_and_enforces_physical_format() -> None:
    config = ExperimentConfig(quant_format="hif4")
    assert config.weight_group_size == 64
    assert config.activation_group_size == 64
    with pytest.raises(ValueError, match="group"):
        ExperimentConfig(quant_format="hif4", weight_group_size=128)
    with pytest.raises(ValueError, match="asymmetric"):
        ExperimentConfig(quant_format="hif4", symmetric=False)
    with pytest.raises(ValueError, match="clip_ratio"):
        ExperimentConfig(quant_format="hif4", clip_ratio=0.9)


def test_pv_normalizer_mode_is_validated_against_the_kernel() -> None:
    assert ExperimentConfig().pv_normalizer_mode == "unquantized"
    assert (
        ExperimentConfig(
            attention_kernel="simulated", pv_normalizer_mode="quantized_same"
        ).pv_normalizer_mode
        == "quantized_same"
    )
    with pytest.raises(ValueError, match="pv_normalizer_mode must be one of"):
        ExperimentConfig(pv_normalizer_mode="quantized")
    # The native kernel has no tiled normalizer to reorder, so accepting the flag
    # there would silently report a setting the run never applied.
    with pytest.raises(ValueError, match="requires the simulated kernel"):
        ExperimentConfig(pv_normalizer_mode="quantized_same")


def test_hif4_parameter_api_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        fit_hif4_parameters(torch.tensor(1.0))
    values = torch.randn(2, 64)
    parameters = fit_hif4_parameters(values)
    invalid = HiF4Parameters(
        quant_multiplier=parameters.quant_multiplier,
        dequant_scale=parameters.dequant_scale[:, :1],
    )
    with pytest.raises(ValueError, match="matching shapes"):
        quantize_hif4_with_parameters(values, invalid)
