import copy

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from faquant.config import ExperimentConfig
from faquant.quantization import FakeQuantLinear
from faquant.qwen3 import prepare_model
from faquant.smoothquant import (
    precondition_linear_for_smoothquant,
    smoothquant_input_scale,
)


def test_official_smoothquant_convention_preserves_linear_output() -> None:
    torch.manual_seed(31)
    original = nn.Linear(64, 23, bias=True).eval()
    transformed = copy.deepcopy(original)
    inputs = torch.randn(3, 5, 64)
    activation_absmax = inputs.abs().amax(dim=(0, 1))
    scale = smoothquant_input_scale(
        transformed.weight,
        activation_absmax,
        alpha=0.9,
    )
    precondition_linear_for_smoothquant(transformed, scale)

    expected = original(inputs)
    actual = transformed(inputs * transformed.faquant_input_scale)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_qwen_smoothquant_config_is_an_isolated_hif4_gptq_ablation() -> None:
    with pytest.raises(ValueError, match="all-linear HiF4 GPTQ"):
        ExperimentConfig(qwen_smoothquant=True)
    with pytest.raises(ValueError, match="no-rotation"):
        ExperimentConfig(
            rotation="hadamard",
            quant_target="all",
            quant_format="hif4",
            weight_quant="gptq",
            qwen_smoothquant=True,
        )


def test_tiny_qwen_hif4_gptq_smoothquant_covers_every_linear() -> None:
    torch.manual_seed(32)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            attention_dropout=0.0,
        )
    ).eval()
    calibration = torch.randint(0, 64, (2, 8))
    config = ExperimentConfig(
        rotation="none",
        online_hadamard=False,
        attention_input_quant=False,
        attention_output_quant=True,
        qk_matmul_quant=False,
        pv_matmul_quant=False,
        quant_target="all",
        quant_format="hif4",
        weight_quant="gptq",
        qwen_smoothquant=True,
        qwen_smoothquant_alpha=0.9,
    )
    prepare_model(model, config, calibration_input_ids=calibration)

    modules = [
        module for module in model.modules() if isinstance(module, FakeQuantLinear)
    ]
    assert len(modules) == 7
    assert all(module.input_scale is not None for module in modules)
    assert all(torch.isfinite(module.input_scale).all() for module in modules)
    assert len(model.faquant_smoothquant_stats["layers"]) == 7
    assert model.faquant_smoothquant_stats["alpha"] == 0.9
    assert not hasattr(model, "faquant_rotation_signs")
    assert not model.model.layers[0].self_attn.faquant_qk_rotation

    with torch.inference_mode():
        logits = model(calibration[:1], use_cache=False).logits
    assert torch.isfinite(logits).all()
