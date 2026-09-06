import copy
import hashlib
import json
import math

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from faquant.config import ExperimentConfig
from faquant.hisq_rotation import (
    apply_hisq_rotation,
    derive_hisq_rotation,
    precondition_linear_for_hisq_rotation,
    resolve_hisq_block_size,
)
from faquant.qad_checkpoint import load_qad_student_checkpoint
from faquant.quantization import FakeQuantLinear
from faquant.qwen3 import prepare_model


def _official_reference(
    x: torch.Tensor, layer_name: str, seed: int, block_size: int
) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}::{layer_name}".encode("utf-8")).digest()
    layer_seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    generator = torch.Generator(device="cpu").manual_seed(layer_seed)
    signs = torch.randint(0, 2, (x.shape[-1],), generator=generator)
    signs = signs.mul(2).sub(1).to(x)
    permutation = torch.randperm(x.shape[-1], generator=generator)
    hadamard = torch.ones((1, 1))
    while hadamard.shape[0] < block_size:
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            ),
            dim=0,
        )
    hadamard = hadamard.to(x) / math.sqrt(block_size)
    work = (x * signs).index_select(-1, permutation)
    return (work.reshape(-1, block_size) @ hadamard).reshape_as(x)


def _tiny_qwen() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
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


def test_hisq_rotation_matches_official_construction() -> None:
    torch.manual_seed(41)
    x = torch.randn(3, 16)
    rotation = derive_hisq_rotation(
        "model.layers.0.self_attn.q_proj", 16, seed=17, block_size=8
    )
    actual = apply_hisq_rotation(x, rotation)
    expected = _official_reference(
        x, "model.layers.0.self_attn.q_proj", 17, 8
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_hisq_rotation_preserves_linear_output_before_quantization() -> None:
    torch.manual_seed(42)
    original = nn.Linear(64, 23, bias=True).eval()
    transformed = copy.deepcopy(original)
    inputs = torch.randn(3, 5, 64)
    rotation = derive_hisq_rotation("linear", 64, block_size=16)
    precondition_linear_for_hisq_rotation(transformed, rotation)

    expected = original(inputs)
    actual = transformed(apply_hisq_rotation(inputs, rotation))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_qwen_hisq_rotation_config_is_isolated() -> None:
    with pytest.raises(ValueError, match="all-linear HiF4 quantization"):
        ExperimentConfig(qwen_hisq_input_rotation=True)
    with pytest.raises(ValueError, match="separate ablations"):
        ExperimentConfig(
            rotation="hadamard",
            quant_target="all",
            quant_format="hif4",
            weight_quant="gptq",
            qwen_hisq_input_rotation=True,
        )
    with pytest.raises(ValueError, match="isolated from SmoothQuant"):
        ExperimentConfig(
            quant_target="all",
            quant_format="hif4",
            weight_quant="gptq",
            qwen_hisq_input_rotation=True,
            qwen_smoothquant=True,
        )
    with pytest.raises(ValueError, match="positive power of two"):
        ExperimentConfig(
            quant_target="all",
            quant_format="hif4",
            weight_quant="gptq",
            qwen_hisq_input_rotation=True,
            qwen_hisq_rotation_block_size=96,
        )


def test_tiny_qwen_hif4_gptq_hisq_rotation_covers_every_linear() -> None:
    torch.manual_seed(43)
    model = _tiny_qwen()
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
        qwen_hisq_input_rotation=True,
        qwen_hisq_rotation_block_size=16,
        qwen_hisq_rotation_seed=17,
    )
    prepare_model(model, config, calibration_input_ids=calibration)

    modules = [
        module for module in model.modules() if isinstance(module, FakeQuantLinear)
    ]
    assert len(modules) == 7
    assert all(module.input_rotation_signs is not None for module in modules)
    assert all(module.input_rotation_permutation is not None for module in modules)
    assert all(module.input_rotation_block_size == 16 for module in modules)
    assert len(model.faquant_hisq_rotation_stats["layers"]) == 7
    assert not hasattr(model, "faquant_rotation_signs")
    assert not model.model.layers[0].self_attn.faquant_qk_rotation

    with torch.inference_mode():
        logits = model(calibration[:1], use_cache=False).logits
    assert torch.isfinite(logits).all()


def test_tiny_qwen_hif4_rtn_hisq_rotation_covers_every_linear() -> None:
    torch.manual_seed(44)
    model = _tiny_qwen()
    config = ExperimentConfig(
        rotation="none",
        online_hadamard=False,
        attention_input_quant=False,
        attention_output_quant=True,
        qk_matmul_quant=False,
        pv_matmul_quant=False,
        quant_target="all",
        quant_format="hif4",
        weight_quant="rtn",
        qwen_hisq_input_rotation=True,
        qwen_hisq_rotation_block_size=16,
        qwen_hisq_rotation_seed=17,
    )
    prepare_model(model, config)

    modules = [
        module for module in model.modules() if isinstance(module, FakeQuantLinear)
    ]
    assert len(modules) == 7
    assert all(module.input_rotation_signs is not None for module in modules)
    assert all(module.input_rotation_block_size == 16 for module in modules)
    assert all(module.weight.dtype == torch.float32 for module in modules)
    assert len(model.faquant_hisq_rotation_stats["layers"]) == 7
    assert not hasattr(model, "faquant_gptq_stats")

    inputs = torch.randint(0, 64, (1, 8))
    with torch.inference_mode():
        logits = model(inputs, use_cache=False).logits
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("weight_quant", ("rtn", "gptq"))
def test_down_proj_hisq_block_can_be_widened_on_its_own(weight_quant: str) -> None:
    torch.manual_seed(45)
    # down_proj is the only projection whose input is wider than the residual
    # stream, which is why a single global block under-mixes it, so the tiny
    # model needs an intermediate size that actually differs from hidden_size.
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
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
        weight_quant=weight_quant,
        qwen_hisq_input_rotation=True,
        qwen_hisq_rotation_block_size=16,
        qwen_hisq_rotation_block_size_down_proj=32,
        qwen_hisq_rotation_seed=17,
    )
    prepare_model(
        model,
        config,
        calibration_input_ids=calibration if weight_quant == "gptq" else None,
    )

    blocks = {
        name.split(".")[-1]: module.input_rotation_block_size
        for name, module in model.named_modules()
        if isinstance(module, FakeQuantLinear)
    }
    assert blocks.pop("down_proj") == 32
    assert set(blocks.values()) == {16}

    with torch.inference_mode():
        logits = model(calibration[:1], use_cache=False).logits
    assert torch.isfinite(logits).all()


def test_down_proj_block_override_must_divide_the_input_width() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        resolve_hisq_block_size(
            "model.layers.0.mlp.down_proj",
            96,
            block_size=16,
            down_proj_block_size=64,
        )
    # Every other projection keeps the global block whatever the override says.
    assert (
        resolve_hisq_block_size(
            "model.layers.0.self_attn.q_proj",
            96,
            block_size=16,
            down_proj_block_size=64,
        )
        == 16
    )


def test_checkpoint_loader_rejects_a_down_proj_block_size_it_did_not_record(
    tmp_path,
) -> None:
    # The saved permutation and signs are the same shape for any block size, so
    # a disagreement here would otherwise load cleanly and rotate wrongly.
    (tmp_path / "qad_init.json").write_text(
        json.dumps({"hisq_down_proj_block_size": None}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="disagrees with the checkpoint"):
        load_qad_student_checkpoint(
            _tiny_qwen(),
            tmp_path,
            rotation="hisq1024",
            hisq_down_proj_block_size=4096,
        )
