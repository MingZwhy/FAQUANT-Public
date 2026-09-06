import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from faquant.qad_checkpoint import load_qad_student_checkpoint
from faquant.qad_quantization import (
    enable_hif4_qat,
    promote_hif4_activation_clips,
    promote_hif4_activation_companding,
)
from faquant.qwen3 import prepare_model


def _tiny_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    )


@pytest.mark.parametrize("rotation", ["hadamard", "hisq1024"])
def test_qad_checkpoint_round_trip(tmp_path, rotation: str) -> None:
    torch.manual_seed(31)
    source = _tiny_model()
    calibration = torch.randint(0, 64, (2, 8))
    inputs = torch.randint(0, 64, (1, 7))
    from faquant.qad import qwen3_hif4_qad_config

    config = qwen3_hif4_qad_config(
        rotation=rotation,
        gptq_nsamples=2,
        gptq_seqlen=8,
        hisq_block_size=64,
    )
    prepare_model(source, config, calibration_input_ids=calibration)
    enable_hif4_qat(source)
    with torch.no_grad():
        expected = source(inputs, use_cache=False).logits
    source.save_pretrained(tmp_path, safe_serialization=True)

    torch.manual_seed(99)
    target = _tiny_model()
    result = load_qad_student_checkpoint(
        target,
        tmp_path,
        rotation=rotation,
        hisq_block_size=64,
    )
    assert result.qat.converted == 7
    assert not result.missing_keys
    assert not result.unexpected_keys
    with torch.no_grad():
        actual = target(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qad_checkpoint_round_trip_with_activation_clipping(tmp_path) -> None:
    torch.manual_seed(37)
    source = _tiny_model()
    calibration = torch.randint(0, 64, (2, 8))
    inputs = torch.randint(0, 64, (1, 7))
    from faquant.qad import qwen3_hif4_qad_config

    config = qwen3_hif4_qad_config(
        rotation="hisq1024",
        gptq_nsamples=2,
        gptq_seqlen=8,
        hisq_block_size=64,
    )
    prepare_model(source, config, calibration_input_ids=calibration)
    enable_hif4_qat(source)
    promote_hif4_activation_clips(source)
    first = next(module for module in source.modules() if hasattr(module, "activation_max"))
    first.activation_min.fill_(-0.25)
    first.activation_max.fill_(0.25)
    with torch.no_grad():
        expected = source(inputs, use_cache=False).logits
    source.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "activation_clipping.json").write_text("{}\n")

    torch.manual_seed(101)
    target = _tiny_model()
    result = load_qad_student_checkpoint(
        target,
        tmp_path,
        rotation="hisq1024",
        hisq_block_size=64,
    )
    assert not result.missing_keys
    assert not result.unexpected_keys
    with torch.no_grad():
        actual = target(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qad_checkpoint_round_trip_with_activation_companding(tmp_path) -> None:
    torch.manual_seed(39)
    source = _tiny_model()
    calibration = torch.randint(0, 64, (2, 8))
    inputs = torch.randint(0, 64, (1, 7))
    from faquant.qad import qwen3_hif4_qad_config

    config = qwen3_hif4_qad_config(
        rotation="hisq1024",
        gptq_nsamples=2,
        gptq_seqlen=8,
        hisq_block_size=64,
    )
    prepare_model(source, config, calibration_input_ids=calibration)
    enable_hif4_qat(source)
    promote_hif4_activation_companding(source)
    first = next(
        module
        for module in source.modules()
        if getattr(module, "activation_compand_log_scale", None) is not None
    )
    with torch.no_grad():
        first.activation_compand_log_scale.copy_(
            torch.linspace(-0.1, 0.1, first.in_features)
        )
        expected = source(inputs, use_cache=False).logits
    source.save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "activation_companding.json").write_text("{}\n")

    torch.manual_seed(103)
    target = _tiny_model()
    result = load_qad_student_checkpoint(
        target,
        tmp_path,
        rotation="hisq1024",
        hisq_block_size=64,
    )
    assert not result.missing_keys
    assert not result.unexpected_keys
    with torch.no_grad():
        actual = target(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
