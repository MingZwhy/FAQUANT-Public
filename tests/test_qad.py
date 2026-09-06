import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from faquant.gptq import calibration_tokens
from faquant.qad import (
    prepare_qad_student,
    prepare_qad_teacher,
    qwen3_hif4_qad_config,
)
from faquant.qad_quantization import HiF4QATLinear


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


def test_qad_config_is_exact_online_none_target() -> None:
    config = qwen3_hif4_qad_config(rotation="hadamard")
    assert config.online_hadamard
    assert config.attention_kernel == "native"
    assert not config.attention_input_quant
    assert config.attention_output_quant
    assert not config.qk_matmul_quant
    assert not config.pv_matmul_quant
    assert config.quant_target == "all"
    assert config.quant_format == "hif4"
    assert config.qad_capture_weight_metadata

    hisq = qwen3_hif4_qad_config(rotation="hisq1024")
    assert hisq.rotation == "none"
    assert not hisq.online_hadamard
    assert hisq.qwen_hisq_input_rotation
    assert hisq.qwen_hisq_rotation_block_size == 1024
    assert hisq.qwen_hisq_rotation_seed == 17


def test_tiny_qad_student_and_aligned_teacher_have_gradients() -> None:
    torch.manual_seed(7)
    student = _tiny_model()
    teacher = _tiny_model()
    teacher.load_state_dict(student.state_dict())
    calibration = torch.randint(0, 64, (2, 8))
    inputs = torch.randint(0, 64, (1, 7))

    stats = prepare_qad_student(
        student,
        rotation="hadamard",
        calibration_input_ids=calibration,
        gptq_nsamples=2,
        gptq_seqlen=8,
        hisq_block_size=64,
    )
    prepare_qad_teacher(teacher, rotation="hadamard")
    assert stats.qat.converted == 7
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    output = student(inputs, use_cache=False).logits
    output.float().square().mean().backward()
    qats = [
        module for module in student.modules() if isinstance(module, HiF4QATLinear)
    ]
    assert len(qats) == 7
    assert all(module.weight.grad is not None for module in qats)


def test_tiny_hisq_qad_student_preserves_local_rotations() -> None:
    torch.manual_seed(8)
    student = _tiny_model()
    teacher = _tiny_model()
    calibration = torch.randint(0, 64, (2, 8))
    inputs = torch.randint(0, 64, (1, 7))

    stats = prepare_qad_student(
        student,
        rotation="hisq1024",
        calibration_input_ids=calibration,
        gptq_nsamples=2,
        gptq_seqlen=8,
        hisq_block_size=64,
    )
    prepare_qad_teacher(teacher, rotation="hisq1024")
    qats = [
        module for module in student.modules() if isinstance(module, HiF4QATLinear)
    ]
    assert stats.qat.converted == 7
    assert len(qats) == 7
    assert all(module.input_rotation_signs is not None for module in qats)
    assert all(module.input_rotation_permutation is not None for module in qats)
    assert all(module.input_rotation_block_size == 64 for module in qats)

    student(inputs, use_cache=False).logits.float().square().mean().backward()
    assert all(module.weight.grad is not None for module in qats)
