import math

import pytest
import torch

from faquant.qad_losses import (
    QADLossConfig,
    adaptive_layer_feature_distillation,
    compute_qad_loss,
    entropy_adaptive_bidirectional_kl,
    policy_mixed_bidirectional_kl,
    select_adaptive_hidden_layers,
)


def test_qad_ce_uses_causal_shift_and_first_assistant_prediction() -> None:
    labels = torch.tensor([[-100, -100, 2, 3]])
    student = torch.full((1, 4, 8), -10.0)
    teacher = student.clone()
    # positions 1 and 2 predict the two assistant labels at positions 2 and 3.
    student[0, 1, 2] = 10.0
    student[0, 2, 3] = 10.0
    teacher.copy_(student)
    output = compute_qad_loss(
        student,
        teacher,
        labels,
        config=QADLossConfig(
            task_alpha=1.0,
            logit_alpha=0.0,
            feature_alpha=0.0,
        ),
    )
    assert output.assistant_tokens == 2
    assert output.task < 1e-6
    torch.testing.assert_close(output.total, output.task)


def test_qad_task_weights_apply_ce_only_to_gold_samples() -> None:
    labels = torch.tensor([[-100, 2, 3], [-100, 2, 3]])
    student = torch.full((2, 3, 8), -10.0)
    teacher = student.clone()
    # Student trajectory is intentionally wrong; gold row predicts perfectly.
    student[0, 0, 7] = 10.0
    student[0, 1, 7] = 10.0
    student[1, 0, 2] = 10.0
    student[1, 1, 3] = 10.0
    output = compute_qad_loss(
        student,
        teacher,
        labels,
        task_weights=torch.tensor([0.0, 1.0]),
        config=QADLossConfig(
            task_alpha=1.0, logit_alpha=0.0, feature_alpha=0.0,
        ),
    )
    assert output.task < 1e-6


def test_qad_zero_task_weights_have_zero_finite_task_loss() -> None:
    torch.manual_seed(23)
    student = torch.randn(2, 4, 9, requires_grad=True)
    teacher = torch.randn(2, 4, 9)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])
    output = compute_qad_loss(
        student,
        teacher,
        labels,
        task_weights=torch.zeros(2),
        config=QADLossConfig(
            task_alpha=0.05,
            logit_alpha=1.0,
            feature_alpha=0.0,
            kl_mode="reverse",
        ),
    )
    assert output.task == 0
    assert torch.isfinite(output.total)
    output.total.backward()
    assert student.grad is not None


def test_policy_mixed_kl_selects_reverse_for_student_and_forward_for_gold() -> None:
    torch.manual_seed(29)
    student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, 5, 6, 7, 8]])
    mixed, forward, reverse, _ = policy_mixed_bidirectional_kl(
        student, teacher, labels, torch.tensor([1.0, 0.0]),
        token_chunk_size=2,
    )
    student_row = compute_qad_loss(
        student[:1], teacher[:1], labels[:1],
        config=QADLossConfig(
            task_alpha=0.0, logit_alpha=1.0, feature_alpha=0.0,
            kl_mode="reverse", token_chunk_size=2,
        ),
    )
    gold_row = compute_qad_loss(
        student[1:], teacher[1:], labels[1:],
        config=QADLossConfig(
            task_alpha=0.0, logit_alpha=1.0, feature_alpha=0.0,
            kl_mode="forward", token_chunk_size=2,
        ),
    )
    torch.testing.assert_close(mixed, (student_row.total + gold_row.total) / 2)
    assert forward >= 0
    assert reverse >= 0


def test_policy_kl_requires_policy_weights() -> None:
    logits = torch.randn(1, 3, 7)
    labels = torch.tensor([[-100, 1, 2]])
    with pytest.raises(ValueError, match="requires kl_reverse_weights"):
        compute_qad_loss(
            logits, logits, labels,
            config=QADLossConfig(
                task_alpha=0.0, logit_alpha=1.0, feature_alpha=0.0,
                kl_mode="policy",
            ),
        )


def test_identical_logits_have_zero_eakld_and_valid_lambda() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 17)
    labels = torch.tensor(
        [
            [-100, -100, 4, 5, 6],
            [-100, 3, 2, -100, -100],
        ]
    )
    eakld, forward, reverse, entropy = entropy_adaptive_bidirectional_kl(
        logits,
        logits,
        labels,
        token_chunk_size=2,
    )
    torch.testing.assert_close(eakld, torch.zeros_like(eakld), atol=1e-7, rtol=0)
    torch.testing.assert_close(forward, torch.zeros_like(forward), atol=1e-7, rtol=0)
    torch.testing.assert_close(reverse, torch.zeros_like(reverse), atol=1e-7, rtol=0)
    assert entropy > 0

    output = compute_qad_loss(
        logits,
        logits,
        labels,
        config=QADLossConfig(feature_alpha=0.0, token_chunk_size=2),
    )
    assert 0 <= output.entropy_lambda <= 1
    torch.testing.assert_close(
        output.entropy_lambda,
        (entropy / math.log(16)).clamp(0.0, 1.0),
    )


def test_eakld_backpropagates_only_to_student() -> None:
    torch.manual_seed(3)
    student = torch.randn(2, 4, 13, requires_grad=True)
    teacher = torch.randn(2, 4, 13)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])
    eakld, _, _, _ = entropy_adaptive_bidirectional_kl(
        student,
        teacher,
        labels,
        token_chunk_size=1,
    )
    eakld.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_adaptive_lafd_selects_lowest_adjacent_cosine_layer() -> None:
    embedding = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    layer1 = embedding.clone()
    layer2 = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    layer3 = layer2.clone()
    teacher = (embedding, layer1, layer2, layer3)
    selected = select_adaptive_hidden_layers(teacher, topk=1)
    assert selected == (2,)

    student = tuple(value.clone() for value in teacher)
    labels = torch.tensor([[-100, 7]])
    loss, selected = adaptive_layer_feature_distillation(
        student,
        teacher,
        labels,
        topk=1,
    )
    assert selected == (2,)
    torch.testing.assert_close(loss, torch.zeros_like(loss))

    transformed_loss, _ = adaptive_layer_feature_distillation(
        student,
        teacher,
        labels,
        topk=1,
        teacher_transform=lambda value, _index: value + 1.0,
    )
    assert transformed_loss > 0


def test_full_qad_loss_is_finite_and_has_expected_components() -> None:
    torch.manual_seed(5)
    student_logits = torch.randn(2, 4, 11, requires_grad=True)
    teacher_logits = torch.randn(2, 4, 11)
    labels = torch.tensor([[-100, -100, 2, 3], [-100, 4, 5, 6]])
    student_hidden = tuple(
        torch.randn(2, 4, 6, requires_grad=True) for _ in range(4)
    )
    teacher_hidden = tuple(torch.randn(2, 4, 6) for _ in range(4))
    output = compute_qad_loss(
        student_logits,
        teacher_logits,
        labels,
        student_hidden_states=student_hidden,
        teacher_hidden_states=teacher_hidden,
        attention_mask=torch.ones_like(labels),
        config=QADLossConfig(feature_topk=2, token_chunk_size=2),
    )
    assert output.assistant_tokens == 5
    assert len(output.selected_layers) == 2
    assert torch.isfinite(output.total)
    output.total.backward()
    assert student_logits.grad is not None
    assert any(value.grad is not None for value in student_hidden)


@pytest.mark.parametrize("mode", ["forward", "reverse", "symmetric"])
def test_logit_distillation_mode_selects_requested_kl(mode: str) -> None:
    torch.manual_seed(19)
    student = torch.randn(1, 5, 7)
    teacher = torch.randn(1, 5, 7)
    labels = torch.tensor([[-100, -100, 2, 3, 4]])
    output = compute_qad_loss(
        student,
        teacher,
        labels,
        config=QADLossConfig(
            task_alpha=0.0,
            logit_alpha=1.0,
            feature_alpha=0.0,
            kl_mode=mode,
        ),
    )
    if mode == "forward":
        expected = output.forward_kl
    elif mode == "reverse":
        expected = output.reverse_kl
    else:
        expected = 0.5 * (output.forward_kl + output.reverse_kl)
    torch.testing.assert_close(output.eakld, expected)
    torch.testing.assert_close(output.total, expected)
