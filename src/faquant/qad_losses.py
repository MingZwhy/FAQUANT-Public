from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class QADLossConfig:
    task_alpha: float = 0.05
    logit_alpha: float = 2.0
    feature_alpha: float = 0.5
    temperature: float = 1.0
    entropy_k: int = 16
    feature_topk: int = 3
    token_chunk_size: int = 128
    kl_mode: str = "eakld"

    def __post_init__(self) -> None:
        if min(self.task_alpha, self.logit_alpha, self.feature_alpha) < 0:
            raise ValueError("QAD loss weights must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.entropy_k <= 1:
            raise ValueError("entropy_k must be greater than one")
        if self.feature_topk <= 0:
            raise ValueError("feature_topk must be positive")
        if self.token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        if self.kl_mode not in (
            "eakld", "forward", "reverse", "symmetric", "policy",
        ):
            raise ValueError(
                "kl_mode must be 'eakld', 'forward', 'reverse', 'symmetric', "
                "or 'policy'"
            )


@dataclass(frozen=True)
class QADLossOutput:
    total: torch.Tensor
    task: torch.Tensor
    eakld: torch.Tensor
    forward_kl: torch.Tensor
    reverse_kl: torch.Tensor
    teacher_entropy: torch.Tensor
    entropy_lambda: torch.Tensor
    lafd: torch.Tensor
    selected_layers: tuple[int, ...]
    assistant_tokens: int


def _causal_assistant_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    mask = labels[:, 1:].ne(-100)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every QAD sample must contain an assistant target token")
    return mask


def _batch_mean(
    values: torch.Tensor,
    sample_indices: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    return _batch_values(values, sample_indices, counts).mean()


def _batch_values(
    values: torch.Tensor,
    sample_indices: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    totals = torch.zeros(
        counts.shape[0],
        device=values.device,
        dtype=values.dtype,
    )
    totals.index_add_(0, sample_indices, values)
    return totals / counts.to(values.dtype)


def _per_sample_bidirectional_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if student_logits.ndim != 3 or student_logits.shape[:2] != labels.shape:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if temperature <= 0 or token_chunk_size <= 0:
        raise ValueError("invalid KL temperature or token chunk size")

    mask = _causal_assistant_mask(labels)
    batch, sequence = mask.shape
    sample_grid = (
        torch.arange(batch, device=labels.device)[:, None]
        .expand(batch, sequence)
    )
    sample_indices = sample_grid[mask]
    counts = mask.sum(dim=-1)
    selected_student = student_logits[:, :-1][mask]
    selected_teacher = teacher_logits[:, :-1][mask]

    forward_parts: list[torch.Tensor] = []
    reverse_parts: list[torch.Tensor] = []
    entropy_parts: list[torch.Tensor] = []
    for start in range(0, selected_student.shape[0], token_chunk_size):
        stop = min(start + token_chunk_size, selected_student.shape[0])
        student_log_probs = F.log_softmax(
            selected_student[start:stop].float() / temperature,
            dim=-1,
        )
        teacher_log_probs = F.log_softmax(
            selected_teacher[start:stop].float() / temperature,
            dim=-1,
        )
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()
        forward_parts.append(
            teacher_probs.mul(teacher_log_probs - student_log_probs).sum(dim=-1)
            * temperature**2
        )
        reverse_parts.append(
            student_probs.mul(student_log_probs - teacher_log_probs).sum(dim=-1)
            * temperature**2
        )
        entropy_parts.append(-(teacher_probs * teacher_log_probs).sum(dim=-1))
    return (
        _batch_values(torch.cat(forward_parts), sample_indices, counts),
        _batch_values(torch.cat(reverse_parts), sample_indices, counts),
        _batch_values(torch.cat(entropy_parts), sample_indices, counts),
    )


def entropy_adaptive_bidirectional_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    entropy_k: int = 16,
    token_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal-shift-correct full-vocabulary entropy-aware KL distillation.

    The vocabulary distribution remains exact; only valid assistant-token rows
    are chunked to bound the temporary FP32 log-probability tensors.
    """

    if temperature <= 0 or entropy_k <= 1 or token_chunk_size <= 0:
        raise ValueError("invalid EAKLD temperature, entropy_k, or chunk size")
    forward_values, reverse_values, entropy_values = _per_sample_bidirectional_kl(
        student_logits,
        teacher_logits,
        labels,
        temperature=temperature,
        token_chunk_size=token_chunk_size,
    )
    forward_kl = forward_values.mean()
    reverse_kl = reverse_values.mean()
    teacher_entropy = entropy_values.mean()
    entropy_lambda = (
        teacher_entropy / math.log(entropy_k)
    ).clamp(0.0, 1.0)
    eakld = (
        entropy_lambda * forward_kl
        + (1.0 - entropy_lambda) * reverse_kl
    )
    return eakld, forward_kl, reverse_kl, teacher_entropy


def policy_mixed_bidirectional_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    kl_reverse_weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    token_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use reverse KL on student trajectories and forward KL on gold rows."""

    if (
        kl_reverse_weights.ndim != 1
        or kl_reverse_weights.shape[0] != labels.shape[0]
    ):
        raise ValueError("kl_reverse_weights must have shape [batch]")
    weights = kl_reverse_weights.to(device=labels.device, dtype=torch.float32)
    if not torch.isfinite(weights).all() or torch.any((weights < 0) | (weights > 1)):
        raise ValueError("kl_reverse_weights must be finite and in [0, 1]")
    forward_values, reverse_values, entropy_values = _per_sample_bidirectional_kl(
        student_logits,
        teacher_logits,
        labels,
        temperature=temperature,
        token_chunk_size=token_chunk_size,
    )
    weights = weights.to(device=forward_values.device, dtype=forward_values.dtype)
    mixed = ((1.0 - weights) * forward_values + weights * reverse_values).mean()
    return (
        mixed,
        forward_values.mean(),
        reverse_values.mean(),
        entropy_values.mean(),
    )


def select_adaptive_hidden_layers(
    teacher_hidden_states: tuple[torch.Tensor, ...],
    *,
    topk: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[int, ...]:
    """Select teacher layers with the lowest adjacent-layer cosine similarity."""

    if len(teacher_hidden_states) < 2:
        raise ValueError("adaptive LAFD requires embedding plus layer hidden states")
    layers = len(teacher_hidden_states) - 1
    if not 1 <= topk <= layers:
        raise ValueError("feature_topk exceeds the number of transformer layers")
    if attention_mask is not None and attention_mask.shape != teacher_hidden_states[0].shape[:2]:
        raise ValueError("attention_mask must match hidden-state batch and sequence")

    similarities: list[torch.Tensor] = []
    for index in range(1, len(teacher_hidden_states)):
        cosine = F.cosine_similarity(
            teacher_hidden_states[index].float(),
            teacher_hidden_states[index - 1].float(),
            dim=-1,
        )
        if attention_mask is None:
            similarity = cosine.mean()
        else:
            valid = attention_mask.to(device=cosine.device, dtype=torch.bool)
            similarity = cosine.masked_select(valid).mean()
        similarities.append(similarity.detach())
    values = torch.stack(similarities)
    return tuple(
        int(index)
        for index in torch.topk(values, k=topk, largest=False).indices.sort().values
        + 1
    )


def adaptive_layer_feature_distillation(
    student_hidden_states: tuple[torch.Tensor, ...],
    teacher_hidden_states: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    *,
    topk: int = 3,
    attention_mask: torch.Tensor | None = None,
    teacher_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Assistant-token LAFD over adaptively selected teacher layers."""

    if len(student_hidden_states) != len(teacher_hidden_states):
        raise ValueError("student and teacher hidden-state tuples must align")
    selected_layers = select_adaptive_hidden_layers(
        teacher_hidden_states,
        topk=topk,
        attention_mask=attention_mask,
    )
    mask = labels.ne(-100)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every QAD sample must contain an assistant token")
    counts = mask.sum(dim=-1)
    layer_losses: list[torch.Tensor] = []
    for index in selected_layers:
        student = student_hidden_states[index]
        teacher = teacher_hidden_states[index]
        if teacher_transform is not None:
            teacher = teacher_transform(teacher, index)
        if student.shape != teacher.shape or student.shape[:2] != labels.shape:
            raise ValueError("selected student and teacher hidden states must align")
        token_mse = (student.float() - teacher.float()).square().mean(dim=-1)
        per_sample = token_mse.masked_fill(~mask, 0.0).sum(dim=-1)
        layer_losses.append((per_sample / counts.to(per_sample.dtype)).mean())
    return torch.stack(layer_losses).mean(), selected_layers


def compute_qad_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: QADLossConfig | None = None,
    student_hidden_states: tuple[torch.Tensor, ...] | None = None,
    teacher_hidden_states: tuple[torch.Tensor, ...] | None = None,
    attention_mask: torch.Tensor | None = None,
    task_weights: torch.Tensor | None = None,
    kl_reverse_weights: torch.Tensor | None = None,
    teacher_hidden_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> QADLossOutput:
    config = config or QADLossConfig()
    mask = _causal_assistant_mask(labels)
    targets = labels[:, 1:][mask]
    token_task = F.cross_entropy(
        student_logits[:, :-1][mask].float(),
        targets,
        reduction="none",
    )
    if task_weights is None:
        task = token_task.mean()
    else:
        if task_weights.ndim != 1 or task_weights.shape[0] != labels.shape[0]:
            raise ValueError("task_weights must have shape [batch]")
        if torch.any(task_weights < 0):
            raise ValueError("task_weights must be non-negative")
        batch, sequence = mask.shape
        sample_grid = (
            torch.arange(batch, device=labels.device)[:, None]
            .expand(batch, sequence)
        )
        sample_indices = sample_grid[mask]
        counts = mask.sum(dim=-1)
        totals = torch.zeros(
            batch,
            device=token_task.device,
            dtype=token_task.dtype,
        )
        totals.index_add_(0, sample_indices, token_task)
        per_sample = totals / counts.to(token_task.dtype)
        weights = task_weights.to(device=token_task.device, dtype=token_task.dtype)
        weight_sum = weights.sum()
        task = torch.where(
            weight_sum > 0,
            (per_sample * weights).sum() / weight_sum.clamp_min(1.0),
            token_task.sum() * 0.0,
        )

    zero = task.new_zeros(())
    eakld = forward_kl = reverse_kl = teacher_entropy = entropy_lambda = zero
    if config.logit_alpha:
        if config.kl_mode == "policy":
            if kl_reverse_weights is None:
                raise ValueError("policy KL requires kl_reverse_weights")
            eakld, forward_kl, reverse_kl, teacher_entropy = (
                policy_mixed_bidirectional_kl(
                    student_logits,
                    teacher_logits,
                    labels,
                    kl_reverse_weights,
                    temperature=config.temperature,
                    token_chunk_size=config.token_chunk_size,
                )
            )
        else:
            eakld, forward_kl, reverse_kl, teacher_entropy = (
                entropy_adaptive_bidirectional_kl(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=config.temperature,
                    entropy_k=config.entropy_k,
                    token_chunk_size=config.token_chunk_size,
                )
            )
        entropy_lambda = (
            teacher_entropy / math.log(config.entropy_k)
        ).clamp(0.0, 1.0)
        if config.kl_mode == "forward":
            eakld = forward_kl
        elif config.kl_mode == "reverse":
            eakld = reverse_kl
        elif config.kl_mode == "symmetric":
            eakld = 0.5 * (forward_kl + reverse_kl)

    lafd = zero
    selected_layers: tuple[int, ...] = ()
    if config.feature_alpha:
        if student_hidden_states is None or teacher_hidden_states is None:
            raise ValueError("LAFD requires student and teacher hidden states")
        lafd, selected_layers = adaptive_layer_feature_distillation(
            student_hidden_states,
            teacher_hidden_states,
            labels,
            topk=config.feature_topk,
            attention_mask=attention_mask,
            teacher_transform=teacher_hidden_transform,
        )

    total = (
        config.task_alpha * task
        + config.logit_alpha * eakld
        + config.feature_alpha * lafd
    )
    return QADLossOutput(
        total=total,
        task=task,
        eakld=eakld,
        forward_kl=forward_kl,
        reverse_kl=reverse_kl,
        teacher_entropy=teacher_entropy,
        entropy_lambda=entropy_lambda,
        lafd=lafd,
        selected_layers=selected_layers,
        assistant_tokens=int(mask.sum().item()),
    )
