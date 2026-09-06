#!/usr/bin/env python
"""Train Qwen3-8B with exact HiF4 W4A4 quantization-aware distillation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Subset

from faquant.gptq import calibration_tokens
from faquant.qad import prepare_qad_student, prepare_qad_teacher
from faquant.qad_checkpoint import load_qad_student_checkpoint
from faquant.qad_data import AssistantOnlyDataCollator, QADJsonlDataset
from faquant.qad_losses import QADLossConfig, compute_qad_loss
from faquant.qad_quantization import (
    HiF4QATLinear,
    TrainingTensorStats,
    collect_hif4_qat_stats,
    configure_qat_trainable_scope,
    hif4_code_step_parameter_groups,
    materialize_training_tensors,
    set_hif4_qat_metadata_mode,
)
from faquant.qwen3 import DEFAULT_MODEL
from faquant.rotation import rotate_qwen3_hidden_state


def parse_layer_indices(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("layer indices must not contain duplicates")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("layer indices must be non-negative")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--train-data",
        default="data/qad/qwen3_8b_mmlu_longbench_train.jsonl",
    )
    parser.add_argument(
        "--validation-data",
        default="data/qad/qwen3_8b_mmlu_longbench_validation.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rotation",
        choices=("hadamard", "none", "hisq1024"),
        default="hadamard",
    )
    parser.add_argument("--loss-mode", choices=("ce", "eakld", "full"), default="full")
    parser.add_argument("--task-alpha", type=float)
    parser.add_argument("--logit-alpha", type=float)
    parser.add_argument("--feature-alpha", type=float)
    parser.add_argument(
        "--feature-topk",
        type=int,
        default=3,
        help="Number of adaptively selected transformer hidden states for LAFD.",
    )
    parser.add_argument(
        "--kl-mode",
        choices=("eakld", "forward", "reverse", "symmetric", "policy"),
        default="eakld",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--completed-steps",
        type=int,
        default=0,
        help=(
            "Optimizer steps already consumed by the student init. The run "
            "trains from this point up to --max-steps, and the cosine/linear "
            "schedule is advanced to the same place so a second phase keeps "
            "the original rate trajectory. Zero (default) is a start from the "
            "PTQ init."
        ),
    )
    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        help=(
            "Training horizon used only to derive warmup length. Defaults to "
            "--max-steps; set it when reproducing an early validation point."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--linear-lr-scaling",
        choices=("none", "hif4-code-step"),
        default="none",
        help=(
            "Optionally scale each QAT Linear learning rate by its median "
            "fixed-metadata HiF4 code step."
        ),
    )
    parser.add_argument("--linear-lr-scale-min", type=float, default=0.125)
    parser.add_argument("--linear-lr-scale-max", type=float, default=8.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine", "linear"),
        default="constant",
        help=(
            "Post-warmup shape. 'constant' reproduces every run before "
            "2026-08-06 and never anneals, so each checkpoint is a snapshot of "
            "a walk that is still moving; prefer cosine/linear decay over the "
            "real horizon when the checkpoint-to-checkpoint curve has to be "
            "readable."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--trainable-scope",
        choices=("qat-linears",),
        default="qat-linears",
    )
    parser.add_argument(
        "--activation-companding-group-size",
        type=int,
        default=1,
        help=(
            "Tie this many contiguous post-rotation channels to one reversible "
            "activation companding scale."
        ),
    )
    parser.add_argument(
        "--residual-affine-max-active-fraction",
        type=float,
        default=1.0,
        help=(
            "For legacy residual-affine QAD, cap the fraction of deployed "
            "BF16 gains that may leave the identity cell after each step."
        ),
    )
    parser.add_argument(
        "--trainable-layer-min",
        type=int,
        default=0,
        help=(
            "With qat-linears scope, freeze quantized transformer layers below "
            "this zero-based layer index."
        ),
    )
    parser.add_argument(
        "--metadata-mode",
        choices=("fixed", "dynamic"),
        default="fixed",
    )
    parser.add_argument(
        "--deployed-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the student's attention core on the deployed path during "
            "training: simulated kernel, HiF4 QK and PV matmul operands, and "
            "the post-RoPE Q/K Hadamard. Without this the student adapts its "
            "linear weights to an exact attention core, which is why rung-1's "
            "+1.04 pp linear-only gain halved on the full recipe."
        ),
    )
    parser.add_argument(
        "--qk-smooth-scales",
        type=Path,
        help="Frozen Smooth-QK artifact; only meaningful with --deployed-attention.",
    )
    parser.add_argument(
        "--train-qk-smooth-scales",
        action="store_true",
        help="Optimize the installed inverse-preserving Q/K channel scales.",
    )
    parser.add_argument(
        "--qk-smooth-only",
        action="store_true",
        help="Freeze every model weight and optimize only Smooth-QK scales.",
    )
    parser.add_argument(
        "--qk-smooth-learning-rate",
        type=float,
        help="Optional separate learning rate for Smooth-QK log-deltas.",
    )
    parser.add_argument(
        "--qk-mxfp8-layers",
        type=parse_layer_indices,
        default=(),
        help=(
            "Comma-separated decoder layers whose QK matmul uses block-32 "
            "MXFP8 E4M3 during deployed-attention training."
        ),
    )
    parser.add_argument(
        "--pv-mxfp8-layers",
        type=parse_layer_indices,
        default=(),
        help=(
            "Comma-separated decoder layers whose PV matmul uses block-32 "
            "MXFP8 E4M3 during deployed-attention training."
        ),
    )
    parser.add_argument(
        "--rollback-checkpoint",
        type=Path,
        help="Optional QAD checkpoint supplying selected projection weights.",
    )
    parser.add_argument(
        "--rollback-projections",
        default="",
        help="Comma-separated projection families restored before training.",
    )
    parser.add_argument(
        "--rollback-layers",
        type=parse_layer_indices,
        default=(),
        help="Optional layer subset for projection rollback.",
    )
    parser.add_argument(
        "--freeze-projections",
        default="",
        help="Comma-separated projection families frozen during QAD.",
    )
    parser.add_argument(
        "--freeze-layers",
        type=parse_layer_indices,
        default=(),
        help="Optional layer subset for projection freezing.",
    )
    parser.add_argument(
        "--pv-normalizer-mode",
        choices=("unquantized", "quantized_same"),
        default="quantized_same",
        help="Softmax denominator source; 'quantized_same' is P-Reordering.",
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="Evaluate this many held-out generic QAD examples after training.",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=0,
        help=(
            "Also evaluate the held-out generic QAD split every N optimizer "
            "steps. Zero keeps end-only validation."
        ),
    )
    parser.add_argument("--gptq-nsamples", type=int, default=128)
    parser.add_argument("--gptq-seqlen", type=int, default=2048)
    parser.add_argument(
        "--student-init",
        type=Path,
        help="Reusable sharded QAD step-0 checkpoint; skips GPTQ calibration.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--parity-tokens", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-fsdp", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    for name in (
        "max_steps",
        "microbatch_size",
        "gradient_accumulation_steps",
        "max_length",
        "gptq_nsamples",
        "gptq_seqlen",
        "log_steps",
        "parity_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("invalid learning rate or warmup ratio")
    if args.qk_smooth_only and not args.train_qk_smooth_scales:
        raise ValueError("--qk-smooth-only requires --train-qk-smooth-scales")
    if args.train_qk_smooth_scales and (
        not args.deployed_attention or args.qk_smooth_scales is None
    ):
        raise ValueError(
            "trainable Smooth-QK requires --deployed-attention and "
            "--qk-smooth-scales"
        )
    if (
        args.qk_smooth_learning_rate is not None
        and args.qk_smooth_learning_rate <= 0
    ):
        raise ValueError("--qk-smooth-learning-rate must be positive")
    if (
        args.qk_smooth_learning_rate is not None
        and not args.train_qk_smooth_scales
    ):
        raise ValueError(
            "--qk-smooth-learning-rate requires --train-qk-smooth-scales"
        )
    if not 0 < args.linear_lr_scale_min <= args.linear_lr_scale_max:
        raise ValueError("invalid linear learning-rate scale bounds")
    if args.linear_lr_scaling != "none" and args.trainable_scope != "qat-linears":
        raise ValueError("linear LR scaling requires --trainable-scope qat-linears")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.feature_topk <= 0:
        raise ValueError("--feature-topk must be positive")
    for name in ("task_alpha", "logit_alpha", "feature_alpha"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.validation_samples < 0:
        raise ValueError("--validation-samples must be non-negative")
    if args.trainable_layer_min < 0:
        raise ValueError("--trainable-layer-min must be non-negative")
    if args.trainable_layer_min and args.trainable_scope != "qat-linears":
        raise ValueError("--trainable-layer-min requires qat-linears scope")
    if args.validation_steps < 0:
        raise ValueError("--validation-steps must be non-negative")
    if args.validation_steps and not args.validation_samples:
        raise ValueError("--validation-steps requires --validation-samples")
    if not 0 < args.residual_affine_max_active_fraction <= 1:
        raise ValueError(
            "--residual-affine-max-active-fraction must be in (0, 1]"
        )
    if (
        args.residual_affine_max_active_fraction < 1
        and args.trainable_scope != "residual-affine-only"
    ):
        raise ValueError(
            "residual affine active cap requires "
            "--trainable-scope residual-affine-only"
        )
    if args.scheduler_total_steps is not None and args.scheduler_total_steps <= 0:
        raise ValueError("--scheduler-total-steps must be positive")
    if args.completed_steps < 0:
        raise ValueError("--completed-steps must be non-negative")
    if args.completed_steps >= args.max_steps:
        raise ValueError("--completed-steps must be smaller than --max-steps")
    return args


def advance_scheduler(scheduler, *, optimizer_steps: int, stride: int) -> int:
    """Wind a raw (unprepared) scheduler forward by ``optimizer_steps``.

    Accelerate later multiplies each optimizer step by ``stride``, so the
    matching advance is ``optimizer_steps * stride`` calls of ``scheduler.step``.
    Returns the scheduler's ``last_epoch`` afterwards, which is what a run that
    had actually taken those steps would show.
    """

    if optimizer_steps < 0:
        raise ValueError("optimizer_steps must be non-negative")
    if stride < 1:
        raise ValueError("stride must be positive")
    for _ in range(optimizer_steps * stride):
        scheduler.step()
    return int(scheduler.last_epoch)


def scheduler_stride_for(accelerator) -> int:
    """How many times accelerate advances a prepared scheduler per optimizer step.

    ``accelerator.prepare`` wraps the scheduler so that it steps once per
    process per optimizer step. A schedule handed the nominal horizon would
    therefore reach its end ``num_processes`` times too early -- the same quirk
    that made ``--warmup-ratio`` complete 8x sooner than intended on this host.
    """

    if getattr(accelerator, "split_batches", False):
        return 1
    return max(1, int(getattr(accelerator, "num_processes", 1) or 1))


def loss_config(
    mode: str,
    *,
    task_alpha: float | None = None,
    logit_alpha: float | None = None,
    feature_alpha: float | None = None,
    kl_mode: str = "eakld",
    temperature: float = 1.0,
) -> QADLossConfig:
    if mode == "ce":
        config = QADLossConfig(
            task_alpha=1.0,
            logit_alpha=0.0,
            feature_alpha=0.0,
        )
    elif mode == "eakld":
        config = QADLossConfig(
            task_alpha=0.05,
            logit_alpha=2.0,
            feature_alpha=0.0,
        )
    else:
        config = QADLossConfig(
            task_alpha=0.05,
            logit_alpha=2.0,
            feature_alpha=0.5,
        )
    return QADLossConfig(
        task_alpha=(
            config.task_alpha if task_alpha is None else task_alpha
        ),
        logit_alpha=(
            config.logit_alpha if logit_alpha is None else logit_alpha
        ),
        feature_alpha=(
            config.feature_alpha if feature_alpha is None else feature_alpha
        ),
        temperature=temperature,
        entropy_k=config.entropy_k,
        feature_topk=config.feature_topk,
        token_chunk_size=config.token_chunk_size,
        kl_mode=kl_mode,
    )


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_safe(value: object) -> object:
    """Recursively replace Path values so a receipt can always be written."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def save_student_checkpoint(
    accelerator: object,
    student: torch.nn.Module,
    tokenizer: object,
    checkpoint: Path,
    *,
    receipt: dict[str, object],
    max_shard_size: str,
) -> dict[str, int]:
    """Save only the QAD student, with BF16 parameters and exact FP32 metadata.

    ``Accelerator.save_state`` would also persist the equally large frozen
    teacher and the Adam state.  Those are unnecessary for inference and make
    each QAD ablation prohibitively large.  FSDP state gathering is collective,
    so every rank enters this function; only rank 0 receives and writes the
    full state dictionary.
    """

    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(student)
    state_dict = accelerator.get_state_dict(student)
    if accelerator.is_main_process:
        # Nested FSDP wrappers do not use the same names in
        # ``named_parameters`` as the gathered, prefix-free state dict.  Qwen
        # parameters are weights/biases, whereas HiF4 scale/reciprocal tensors
        # are persistent buffers and must stay FP32.
        for name, tensor in list(state_dict.items()):
            if (
                name.endswith((".weight", ".bias"))
                and tensor.is_floating_point()
            ):
                state_dict[name] = tensor.to(torch.bfloat16)
    # ``save_pretrained(is_main_process=False)`` still enters safetensors
    # serialization in current Transformers releases.  Under ordinary DDP
    # that made every rank race to write the same 19 GB checkpoint.  State
    # gathering above remains collective for FSDP; only rank 0 serializes.
    if accelerator.is_main_process:
        unwrapped.save_pretrained(
            checkpoint,
            is_main_process=True,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
    del state_dict
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "training_receipt.json", receipt)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return {}
    return {
        path.name: path.stat().st_size
        for path in sorted(checkpoint.glob("model*.safetensors"))
    }


def qat_gradient_coverage(
    model: torch.nn.Module,
    accelerator: object,
) -> tuple[int, int, int]:
    modules = [
        module for module in accelerator.unwrap_model(model).modules()
        if isinstance(module, HiF4QATLinear)
    ]
    flags = torch.zeros(
        (len(modules), 3),
        device=accelerator.device,
        dtype=torch.int32,
    )
    for index, module in enumerate(modules):
        gradient = module.weight.grad
        if gradient is None or gradient.numel() == 0:
            continue
        flags[index, 0] = 1
        flags[index, 1] = int(not bool(torch.isfinite(gradient).all()))
        flags[index, 2] = int(bool(gradient.abs().sum() > 0))
    flags = accelerator.reduce(flags, reduction="sum")
    present = flags[:, 0] > 0
    finite = present & flags[:, 1].eq(0)
    nonzero = flags[:, 2] > 0
    return (
        int(present.sum().item()),
        int(finite.sum().item()),
        int(nonzero.sum().item()),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from accelerate import (
        Accelerator,
        FullyShardedDataParallelPlugin,
    )
    from accelerate.utils import ProjectConfiguration, set_seed
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    fsdp_plugin = None
    if not args.no_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            backward_prefetch="BACKWARD_PRE",
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
            state_dict_type="SHARDED_STATE_DICT",
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
        )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    set_seed(args.seed, device_specific=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qad_config = loss_config(
        args.loss_mode,
        task_alpha=args.task_alpha,
        logit_alpha=args.logit_alpha,
        feature_alpha=args.feature_alpha,
        kl_mode=args.kl_mode,
        temperature=args.temperature,
    )
    qad_config = replace(qad_config, feature_topk=args.feature_topk)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = QADJsonlDataset(args.train_data)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    validation_dataset = QADJsonlDataset(args.validation_data)
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_length=args.max_length,
        max_answer_tokens=args.max_answer_tokens,
    )
    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    validation_dataloader = None
    if args.validation_samples:
        validation_subset = Subset(
            validation_dataset,
            range(min(args.validation_samples, len(validation_dataset))),
        )
        validation_dataloader = DataLoader(
            validation_subset,
            batch_size=args.microbatch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    parity_batch = collator([validation_dataset[0]])
    parity_batch = {
        key: value[:, : args.parity_tokens].to(accelerator.device)
        for key, value in parity_batch.items()
    }

    def load_model() -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": accelerator.device},
            low_cpu_mem_usage=True,
        )

    student = load_model()
    init_receipt = None
    if args.student_init is not None:
        init_receipt_path = args.student_init / "qad_init.json"
        training_receipt_path = args.student_init / "training_receipt.json"
        if init_receipt_path.exists():
            checkpoint_receipt = json.loads(
                init_receipt_path.read_text(encoding="utf-8")
            )
            init_receipt = checkpoint_receipt
            checkpoint_rotation = checkpoint_receipt["rotation"]
            student_step0_max_abs = float(
                checkpoint_receipt["step0_parity_max_abs"]
            )
        elif training_receipt_path.exists():
            checkpoint_receipt = json.loads(
                training_receipt_path.read_text(encoding="utf-8")
            )
            init_receipt = checkpoint_receipt
            checkpoint_rotation = checkpoint_receipt["args"]["rotation"]
            student_step0_max_abs = float(
                checkpoint_receipt["parity"]["student_step0_max_abs"]
            )
        else:
            raise FileNotFoundError(
                f"{args.student_init} contains neither qad_init.json nor "
                "training_receipt.json"
            )
        if checkpoint_rotation != args.rotation:
            raise ValueError(
                "student init rotation does not match the requested training run"
            )
        accelerator.print(f"Loading reusable QAD init from {args.student_init}")
        loaded = load_qad_student_checkpoint(
            student,
            args.student_init,
            rotation=args.rotation,
            seed=0,
        )
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(
                f"QAD init state mismatch: missing={loaded.missing_keys}, "
                f"unexpected={loaded.unexpected_keys}"
            )
        rollback_projections = tuple(
            item.strip()
            for item in args.rollback_projections.split(",")
            if item.strip()
        )
        valid_projections = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        invalid_rollbacks = sorted(
            set(rollback_projections) - valid_projections
        )
        if invalid_rollbacks:
            raise ValueError(
                f"unsupported rollback projections: {invalid_rollbacks}"
            )
        if rollback_projections:
            if args.rollback_checkpoint is None:
                raise ValueError(
                    "--rollback-projections requires --rollback-checkpoint"
                )
            index = json.loads(
                (
                    args.rollback_checkpoint / "model.safetensors.index.json"
                ).read_text(encoding="utf-8")
            )
            rollback_layer_set = (
                frozenset(range(36))
                if not args.rollback_layers
                else frozenset(args.rollback_layers)
            )
            selected: dict[str, list[str]] = {}
            for key, shard in index["weight_map"].items():
                if not key.endswith(".weight"):
                    continue
                if key.rsplit(".", 2)[-2] not in rollback_projections:
                    continue
                parts = key.split(".")
                if len(parts) < 4 or parts[0:2] != ["model", "layers"]:
                    continue
                if int(parts[2]) not in rollback_layer_set:
                    continue
                selected.setdefault(shard, []).append(key)
            restored = 0
            for shard, keys in selected.items():
                with safe_open(
                    args.rollback_checkpoint / shard,
                    framework="pt",
                    device="cpu",
                ) as tensors:
                    for key in keys:
                        module = student.get_submodule(key[: -len(".weight")])
                        module.weight.data.copy_(
                            tensors.get_tensor(key).to(
                                device=module.weight.device,
                                dtype=module.weight.dtype,
                            )
                        )
                        restored += 1
            expected_restored = len(rollback_layer_set) * len(
                rollback_projections
            )
            if restored != expected_restored:
                raise RuntimeError(
                    "rollback did not cover every requested projection: "
                    f"got {restored}, expected {expected_restored}"
                )
        qat_conversion = loaded.qat
    else:
        accelerator.print(
            f"Preparing {args.rotation} HiF4 GPTQ student on "
            f"{accelerator.num_processes} processes "
            f"({args.gptq_nsamples}x{args.gptq_seqlen} calibration)"
        )
        calibration_input_ids = calibration_tokens(
            tokenizer,
            nsamples=args.gptq_nsamples,
            seqlen=args.gptq_seqlen,
            seed=0,
        )
        from faquant.qad import qwen3_hif4_qad_config
        from faquant.qwen3 import prepare_model

        student_config = qwen3_hif4_qad_config(
            rotation=args.rotation,
            gptq_nsamples=args.gptq_nsamples,
            gptq_seqlen=args.gptq_seqlen,
            seed=0,
        )
        prepare_model(
            student,
            student_config,
            calibration_input_ids=calibration_input_ids,
        )
        with torch.no_grad():
            inference_step0 = student(
                input_ids=parity_batch["input_ids"],
                attention_mask=parity_batch["attention_mask"],
                use_cache=False,
            ).logits
        from faquant.qad_quantization import enable_hif4_qat

        with torch.inference_mode(False):
            qat_conversion = enable_hif4_qat(student)
        student.eval()
        with torch.no_grad():
            qat_step0 = student(
                input_ids=parity_batch["input_ids"],
                attention_mask=parity_batch["attention_mask"],
                use_cache=False,
            ).logits
        student_step0_max_abs = float(
            (qat_step0.float() - inference_step0.float()).abs().max().item()
        )
        if student_step0_max_abs != 0.0:
            raise RuntimeError(
                f"QAT step-0 parity failed: max_abs={student_step0_max_abs}"
            )
        del qat_step0, inference_step0, calibration_input_ids
    torch.cuda.empty_cache()

    # Configured after the step-0 parity check so parity keeps measuring the
    # student against its own inference modules on the same attention path.
    deployed_attention_metadata: dict[str, object] | None = None
    if args.deployed_attention:
        # Previously staged-but-unrunnable: the tiled kernel accumulates its
        # running max, denominator and output in place, which autograd rejects.
        # faquant.qad_attention re-derives the same arithmetic in one tile,
        # out of place -- exactly equal to the tiled kernel once a tile covers
        # every key, which tests/test_qad_attention.py pins against the
        # simulator.  The other blocker on record was memory, measured when a
        # tiled simulator was going to run inside the training graph; the
        # single-tile score matrix costs ~0.77 GB across 36 layers at the
        # 300-token corpus cap, so it is now a matter of measurement rather
        # than of principle.  Section 11 has why this matters: the attention
        # chain is 46% of the deployed gap and half of its run-to-run noise,
        # and training has never seen any of it.
        from faquant.qwen3 import (
            apply_qwen3_qk_smooth_scales,
            configure_qwen3_attention_runtime,
            promote_qwen3_qk_smooth_scales,
        )

        qk_smooth_metadata = None
        if args.qk_smooth_scales is not None:
            qk_smooth_metadata = apply_qwen3_qk_smooth_scales(
                student, args.qk_smooth_scales
            )
        trainable_qk_smooth_scales = 0
        if args.train_qk_smooth_scales:
            trainable_qk_smooth_scales = promote_qwen3_qk_smooth_scales(student)
        configure_qwen3_attention_runtime(
            student,
            kernel="differentiable",
            input_quant=False,
            output_quant=True,
            qk_matmul_quant=True,
            pv_matmul_quant=True,
            post_rope_qk_rotation=True,
            pv_normalizer_mode=args.pv_normalizer_mode,
            qk_mxfp8_layers=args.qk_mxfp8_layers,
            pv_mxfp8_layers=args.pv_mxfp8_layers,
        )
        deployed_attention_metadata = {
            "attention_kernel": "differentiable",
            "qk_matmul_quant": True,
            "pv_matmul_quant": True,
            "post_rope_qk_rotation": True,
            "pv_normalizer_mode": args.pv_normalizer_mode,
            "qk_mxfp8_layers": list(args.qk_mxfp8_layers),
            "pv_mxfp8_layers": list(args.pv_mxfp8_layers),
            "qk_smooth_scales": (
                None if args.qk_smooth_scales is None else str(args.qk_smooth_scales)
            ),
            "qk_smooth_metadata": qk_smooth_metadata,
            "trainable_qk_smooth_scales": trainable_qk_smooth_scales,
        }
        accelerator.print(
            "student attention core running on the deployed path "
            f"(differentiable kernel, pv_normalizer_mode={args.pv_normalizer_mode}, "
            f"qk_mxfp8_layers={list(args.qk_mxfp8_layers)}, "
            f"pv_mxfp8_layers={list(args.pv_mxfp8_layers)})"
        )
    elif args.qk_smooth_scales is not None:
        raise ValueError("--qk-smooth-scales requires --deployed-attention")
    elif args.qk_mxfp8_layers or args.pv_mxfp8_layers:
        raise ValueError("MXFP8 attention layers require --deployed-attention")

    metadata_modules = set_hif4_qat_metadata_mode(student, args.metadata_mode)
    metadata_step0_max_abs = 0.0
    if args.metadata_mode == "dynamic":
        set_hif4_qat_metadata_mode(student, "fixed")
        student.eval()
        with torch.no_grad():
            fixed_logits = student(
                input_ids=parity_batch["input_ids"],
                attention_mask=parity_batch["attention_mask"],
                use_cache=False,
            ).logits
        set_hif4_qat_metadata_mode(student, "dynamic")
        with torch.no_grad():
            dynamic_logits = student(
                input_ids=parity_batch["input_ids"],
                attention_mask=parity_batch["attention_mask"],
                use_cache=False,
            ).logits
        metadata_step0_max_abs = float(
            (fixed_logits.float() - dynamic_logits.float()).abs().max().item()
        )
        del fixed_logits, dynamic_logits

    teacher = None
    teacher_final_norm_scale = None
    rotation_signs = None
    if qad_config.logit_alpha or qad_config.feature_alpha:
        accelerator.print("Loading and aligning frozen BF16 teacher")
        teacher = load_model()
        prepare_qad_teacher(teacher, rotation=args.rotation, seed=0)
        if qad_config.feature_alpha and args.rotation == "hadamard":
            teacher_final_norm_scale = teacher.model.norm.weight.detach().clone()
            rotation_signs = student.faquant_rotation_signs.detach().clone()
    else:
        accelerator.print("CE-only mode: frozen teacher is not loaded")
    del parity_batch
    torch.cuda.empty_cache()

    student.train()
    student.config.use_cache = False
    trainable_scope_stats = configure_qat_trainable_scope(
        student,
        args.trainable_scope,
        minimum_transformer_layer=args.trainable_layer_min,
        activation_companding_group_size=args.activation_companding_group_size,
    )
    if args.train_qk_smooth_scales:
        from faquant.qwen3 import promote_qwen3_qk_smooth_scales

        if args.qk_smooth_only:
            student.requires_grad_(False)
        promote_qwen3_qk_smooth_scales(student)
    freeze_projections = tuple(
        item.strip() for item in args.freeze_projections.split(",") if item.strip()
    )
    valid_freeze_projections = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    invalid_frozen = sorted(
        set(freeze_projections) - valid_freeze_projections
    )
    if invalid_frozen:
        raise ValueError(
            f"unsupported frozen projections: {invalid_frozen}"
        )
    frozen_projection_weights = 0
    freeze_layer_set = (
        frozenset(range(36))
        if not args.freeze_layers
        else frozenset(args.freeze_layers)
    )
    for name, module in student.named_modules():
        parts = name.split(".")
        layer_index = (
            int(parts[2])
            if len(parts) >= 4 and parts[0:2] == ["model", "layers"]
            else None
        )
        if (
            isinstance(module, HiF4QATLinear)
            and name.rsplit(".", 1)[-1] in freeze_projections
            and layer_index in freeze_layer_set
        ):
            module.weight.requires_grad_(False)
            frozen_projection_weights += 1
    expected_frozen = len(freeze_layer_set) * len(freeze_projections)
    if frozen_projection_weights != expected_frozen:
        raise RuntimeError(
            "freeze did not cover every requested projection: "
            f"got {frozen_projection_weights}, expected {expected_frozen}"
        )
    training_tensor_stats = materialize_training_tensors(student)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    teacher_training_tensor_stats = TrainingTensorStats(0, 0)
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)
        teacher_training_tensor_stats = materialize_training_tensors(teacher)
    lr_scale_stats = None
    optimizer_parameters: object = (
        parameter for parameter in student.parameters() if parameter.requires_grad
    )
    if args.qk_smooth_learning_rate is not None:
        scale_parameters = []
        other_parameters = []
        for name, parameter in student.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("faquant_qk_smooth_delta"):
                scale_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        if len(scale_parameters) != 36:
            raise RuntimeError(
                f"expected 36 trainable Smooth-QK deltas, got "
                f"{len(scale_parameters)}"
            )
        optimizer_parameters = []
        if other_parameters:
            optimizer_parameters.append(
                {"params": other_parameters, "lr": args.learning_rate}
            )
        optimizer_parameters.append(
            {
                "params": scale_parameters,
                "lr": args.qk_smooth_learning_rate,
                "weight_decay": 0.0,
            }
        )
    if args.linear_lr_scaling == "hif4-code-step":
        optimizer_parameters, lr_scale_stats = hif4_code_step_parameter_groups(
            student,
            base_lr=args.learning_rate,
            minimum_factor=args.linear_lr_scale_min,
            maximum_factor=args.linear_lr_scale_max,
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler_total_steps = args.scheduler_total_steps or args.max_steps
    warmup_steps = math.ceil(scheduler_total_steps * args.warmup_ratio)
    scheduler_stride = scheduler_stride_for(accelerator)
    if args.lr_schedule == "constant":
        # Historical path: warmup is *not* multiplied by the stride, so it
        # actually completes num_processes times sooner than --warmup-ratio
        # implies. Left as-is so every pre-2026-08-06 receipt stays reproducible.
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        decay = (
            get_cosine_schedule_with_warmup
            if args.lr_schedule == "cosine"
            else get_linear_schedule_with_warmup
        )
        scheduler = decay(
            optimizer,
            num_warmup_steps=warmup_steps * scheduler_stride,
            num_training_steps=scheduler_total_steps * scheduler_stride,
        )
    if args.completed_steps:
        # Advance the raw scheduler before accelerate wraps it.  The wrapper
        # only steps on gradient-sync, so doing this afterwards would be a
        # no-op and the second phase would restart warmup.
        advance_scheduler(
            scheduler,
            optimizer_steps=args.completed_steps,
            stride=scheduler_stride,
        )
    if teacher is None:
        student, optimizer, dataloader, scheduler = accelerator.prepare(
            student,
            optimizer,
            dataloader,
            scheduler,
        )
    else:
        student, teacher, optimizer, dataloader, scheduler = accelerator.prepare(
            student,
            teacher,
            optimizer,
            dataloader,
            scheduler,
        )
    if validation_dataloader is not None:
        validation_dataloader = accelerator.prepare(validation_dataloader)

    manifest_path = Path(args.train_data).parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    receipt = {
        "args": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "student_init": (
                None if args.student_init is None else str(args.student_init)
            ),
        },
        "git_commit": git_commit(),
        "world_size": accelerator.num_processes,
        "global_batch_examples": (
            accelerator.num_processes
            * args.microbatch_size
            * args.gradient_accumulation_steps
        ),
        "loss": asdict(qad_config),
        "scheduler": {
            "name": (
                "constant_with_warmup"
                if args.lr_schedule == "constant"
                else f"{args.lr_schedule}_with_warmup"
            ),
            "total_steps_for_warmup": scheduler_total_steps,
            "warmup_steps": warmup_steps,
            "accelerate_stride": scheduler_stride,
            "effective_warmup_optimizer_steps": (
                math.ceil(warmup_steps / scheduler_stride)
                if args.lr_schedule == "constant"
                else warmup_steps
            ),
            "anneals_to_zero_at_optimizer_step": (
                None if args.lr_schedule == "constant" else scheduler_total_steps
            ),
            "completed_steps": args.completed_steps,
            "linear_lr_scaling": args.linear_lr_scaling,
            "linear_lr_scale_stats": (
                None if lr_scale_stats is None else asdict(lr_scale_stats)
            ),
        },
        "data_manifest": manifest,
        "deployed_attention": deployed_attention_metadata,
        "student_init_receipt": init_receipt,
        "parity": {
            "student_step0_max_abs": student_step0_max_abs,
            "teacher_logits_basis": "original_bf16",
            "lafd_hidden_basis": (
                "teacher_hidden_times_diag_sign_hadamard"
                if rotation_signs is not None
                else "original"
            ),
            "qat_linears": qat_conversion.converted,
            "metadata_mode": args.metadata_mode,
            "metadata_modules": metadata_modules,
            "metadata_step0_max_abs": metadata_step0_max_abs,
            "trainable_scope": args.trainable_scope,
            "trainable_scope_stats": asdict(trainable_scope_stats),
            "training_tensors": asdict(training_tensor_stats),
            "teacher_training_tensors": asdict(teacher_training_tensor_stats),
        },
        "versions": {
            "torch": torch.__version__,
        },
        "status": "running",
    }
    if accelerator.is_main_process:
        write_json(args.output_dir / "receipt.json", receipt)

    log_path = args.output_dir / f"train_rank{accelerator.process_index}.jsonl"
    validation_log_path = args.output_dir / "validation.jsonl"

    def evaluate_validation() -> dict[str, float | int] | None:
        if validation_dataloader is None:
            return None
        student.eval()
        validation_sums = {
            name: torch.zeros((), device=accelerator.device, dtype=torch.float64)
            for name in ("total", "task", "eakld", "forward_kl", "reverse_kl", "lafd")
        }
        diagnostic_sums = {
            name: torch.zeros((), device=accelerator.device, dtype=torch.float64)
            for name in ("eakld", "forward_kl", "reverse_kl")
        }
        diagnostic_config = replace(
            qad_config,
            task_alpha=0.0,
            logit_alpha=1.0,
            feature_alpha=0.0,
            temperature=1.0,
            kl_mode="eakld",
        )
        validation_batches = torch.zeros(
            (), device=accelerator.device, dtype=torch.float64
        )
        with torch.no_grad():
            for batch in validation_dataloader:
                needs_hidden = qad_config.feature_alpha > 0
                teacher_outputs = None
                if teacher is not None:
                    teacher_outputs = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        output_hidden_states=needs_hidden,
                    )
                student_outputs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    output_hidden_states=needs_hidden,
                )
                losses = compute_qad_loss(
                    student_outputs.logits,
                    (
                        student_outputs.logits.detach()
                        if teacher_outputs is None
                        else teacher_outputs.logits
                    ),
                    batch["labels"],
                    config=qad_config,
                    student_hidden_states=student_outputs.hidden_states,
                    teacher_hidden_states=(
                        None
                        if teacher_outputs is None
                        else teacher_outputs.hidden_states
                    ),
                    attention_mask=batch["attention_mask"],
                )
                for name in validation_sums:
                    validation_sums[name] += getattr(losses, name).double()
                diagnostics = compute_qad_loss(
                    student_outputs.logits,
                    (
                        student_outputs.logits.detach()
                        if teacher_outputs is None
                        else teacher_outputs.logits
                    ),
                    batch["labels"],
                    config=diagnostic_config,
                )
                for name in diagnostic_sums:
                    diagnostic_sums[name] += getattr(diagnostics, name).double()
                validation_batches += 1
        validation_batches = accelerator.reduce(
            validation_batches, reduction="sum"
        )
        metrics: dict[str, float | int] = {
            name: float(
                accelerator.reduce(value, reduction="sum")
                .div(validation_batches)
                .item()
            )
            for name, value in validation_sums.items()
        }
        metrics.update(
            {
                f"diagnostic_t1_{name}": float(
                    accelerator.reduce(value, reduction="sum")
                    .div(validation_batches)
                    .item()
                )
                for name, value in diagnostic_sums.items()
            }
        )
        metrics["samples_requested"] = args.validation_samples
        student.train()
        return metrics

    global_step = args.completed_steps
    micro_step = 0
    examples_seen = 0
    assistant_tokens_seen = 0
    started = time.time()
    data_iterator = iter(dataloader)
    accumulated_metrics: dict[str, torch.Tensor] = {}
    accumulated_microbatches = 0
    validation_history: list[dict[str, object]] = []
    validation_metrics = None
    last_validation_step = 0
    if args.validation_steps:
        validation_metrics = evaluate_validation()
        validation_event = {
            "step": 0,
            "elapsed_seconds": time.time() - started,
            **(validation_metrics or {}),
        }
        validation_history.append(validation_event)
        if accelerator.is_main_process:
            with validation_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_event, sort_keys=True) + "\n")
        accelerator.print(
            "validation " + json.dumps(validation_event, sort_keys=True)
        )
    last_residual_projection = None
    while global_step < args.max_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
        micro_step += 1
        examples_seen += int(batch["input_ids"].shape[0])

        with accelerator.accumulate(student):
            needs_hidden = qad_config.feature_alpha > 0
            teacher_outputs = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        output_hidden_states=needs_hidden,
                    )
            student_outputs = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=needs_hidden,
            )
            losses = compute_qad_loss(
                student_outputs.logits,
                (
                    student_outputs.logits.detach()
                    if teacher_outputs is None
                    else teacher_outputs.logits
                ),
                batch["labels"],
                config=qad_config,
                student_hidden_states=student_outputs.hidden_states,
                teacher_hidden_states=(
                    None
                    if teacher_outputs is None
                    else teacher_outputs.hidden_states
                ),
                attention_mask=batch["attention_mask"],
                task_weights=batch.get("task_weights"),
                kl_reverse_weights=batch.get("kl_reverse_weights"),
                teacher_hidden_transform=(
                    (
                        lambda hidden, layer_index: rotate_qwen3_hidden_state(
                            (
                                hidden
                                / teacher_final_norm_scale.to(
                                    device=hidden.device,
                                    dtype=hidden.dtype,
                                )
                                if layer_index
                                == len(teacher_outputs.hidden_states) - 1
                                else hidden
                            ),
                            rotation_signs,
                        )
                    )
                    if rotation_signs is not None
                    else None
                ),
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"non-finite QAD loss at micro step {micro_step}"
                )
            assistant_tokens_seen += losses.assistant_tokens
            for key, value in {
                "total": losses.total,
                "task": losses.task,
                "eakld": losses.eakld,
                "forward_kl": losses.forward_kl,
                "reverse_kl": losses.reverse_kl,
                "teacher_entropy": losses.teacher_entropy,
                "entropy_lambda": losses.entropy_lambda,
                "lafd": losses.lafd,
            }.items():
                accumulated_metrics[key] = (
                    accumulated_metrics.get(key, value.detach().new_zeros(()))
                    + value.detach()
                )
            accumulated_microbatches += 1
            accelerator.backward(losses.total)
            if accelerator.sync_gradients:
                (
                    qat_grad_present,
                    qat_grad_finite,
                    qat_grad_nonzero,
                ) = qat_gradient_coverage(student, accelerator)
                grad_norm = accelerator.clip_grad_norm_(
                    student.parameters(),
                    args.max_grad_norm,
                )
            else:
                qat_grad_present = qat_grad_finite = qat_grad_nonzero = 0
                grad_norm = losses.total.new_zeros(())
            optimizer.step()
            if (
                accelerator.sync_gradients
                and args.residual_affine_max_active_fraction < 1
            ):
                from faquant.residual_affine import (
                    project_qwen3_residual_affine,
                )

                last_residual_projection = project_qwen3_residual_affine(
                    accelerator.unwrap_model(student),
                    maximum_active_fraction=(
                        args.residual_affine_max_active_fraction
                    ),
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        global_step += 1
        values = {
            key: value / accumulated_microbatches
            for key, value in accumulated_metrics.items()
        } | {"grad_norm": grad_norm.detach()}
        reduced = {
            key: float(accelerator.reduce(value, reduction="mean").item())
            for key, value in values.items()
        }
        event = {
            "step": global_step,
            "micro_step": micro_step,
            "elapsed_seconds": time.time() - started,
            "examples_seen_local": examples_seen,
            "assistant_tokens_seen_local": assistant_tokens_seen,
            "selected_layers": list(losses.selected_layers),
            "qat_grad_present": qat_grad_present,
            "qat_grad_finite": qat_grad_finite,
            "qat_grad_nonzero": qat_grad_nonzero,
            "learning_rate": scheduler.get_last_lr()[0],
            "learning_rate_min": min(scheduler.get_last_lr()),
            "learning_rate_max": max(scheduler.get_last_lr()),
            **reduced,
        }
        if last_residual_projection is not None:
            event["residual_affine_projection"] = asdict(
                last_residual_projection
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        if global_step % args.log_steps == 0:
            accelerator.print(json.dumps(event, sort_keys=True))
        accumulated_metrics = {}
        accumulated_microbatches = 0

        if args.validation_steps and global_step % args.validation_steps == 0:
            validation_metrics = evaluate_validation()
            last_validation_step = global_step
            validation_event = {
                "step": global_step,
                "elapsed_seconds": time.time() - started,
                **(validation_metrics or {}),
            }
            validation_history.append(validation_event)
            if accelerator.is_main_process:
                with validation_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation_event, sort_keys=True) + "\n")
            accelerator.print(
                "validation " + json.dumps(validation_event, sort_keys=True)
            )

        if (
            not args.no_save
            and args.save_steps
            and global_step % args.save_steps == 0
        ):
            checkpoint = args.output_dir / f"student-step-{global_step}"
            save_student_checkpoint(
                accelerator,
                student,
                tokenizer,
                checkpoint,
                receipt=receipt
                | {
                    "status": "intermediate",
                    "completed_steps": global_step,
                    "elapsed_seconds": time.time() - started,
                },
                max_shard_size=args.max_shard_size,
            )

    if validation_dataloader is not None and last_validation_step != global_step:
        validation_metrics = evaluate_validation()
        last_validation_step = global_step
        validation_history.append(
            {
                "step": global_step,
                "elapsed_seconds": time.time() - started,
                **(validation_metrics or {}),
            }
        )

    stats = collect_hif4_qat_stats(accelerator.unwrap_model(student))
    receipt["status"] = "completed"
    receipt["completed_steps"] = global_step
    receipt["elapsed_seconds"] = time.time() - started
    receipt["qat_forward_stats"] = asdict(stats)
    receipt["validation"] = validation_metrics
    receipt["validation_history"] = validation_history
    receipt["residual_affine_projection"] = (
        None
        if last_residual_projection is None
        else asdict(last_residual_projection)
    )
    if not args.no_save and args.save_steps == 0:
        checkpoint = args.output_dir / f"student-step-{global_step}"
        files = save_student_checkpoint(
            accelerator,
            student,
            tokenizer,
            checkpoint,
            receipt=receipt,
            max_shard_size=args.max_shard_size,
        )
        if accelerator.is_main_process:
            receipt["student_checkpoint"] = {
                "path": str(checkpoint),
                "parameter_dtype": "torch.bfloat16",
                "fixed_hif4_metadata_dtype": "preserved",
                "files": files,
            }
    if accelerator.is_main_process:
        write_json(args.output_dir / "receipt.json", receipt)
    accelerator.wait_for_everyone()
    accelerator.print(
        f"QAD completed: {global_step} steps in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    # Avoid a site-local mirror that does not host the QAD data dependencies.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
