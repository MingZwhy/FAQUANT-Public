#!/usr/bin/env python
"""Evaluate a saved Qwen3 HiF4 QAD student with the canonical lm-eval path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from faquant.qad_checkpoint import (
    load_qad_student_checkpoint,
    recorded_quant_exempt_layers,
    recorded_quant_exempt_modules,
)
from faquant.qad_quantization import (
    convert_hif4_qat_to_inference,
    set_hif4_qat_metadata_mode,
)
from faquant.quantization import FakeQuantLinear
from faquant.qwen3 import (
    DEFAULT_MODEL,
    apply_qwen3_qk_smooth_scales,
    collect_qwen3_simulated_attention_stats,
    configure_qwen3_attention_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--rotation",
        choices=("hadamard", "none", "hisq1024"),
        required=True,
    )
    parser.add_argument("--tasks", default="mmlu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="16")
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument(
        "--attention-kernel",
        choices=("native", "simulated"),
        default="simulated",
        help=(
            "Use simulated for exact comparison with the 70.5597%% simfa_none "
            "MMLU setting; native is useful for practical LongBench runs."
        ),
    )
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--key-chunk-size", type=int, default=128)
    parser.add_argument(
        "--qk-matmul-quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fake-HiF4 quantize tiled Q and K operands before QK matmul.",
    )
    parser.add_argument(
        "--pv-matmul-quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fake-HiF4 quantize tiled P and V operands before PV matmul.",
    )
    parser.add_argument(
        "--qk-exempt-layers",
        default="",
        help=(
            "Comma-separated decoder indices whose QK matmul stays in high "
            "precision while every other layer stays quantized."
        ),
    )
    parser.add_argument(
        "--pv-exempt-layers",
        default="",
        help="Same as --qk-exempt-layers, for the PV matmul.",
    )
    parser.add_argument(
        "--qk-mxfp8-layers",
        default="",
        help="Comma-separated QK layers quantized with MXFP8 E4M3.",
    )
    parser.add_argument(
        "--pv-mxfp8-layers",
        default="",
        help="Comma-separated PV layers quantized with MXFP8 E4M3.",
    )
    parser.add_argument(
        "--post-rope-qk-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the same normalized Hadamard to Q and K after RoPE and "
            "before KV-cache insertion/tiled QK quantization."
        ),
    )
    parser.add_argument(
        "--qk-smooth-scales",
        type=Path,
        help=(
            "Frozen Smooth-QK artifact from scripts/calibrate_qk_smooth_scales.py. "
            "Migrates the post-RoPE channel imbalance out of K and into Q without "
            "changing the scores, so it only matters when Q/K are quantized."
        ),
    )
    parser.add_argument(
        "--pv-normalizer-mode",
        choices=("unquantized", "quantized_same"),
        default="unquantized",
        help=(
            "Softmax denominator source. 'quantized_same' is P-Reordering, whose "
            "operator-level gain grows once Smooth-QK pushes the output norm "
            "further below the reference."
        ),
    )
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rollback-checkpoint",
        type=Path,
        help="Optional QAD checkpoint supplying selected projection weights.",
    )
    parser.add_argument(
        "--rollback-projections",
        default="",
        help=(
            "Comma-separated projection families restored from "
            "--rollback-checkpoint: q_proj,k_proj,v_proj,o_proj,gate_proj,"
            "up_proj,down_proj."
        ),
    )
    parser.add_argument(
        "--rollback-layers",
        default="",
        help="Optional comma-separated layer subset for projection rollback.",
    )
    parser.add_argument(
        "--blend-checkpoint",
        type=Path,
        help="Optional QAD checkpoint blended into the loaded projection weights.",
    )
    parser.add_argument("--blend-alpha", type=float, default=0.0)
    parser.add_argument(
        "--blend-projections",
        default="",
        help="Projection subset to blend; empty selects all seven families.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=("auto", "fixed", "dynamic"),
        default="auto",
        help="Auto reads the saved training receipt.",
    )
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="Skip the slow 252-linear SHA-256 pass for screening evaluations.",
    )
    return parser.parse_args()


def quantized_weight_sha256(model: torch.nn.Module) -> tuple[str, int]:
    """Hash frozen fake-quant weights using the historical ablation protocol."""

    digest = hashlib.sha256()
    count = 0
    chunk_elements = 1 << 20
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantLinear):
            continue
        count += 1
        weight = module.weight.detach().contiguous().view(-1)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tuple(module.weight.shape)).encode())
        digest.update(b"\0")
        digest.update(str(module.weight.dtype).encode())
        digest.update(b"\0")
        for start in range(0, weight.numel(), chunk_elements):
            raw = weight[start : start + chunk_elements].view(torch.uint8).cpu()
            digest.update(raw.numpy().tobytes())
    if count == 0:
        raise RuntimeError("no FakeQuantLinear weights were found for hashing")
    return digest.hexdigest(), count


def _expand_protected_modules(
    layers: tuple[int, ...], modules: tuple[str, ...]
) -> frozenset[tuple[int, str]]:
    """Turn a checkpoint's two protection lists into (layer, projection) pairs."""

    from faquant.config import QWEN_PROJECTIONS

    pairs = {(index, name) for index in layers for name in QWEN_PROJECTIONS}
    for entry in modules:
        layer_text, _, projection = entry.partition(".")
        pairs.add((int(layer_text), projection))
    return frozenset(pairs)


def _layer_list(raw: str, flag: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"{flag} takes comma-separated integers, got {raw!r}") from error
    if len(set(values)) != len(values):
        raise ValueError(f"{flag} lists a layer twice: {raw!r}")
    return values


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ["HF_ENDPOINT"] = os.environ.get(
        "FAQUANT_HF_ENDPOINT", "https://huggingface.co"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    receipt_path = args.checkpoint / "training_receipt.json"
    training_receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.exists()
        else {}
    )
    trainable_input_scales = (
        training_receipt.get("args", {}).get("trainable_scope")
        == "input-scales-only"
    )
    # Whole-layer exemptions are a property of the checkpoint, not a runtime
    # choice: those layers were built with plain Linear modules. Read them back
    # so both the module-count check and the attention core agree with how the
    # checkpoint was actually built.
    whole_exempt = recorded_quant_exempt_layers(args.checkpoint)
    protected_modules = _expand_protected_modules(
        whole_exempt, recorded_quant_exempt_modules(args.checkpoint)
    )
    loaded = load_qad_student_checkpoint(
        model,
        args.checkpoint,
        rotation=args.rotation,
        seed=0,
        trainable_input_scales=trainable_input_scales,
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"QAD checkpoint state mismatch: missing={loaded.missing_keys}, "
            f"unexpected={loaded.unexpected_keys}"
        )
    rollback_projections = tuple(
        item.strip() for item in args.rollback_projections.split(",") if item.strip()
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
    invalid_rollbacks = sorted(set(rollback_projections) - valid_projections)
    if invalid_rollbacks:
        raise ValueError(f"unsupported rollback projections: {invalid_rollbacks}")
    rollback_weights = 0
    rollback_layers = _layer_list(args.rollback_layers, "--rollback-layers")
    rollback_layer_set = (
        frozenset(range(36)) if not rollback_layers else frozenset(rollback_layers)
    )
    if rollback_projections:
        if args.rollback_checkpoint is None:
            raise ValueError(
                "--rollback-projections requires --rollback-checkpoint"
            )
        index = json.loads(
            (args.rollback_checkpoint / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
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
        for shard, keys in selected.items():
            with safe_open(
                args.rollback_checkpoint / shard,
                framework="pt",
                device="cpu",
            ) as tensors:
                for key in keys:
                    module = model.get_submodule(key[: -len(".weight")])
                    module.weight.data.copy_(
                        tensors.get_tensor(key).to(
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                        )
                    )
                    rollback_weights += 1
        expected_rollback_weights = len(rollback_layer_set) * len(
            rollback_projections
        )
        if rollback_weights != expected_rollback_weights:
            raise RuntimeError(
                "rollback did not cover every requested projection: "
                f"got {rollback_weights}, expected {expected_rollback_weights}"
            )
    blend_projections = tuple(
        item.strip() for item in args.blend_projections.split(",") if item.strip()
    )
    if not blend_projections:
        blend_projections = tuple(sorted(valid_projections))
    invalid_blends = sorted(set(blend_projections) - valid_projections)
    if invalid_blends:
        raise ValueError(f"unsupported blend projections: {invalid_blends}")
    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("--blend-alpha must be in [0, 1]")
    blended_weights = 0
    if args.blend_checkpoint is not None and args.blend_alpha > 0.0:
        blend_index = json.loads(
            (args.blend_checkpoint / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        selected_blends: dict[str, list[str]] = {}
        for key, shard in blend_index["weight_map"].items():
            if not key.endswith(".weight"):
                continue
            if key.rsplit(".", 2)[-2] not in blend_projections:
                continue
            parts = key.split(".")
            if len(parts) < 4 or parts[0:2] != ["model", "layers"]:
                continue
            selected_blends.setdefault(shard, []).append(key)
        for shard, keys in selected_blends.items():
            with safe_open(
                args.blend_checkpoint / shard,
                framework="pt",
                device="cpu",
            ) as tensors:
                for key in keys:
                    module = model.get_submodule(key[: -len(".weight")])
                    other = tensors.get_tensor(key).to(
                        device=module.weight.device,
                        dtype=torch.float32,
                    )
                    module.weight.data.copy_(
                        torch.lerp(
                            module.weight.data.float(),
                            other,
                            args.blend_alpha,
                        ).to(module.weight.dtype)
                    )
                    blended_weights += 1
        expected_blended = 36 * len(blend_projections)
        if blended_weights != expected_blended:
            raise RuntimeError(
                f"blend covered {blended_weights} weights, expected "
                f"{expected_blended}"
            )
    checkpoint_has_trainable_qk_smooth = bool(
        training_receipt.get("args", {}).get(
            "train_qk_smooth_scales", False
        )
    )
    trained_base_scales = None
    if checkpoint_has_trainable_qk_smooth:
        trained_base_scales = args.qk_smooth_scales
        if trained_base_scales is None:
            recorded_scales = training_receipt.get("args", {}).get(
                "qk_smooth_scales"
            )
            trained_base_scales = (
                None if recorded_scales is None else Path(recorded_scales)
            )
        if trained_base_scales is None:
            raise ValueError(
                "trained Smooth-QK checkpoint is missing its base scale artifact"
            )
        apply_qwen3_qk_smooth_scales(model, trained_base_scales)
    metadata_mode = args.metadata_mode
    if metadata_mode == "auto":
        if receipt_path.exists():
            metadata_mode = training_receipt.get("args", {}).get(
                "metadata_mode", "fixed"
            )
        else:
            metadata_mode = "fixed"
    set_hif4_qat_metadata_mode(model, metadata_mode)
    model.requires_grad_(False)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    parity = tokenizer(
        "The capital of France is",
        return_tensors="pt",
    )
    parity = {key: value.to(args.device) for key, value in parity.items()}
    with torch.no_grad():
        qat_logits = model(**parity, use_cache=False).logits
    converted = convert_hif4_qat_to_inference(model)
    model.eval()
    with torch.no_grad():
        inference_logits = model(**parity, use_cache=False).logits
    parity_max_abs = float(
        (qat_logits.float() - inference_logits.float()).abs().max().item()
    )
    if parity_max_abs != 0.0:
        raise RuntimeError(
            "QAD-to-inference conversion parity failed: "
            f"max_abs={parity_max_abs}"
        )
    fake_quant_linears = sum(
        isinstance(module, FakeQuantLinear) for module in model.modules()
    )
    # A whole-layer exemption removes that layer's seven projections from the
    # quantized set, so the count is no longer a constant.
    expected_linears = 252 - len(protected_modules)
    if fake_quant_linears != expected_linears:
        raise RuntimeError(
            f"expected {expected_linears} fake-quant linears "
            f"(252 minus {len(protected_modules)} protected projections), "
            f"got {fake_quant_linears}"
        )
    if args.skip_weight_hash:
        weight_sha256 = None
        hashed_linears = 0
    else:
        weight_sha256, hashed_linears = quantized_weight_sha256(model)
        if hashed_linears != fake_quant_linears:
            raise RuntimeError("quantized weight hash did not cover every linear")
    qk_smooth_metadata = None
    if checkpoint_has_trainable_qk_smooth:
        qk_smooth_metadata = {
            "source": "trained_log_delta",
            "base_artifact": str(trained_base_scales),
        }
    elif args.qk_smooth_scales is not None:
        if not args.qk_matmul_quant:
            raise ValueError(
                "--qk-smooth-scales only affects quantized Q/K; it is an exact "
                "equivalence transform otherwise"
            )
        qk_smooth_metadata = apply_qwen3_qk_smooth_scales(
            model, args.qk_smooth_scales
        )
    qk_exempt = _layer_list(args.qk_exempt_layers, "--qk-exempt-layers")
    pv_exempt = _layer_list(args.pv_exempt_layers, "--pv-exempt-layers")
    qk_mxfp8 = _layer_list(args.qk_mxfp8_layers, "--qk-mxfp8-layers")
    pv_mxfp8 = _layer_list(args.pv_mxfp8_layers, "--pv-mxfp8-layers")
    configure_qwen3_attention_runtime(
        model,
        kernel=args.attention_kernel,
        input_quant=False,
        output_quant=True,
        qk_matmul_quant=args.qk_matmul_quant,
        pv_matmul_quant=args.pv_matmul_quant,
        post_rope_qk_rotation=args.post_rope_qk_rotation,
        pv_normalizer_mode=args.pv_normalizer_mode,
        qk_exempt_layers=qk_exempt,
        pv_exempt_layers=pv_exempt,
        qk_mxfp8_layers=qk_mxfp8,
        pv_mxfp8_layers=pv_mxfp8,
        exempt_layers=whole_exempt,
        exempt_modules=protected_modules,
    )

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        enable_thinking=False,
    )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    metadata = {
        "qad": {
            "checkpoint": str(args.checkpoint),
            "rotation": args.rotation,
            "metadata_mode": metadata_mode,
            "qat_linears_loaded": loaded.qat.converted,
            "inference_linears": converted.converted,
            "conversion_parity_max_abs": parity_max_abs,
            "weight_bits": 4,
            "activation_bits": 4,
            "quant_format": "hif4",
            "attention_input_quant": False,
            "attention_output_quant": True,
            "qk_matmul_quant": args.qk_matmul_quant,
            "pv_matmul_quant": args.pv_matmul_quant,
            "qk_exempt_layers": list(qk_exempt),
            "pv_exempt_layers": list(pv_exempt),
            "qk_mxfp8_layers": list(qk_mxfp8),
            "pv_mxfp8_layers": list(pv_mxfp8),
            "quant_exempt_layers": list(whole_exempt),
            "quant_exempt_modules": sorted(
                f"{index}.{name}" for index, name in protected_modules
            ),
            "post_rope_qk_rotation": args.post_rope_qk_rotation,
            "pv_normalizer_mode": args.pv_normalizer_mode,
            "qk_smooth_scales": (
                str(args.qk_smooth_scales)
                if args.qk_smooth_scales is not None
                else None
            ),
            "qk_smooth_metadata": qk_smooth_metadata,
            "checkpoint_has_trainable_qk_smooth": (
                checkpoint_has_trainable_qk_smooth
            ),
            "attention_kernel": args.attention_kernel,
            "query_chunk_size": args.query_chunk_size,
            "key_chunk_size": args.key_chunk_size,
            "quantized_weight_sha256": weight_sha256,
            "quantized_weight_hash_linears": hashed_linears,
            "quantized_weight_hash_skipped": args.skip_weight_hash,
            "rollback_checkpoint": (
                str(args.rollback_checkpoint)
                if args.rollback_checkpoint is not None
                else None
            ),
            "rollback_projections": list(rollback_projections),
            "rollback_layers": sorted(rollback_layer_set),
            "rollback_weights": rollback_weights,
            "blend_checkpoint": (
                None
                if args.blend_checkpoint is None
                else str(args.blend_checkpoint)
            ),
            "blend_alpha": args.blend_alpha,
            "blend_projections": list(blend_projections),
            "blended_weights": blended_weights,
        }
    }
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        metadata=metadata,
        apply_chat_template=args.apply_chat_template,
    )
    metadata["qad"]["simulated_attention_stats"] = (
        collect_qwen3_simulated_attention_stats(model)
    )
    results["faquant_qad_runtime"] = metadata["qad"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
