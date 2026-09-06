#!/usr/bin/env python
"""Evaluate QAD with quantized prefill and BF16 autoregressive decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from importlib.resources import files
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from faquant.config import ExperimentConfig
from faquant.hisq_rotation import apply_qwen_hisq_input_rotation
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
    install_qwen3_attention_adapter,
)
from faquant.rotation import (
    apply_qwen3_value_head_rotation,
    chunked_block_hadamard_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--decode-model",
        help="Native BF16 model used after the quantized prefill; defaults to --model.",
    )
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
    parser.add_argument(
        "--sample-start",
        type=int,
        help="Inclusive sample index for a LongBench task-shard.",
    )
    parser.add_argument(
        "--sample-end",
        type=int,
        help="Exclusive sample index for a LongBench task-shard.",
    )
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
        "--overlay-quant-exempt-layers",
        default="",
        help=(
            "Post-hoc whole-layer protection: after loading a QAD checkpoint "
            "that was trained without exemptions, restore those decoder "
            "layers' seven projections to the original BF16 weights and run "
            "them as plain Linear modules. Attention QK/PV quant, post-RoPE "
            "rotation and Smooth-QK are also dropped on those layers. This is "
            "the zero-train swap used for QAD-then-protect; it is not a GPTQ "
            "rebuild."
        ),
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


def _projection_parent(layer, name):
    from faquant.config import QWEN_ATTENTION_PROJECTIONS

    return layer.self_attn if name in QWEN_ATTENTION_PROJECTIONS else layer.mlp


def _capture_original_layer_weights(model, layers: tuple[int, ...]) -> dict:
    from faquant.config import QWEN_PROJECTIONS

    captured: dict = {}
    for index in layers:
        layer = model.model.layers[index]
        for name in QWEN_PROJECTIONS:
            module = getattr(_projection_parent(layer, name), name)
            captured[(index, name)] = {
                "weight": module.weight.detach().cpu().clone(),
                "bias": (
                    None
                    if module.bias is None
                    else module.bias.detach().cpu().clone()
                ),
            }
    return captured


def _apply_overlay_layer_protection(
    model, layers: tuple[int, ...], original_weights: dict
) -> int:
    from faquant.config import QWEN_PROJECTIONS

    replaced = 0
    for index in layers:
        layer = model.model.layers[index]
        for name in QWEN_PROJECTIONS:
            parent = _projection_parent(layer, name)
            old = getattr(parent, name)
            payload = original_weights[(index, name)]
            device = old.weight.device
            dtype = old.weight.dtype
            new = torch.nn.Linear(
                old.in_features,
                old.out_features,
                bias=payload["bias"] is not None,
                device=device,
                dtype=dtype,
            )
            new.weight.data.copy_(payload["weight"].to(device=device, dtype=dtype))
            if payload["bias"] is not None:
                new.bias.data.copy_(
                    payload["bias"].to(device=device, dtype=dtype)
                )
            setattr(parent, name, new)
            replaced += 1
    return replaced


class _BF16HiSQLinear(torch.nn.Module):
    """Apply the saved HiSQ basis without activation or weight quantization."""

    faquant_quantizes_input = False

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.register_buffer(
            "rotation_signs",
            linear.faquant_input_rotation_signs.detach().clone(),
        )
        self.register_buffer(
            "rotation_permutation",
            linear.faquant_input_rotation_permutation.detach().clone(),
        )
        self.rotation_block_size = linear.faquant_input_rotation_block_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape
        flat = inputs.reshape(-1, shape[-1])
        rotated = torch.empty_like(flat)
        signs = self.rotation_signs.to(
            device=flat.device, dtype=torch.float32
        )
        permutation = self.rotation_permutation.to(device=flat.device)
        for start in range(0, flat.shape[0], 256):
            stop = min(start + 256, flat.shape[0])
            work = flat[start:stop].float() * signs
            work = work.index_select(-1, permutation)
            work = chunked_block_hadamard_transform(
                work,
                block_size=self.rotation_block_size,
                chunk_rows=256,
            )
            rotated[start:stop].copy_(work.to(flat.dtype))
        return F.linear(rotated.reshape(shape), self.weight, self.bias)


def _install_bf16_hisq_linears(
    model: torch.nn.Module,
    protected_modules: frozenset[tuple[int, str]],
) -> None:
    for index, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if (index, name) not in protected_modules:
                setattr(
                    layer.self_attn,
                    name,
                    _BF16HiSQLinear(getattr(layer.self_attn, name)),
                )
        for name in ("gate_proj", "up_proj", "down_proj"):
            if (index, name) not in protected_modules:
                setattr(
                    layer.mlp,
                    name,
                    _BF16HiSQLinear(getattr(layer.mlp, name)),
                )


def _prepare_bf16_decode_model(
    *,
    model_path: str,
    device: str,
    smooth_scales: Path,
    exempt_layers: tuple[int, ...],
    protected_modules: frozenset[tuple[int, str]],
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    apply_qwen3_value_head_rotation(
        model,
        exempt_modules=protected_modules,
    )
    apply_qwen_hisq_input_rotation(
        model,
        block_size=1024,
        seed=17,
        exempt_modules=protected_modules,
    )
    _install_bf16_hisq_linears(model, protected_modules)
    adapter_config = ExperimentConfig(
        attention_kernel="native",
        attention_input_quant=False,
        attention_output_quant=False,
        post_rope_qk_rotation=True,
        quant_target="none",
        qwen_quant_exempt_layers=exempt_layers,
    )
    install_qwen3_attention_adapter(model, adapter_config)
    apply_qwen3_qk_smooth_scales(model, smooth_scales)
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        attention.faquant_attention_kernel = "native"
        attention.faquant_attention_input_quant = False
        attention.faquant_attention_output_quant = False
        attention.faquant_qk_matmul_quant = False
        attention.faquant_pv_matmul_quant = False
        if index in exempt_layers:
            attention.faquant_qk_rotation = False
            attention.faquant_qk_smooth_scale = None
            attention.faquant_qk_key_offset = None
        else:
            attention.faquant_qk_rotation = True
        attention.faquant_qk_rotation_block_size = None
        attention.faquant_k_matmul_qdq_cache = None
        attention.faquant_v_matmul_qdq_cache = None
    model.requires_grad_(False)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return model


def _stopped(stopping_criteria, sequence: torch.Tensor, scores: torch.Tensor) -> bool:
    if stopping_criteria is None:
        return False
    stopped = stopping_criteria(sequence, scores)
    if isinstance(stopped, torch.Tensor):
        return bool(stopped.all().item())
    return bool(stopped)


@torch.inference_mode()
def _prefill_quant_decode_bf16_generate(
    self,
    *,
    input_ids: torch.Tensor,
    max_length: int,
    stopping_criteria=None,
    **generation_kwargs,
) -> torch.Tensor:
    if input_ids.shape[0] != 1:
        raise ValueError("prefill-only quantization requires batch size 1")
    if generation_kwargs.get("do_sample", False):
        raise ValueError("prefill-only quantization currently supports greedy decode")
    if int(generation_kwargs.get("num_beams", 1)) != 1:
        raise ValueError("prefill-only quantization currently supports one beam")

    decode_model = self.__dict__["_faquant_bf16_decode_model"]
    outputs = self(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = outputs.past_key_values
    scores = outputs.logits[:, -1, :]
    next_token = scores.argmax(dim=-1, keepdim=True)
    sequence = torch.cat((input_ids, next_token), dim=-1)

    eos = generation_kwargs.get(
        "eos_token_id", self.generation_config.eos_token_id
    )
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(item) for item in eos}

    while sequence.shape[-1] < max_length:
        if int(next_token.item()) in eos_ids:
            break
        if _stopped(stopping_criteria, sequence, scores):
            break
        cache_length = cache.get_seq_length()
        cache_position = torch.tensor(
            [cache_length],
            device=input_ids.device,
            dtype=torch.long,
        )
        outputs = decode_model(
            input_ids=next_token,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        scores = outputs.logits[:, -1, :]
        next_token = scores.argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=-1)
    return sequence


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
    # Whole-layer exemptions are normally a property of the checkpoint: those
    # layers were built with plain Linear modules. Read them back so both the
    # module-count check and the attention core agree with how the checkpoint
    # was actually built. --overlay-quant-exempt-layers is the post-hoc swap
    # for a checkpoint trained without protection: original BF16 weights go
    # back into those layers after the QAD state is loaded.
    overlay_layers = _layer_list(
        args.overlay_quant_exempt_layers, "--overlay-quant-exempt-layers"
    )
    original_overlay_weights = (
        _capture_original_layer_weights(model, overlay_layers)
        if overlay_layers
        else {}
    )
    recorded_exempt = recorded_quant_exempt_layers(args.checkpoint)
    recorded_modules = recorded_quant_exempt_modules(args.checkpoint)
    overlap = sorted(set(overlay_layers) & set(recorded_exempt))
    if overlap:
        raise ValueError(
            "overlay layers already recorded as protected in the checkpoint: "
            f"{overlap}"
        )
    whole_exempt = recorded_exempt
    protected_modules = _expand_protected_modules(whole_exempt, recorded_modules)
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
    overlay_replaced = 0
    if overlay_layers:
        overlay_replaced = _apply_overlay_layer_protection(
            model, overlay_layers, original_overlay_weights
        )
        expected_overlay = 7 * len(overlay_layers)
        if overlay_replaced != expected_overlay:
            raise RuntimeError(
                "overlay did not replace every protected projection: "
                f"got {overlay_replaced}, expected {expected_overlay}"
            )
        whole_exempt = tuple(sorted(set(recorded_exempt) | set(overlay_layers)))
        protected_modules = _expand_protected_modules(
            whole_exempt, recorded_modules
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
    mxfp8_overlap = sorted(set(overlay_layers) & (set(qk_mxfp8) | set(pv_mxfp8)))
    if mxfp8_overlap:
        raise ValueError(
            "overlay protection overlaps MXFP8 layers: "
            f"{mxfp8_overlap}"
        )
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
    if args.qk_smooth_scales is None:
        raise ValueError("prefill-only decode requires --qk-smooth-scales")
    torch.cuda.empty_cache()
    decode_model = _prepare_bf16_decode_model(
        model_path=args.decode_model or args.model,
        device=args.device,
        smooth_scales=args.qk_smooth_scales,
        exempt_layers=whole_exempt,
        protected_modules=protected_modules,
    )
    if any(isinstance(module, FakeQuantLinear) for module in decode_model.modules()):
        raise RuntimeError("BF16 decode model unexpectedly contains FakeQuantLinear")
    model.__dict__["_faquant_bf16_decode_model"] = decode_model
    model.generate = MethodType(_prefill_quant_decode_bf16_generate, model)

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM as BaseHFLM
    from lm_eval.tasks import TaskManager

    class HFLM(BaseHFLM):
        """Add official LongBench middle truncation without patching lm-eval."""

        def tok_batch_encode(
            self,
            strings: list[str],
            padding_side: str = "left",
            left_truncate_len: int | None = None,
            truncation: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            mode = os.environ.get("LM_EVAL_TRUNCATION_MODE", "left").lower()
            if mode != "middle" or left_truncate_len is None:
                return super().tok_batch_encode(
                    strings,
                    padding_side=padding_side,
                    left_truncate_len=left_truncate_len,
                    truncation=truncation,
                )
            input_ids, attention_mask = super().tok_batch_encode(
                strings,
                padding_side=padding_side,
                left_truncate_len=None,
                truncation=truncation,
            )
            if input_ids.shape[-1] <= left_truncate_len:
                return input_ids, attention_mask
            if input_ids.shape[0] != 1:
                raise ValueError("middle truncation requires batch size 1")
            prefix_len = left_truncate_len // 2
            suffix_len = left_truncate_len - prefix_len
            input_ids = torch.cat(
                (input_ids[:, :prefix_len], input_ids[:, -suffix_len:]),
                dim=-1,
            )
            attention_mask = torch.cat(
                (
                    attention_mask[:, :prefix_len],
                    attention_mask[:, -suffix_len:],
                ),
                dim=-1,
            )
            return input_ids, attention_mask

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
    task_manager = TaskManager(
        include_path=str(files("faquant.tasks.longbench"))
    )
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
            "overlay_quant_exempt_layers": list(overlay_layers),
            "overlay_replaced_projections": overlay_replaced,
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
            "quantization_scope": "prefill_only",
            "first_generated_token_source": "quantized_prefill_logits",
            "decode_precision": "bfloat16",
            "decode_model": args.decode_model or args.model,
            "decode_reuses_quantized_prefill_kv_cache": True,
            "truncation_mode": os.environ.get(
                "LM_EVAL_TRUNCATION_MODE", "left"
            ).lower(),
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
    eval_kwargs: dict[str, object] = {}
    if args.sample_start is not None or args.sample_end is not None:
        if args.sample_start is None or args.sample_end is None:
            raise ValueError("--sample-start and --sample-end must be set together")
        if len(tasks) != 1:
            raise ValueError("sample range requires exactly one --tasks entry")
        if args.sample_start < 0 or args.sample_end <= args.sample_start:
            raise ValueError("invalid --sample-start/--sample-end range")
        eval_kwargs["samples"] = {
            tasks[0]: list(range(args.sample_start, args.sample_end))
        }
        metadata["qad"]["sample_start"] = args.sample_start
        metadata["qad"]["sample_end"] = args.sample_end
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
        task_manager=task_manager,
        **eval_kwargs,
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
