"""Materialize post-QAD whole-layer BF16 protection into a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faquant.config import QWEN_PROJECTIONS
from faquant.qad_checkpoint import (
    load_qad_student_checkpoint,
    recorded_quant_exempt_layers,
)
from faquant.qad_quantization import HiF4QATLinear
from faquant.qwen3 import DEFAULT_MODEL


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in value.split(",") if item.strip())
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be unique comma-separated integers")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, default=(16, 17, 18))
    parser.add_argument("--rotation", default="hisq1024")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def projection_parent(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    return layer.self_attn if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp


def capture_original_layers(
    model: torch.nn.Module, layers: tuple[int, ...]
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor | None]]:
    captured = {}
    for index in layers:
        layer = model.model.layers[index]
        for name in QWEN_PROJECTIONS:
            module = getattr(projection_parent(layer, name), name)
            captured[(index, name)] = (
                module.weight.detach().cpu().clone(),
                None if module.bias is None else module.bias.detach().cpu().clone(),
            )
    return captured


def restore_original_layers(
    model: torch.nn.Module,
    captured: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor | None]],
) -> int:
    restored = 0
    for (index, name), (weight, bias) in captured.items():
        parent = projection_parent(model.model.layers[index], name)
        old = getattr(parent, name)
        replacement = torch.nn.Linear(
            old.in_features,
            old.out_features,
            bias=bias is not None,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        replacement.weight.data.copy_(weight.to(old.weight.device, old.weight.dtype))
        if bias is not None:
            replacement.bias.data.copy_(bias.to(old.weight.device, old.weight.dtype))
        setattr(parent, name, replacement)
        restored += 1
    return restored


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if recorded_quant_exempt_layers(args.checkpoint):
        raise ValueError("source checkpoint already contains whole-layer protection")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    original = capture_original_layers(model, args.layers)
    loaded = load_qad_student_checkpoint(
        model,
        args.checkpoint,
        rotation=args.rotation,
        seed=0,
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: {loaded.missing_keys=} {loaded.unexpected_keys=}"
        )
    restored = restore_original_layers(model, original)
    if restored != len(args.layers) * len(QWEN_PROJECTIONS):
        raise RuntimeError(f"restored {restored} projections")
    qat_linears = sum(isinstance(module, HiF4QATLinear) for module in model.modules())
    expected_linears = 252 - restored
    if qat_linears != expected_linears:
        raise RuntimeError(f"expected {expected_linears} QAT linears, got {qat_linears}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)

    source_receipt = args.checkpoint / "qad_init.json"
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    receipt["quant_exempt_layers"] = list(args.layers)
    receipt["quant_exempt_modules"] = []
    receipt["overlay_replaced_projections"] = restored
    receipt["source_qad_checkpoint"] = args.checkpoint.name
    if isinstance(receipt.get("quantization"), dict):
        receipt["quantization"]["qat_linears"] = qat_linears
    shards = {}
    for path in sorted(args.output_dir.glob("*.safetensors")):
        shards[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    receipt["shards"] = shards
    (args.output_dir / "qad_init.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "protected_layers": args.layers,
                "restored_projections": restored,
                "qat_linears": qat_linears,
                "shards": shards,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
