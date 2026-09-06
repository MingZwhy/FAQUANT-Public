#!/usr/bin/env python
"""Prepare and save a reusable full-GPTQ HiF4 QAD step-0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faquant.gptq import (
    CALIBRATION_CORPORA,
    calibration_tokens,
    calibration_tokens_from_jsonl,
)
from faquant.qad_data import AssistantOnlyDataCollator, QADJsonlDataset
from faquant.qad import qwen3_hif4_qad_config
from faquant.qad_quantization import (
    collect_hif4_qat_master_grid_stats,
    enable_hif4_qat,
    materialize_training_tensors,
)
from faquant.qwen3 import DEFAULT_MODEL, prepare_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rotation",
        choices=("hadamard", "none", "hisq1024"),
        default="hadamard",
    )
    parser.add_argument(
        "--weight-quant",
        choices=("gptq", "rtn"),
        default="gptq",
        help=(
            "rtn direct-casts the weights and skips the second-order fit. With "
            "--rotation none and no --value-head-rotation it produces the "
            "starting point the reference QAD recipe distils from: a student "
            "with no PTQ applied at all."
        ),
    )
    parser.add_argument(
        "--value-head-rotation",
        action="store_true",
        help=(
            "Fold a per-head Hadamard through v_proj and o_proj before HiSQ and "
            "GPTQ. Free at runtime; conditions V and the attention output, which "
            "the HiSQ input rotation never touches."
        ),
    )
    parser.add_argument(
        "--hisq-down-proj-block-size",
        type=int,
        default=None,
        help=(
            "Widen the HiSQ block for down_proj only. Its 12288-wide input gets "
            "a twelfth mixed at a time under the default 1024, leaving a 71x "
            "Hessian-diagonal spread; 4096 takes that to 2.7x."
        ),
    )
    parser.add_argument(
        "--validation-data",
        default="data/qad/qwen3_8b_mmlu_longbench_validation.jsonl",
    )
    parser.add_argument("--gptq-nsamples", type=int, default=128)
    parser.add_argument(
        "--gptq-calibration-corpus",
        default="wikitext2",
        choices=CALIBRATION_CORPORA,
        help=(
            "Corpus the GPTQ calibration windows are drawn from. Every choice "
            "goes through the same join/tokenize/window path, so the "
            "distribution is the only thing that changes; RedPajama in "
            "particular is read as raw document text rather than through the "
            "chat template the QAD loader applies."
        ),
    )
    parser.add_argument("--gptq-seqlen", type=int, default=2048)
    parser.add_argument(
        "--gptq-calibration-data",
        type=Path,
        help=(
            "Optional chat JSONL calibration corpus. When omitted, use the "
            "historical WikiText-2 calibration."
        ),
    )
    parser.add_argument("--gptq-max-record-tokens", type=int, default=512)
    parser.add_argument(
        "--gptq-wikitext-fraction",
        type=float,
        default=0.0,
        help=(
            "When JSONL calibration is selected, reserve this fraction of "
            "calibration rows for the historical WikiText-2 corpus."
        ),
    )
    parser.add_argument(
        "--quant-exempt-layers",
        default="",
        help=(
            "Comma-separated decoder indices left entirely in BF16: no weight "
            "or activation quantization, no HiSQ or value-head rotation, and "
            "no attention-core quantization. It has to be set here rather "
            "than at evaluation time because GPTQ feeds each layer the "
            "quantized output of the one before it."
        ),
    )
    parser.add_argument(
        "--qk-matmul-exempt-layers",
        default="",
        help=(
            "Comma-separated decoder indices whose QK matmul stays in high "
            "precision during GPTQ calibration. Projection weights remain "
            "quantized; downstream Hessians are rebuilt around the exemption."
        ),
    )
    parser.add_argument(
        "--pv-matmul-exempt-layers",
        default="",
        help=(
            "Same as --qk-matmul-exempt-layers for the PV matmul."
        ),
    )
    parser.add_argument(
        "--qk-matmul-mxfp8-layers",
        default="",
        help="Comma-separated QK layers protected with MXFP8 E4M3.",
    )
    parser.add_argument(
        "--pv-matmul-mxfp8-layers",
        default="",
        help="Comma-separated PV layers protected with MXFP8 E4M3.",
    )
    parser.add_argument(
        "--gptq-act-order-within-group",
        action="store_true",
        help=(
            "Quantize columns in descending Hessian-diagonal order inside each "
            "64-value HiF4 group. Group membership is unchanged, so the "
            "deployed arithmetic is unaffected; only the order the solver works "
            "through a group changes, letting it spend the remaining free "
            "columns compensating the influential ones."
        ),
    )
    parser.add_argument(
        "--gptq-sequential-groups",
        action="store_true",
        help=(
            "Build each projection group's Hessian only after the earlier "
            "groups in the same layer are quantized. Layers are already "
            "sequential with respect to each other; without this the four "
            "groups inside a layer are not, so gate/up never see a quantized "
            "o_proj and down_proj never sees quantized gate/up. Costs one "
            "calibration forward pass per group instead of one per layer."
        ),
    )
    parser.add_argument(
        "--gptq-hif4-scale-search",
        default="",
        help=(
            "Comma-separated candidate offsets, in whole E6M2 grid steps, for "
            "each HiF4 block scale, e.g. '0,-1,1'. Each output row picks the "
            "one that minimises GPTQ's own weighted error. Include 0 so the "
            "reference scale stays reachable, and both signs: HiF4 rounds the "
            "scale to nearest, so about half the blocks already clip and want "
            "a step up. Empty keeps the reference rule every existing "
            "checkpoint was built with."
        ),
    )
    parser.add_argument(
        "--calibration-deployed-attention",
        action="store_true",
        help=(
            "Run GPTQ calibration through the deployed attention core: "
            "simulated kernel, QK/PV quantized, post-RoPE rotation, "
            "P-Reordering. Without this the Hessians are built from an exact "
            "attention the deployed model never runs, which misstates the "
            "input to o_proj and to every later layer. Calibration is about "
            "three times slower. A checkpoint built this way is only valid "
            "when evaluated with the same attention core."
        ),
    )
    parser.add_argument(
        "--calibration-qk-smooth-scales",
        default=None,
        help=(
            "Smooth-QK artifact to apply during calibration. Chicken-and-egg: "
            "the scales are calibrated on a checkpoint, so bootstrap from the "
            "existing one and recalibrate afterwards. Requires "
            "--calibration-deployed-attention."
        ),
    )
    parser.add_argument(
        "--gptq-damp",
        type=float,
        default=0.01,
        help=(
            "Ridge term added to the Hessian diagonal as damp*mean(diag). The "
            "0.01 default comes from weight-only GPTQ, where the Hessian is "
            "estimated from clean activations. Under W4A4 the calibration "
            "activations carry their own quantization noise, so the estimate "
            "generalises worse and wants more regularisation; a sibling MXFP4 "
            "project found its optimum an order of magnitude higher. Untested "
            "here, so it is a sweep rather than a new default."
        ),
    )
    parser.add_argument(
        "--gptq-damp-overrides",
        default="",
        help=(
            "Optional comma-separated projection=damp overrides, for example "
            "'o_proj=0.03,down_proj=0.03'. Unlisted projections retain "
            "--gptq-damp; empty preserves the historical global setting."
        ),
    )
    parser.add_argument(
        "--gptq-mxfp8-hessian-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of Hessian-collection samples using MXFP8 on selected "
            "QK/PV layers. Final propagation remains fully MXFP8; values below "
            "one shrink the Hessian toward the HiF4 calibration path."
        ),
    )
    parser.add_argument(
        "--gptq-mxfp8-hessian-mode",
        choices=("paired", "prefix_split"),
        default="paired",
        help=(
            "How to mix HiF4 and MXFP8 Hessians. 'paired' evaluates both "
            "formats on every sample; 'prefix_split' assigns the first fraction to "
            "MXFP8 and the remainder to HiF4."
        ),
    )
    parser.add_argument(
        "--gptq-calibration-seed",
        type=int,
        default=0,
        help=(
            "Seed for the WikiText window draw GPTQ calibrates on. Default 0 "
            "reproduces every existing checkpoint. Changing it is how to get an "
            "independent draw of the same recipe: --seed only feeds the global "
            "rotation, which the hisq1024 recipe never enters, so on its own it "
            "leaves the calibration set and the resulting weights untouched."
        ),
    )
    parser.add_argument(
        "--quant-exempt-modules",
        default="",
        help=(
            "Comma-separated '<layer>.<projection>' entries left in BF16, e.g. "
            "'5.down_proj,6.down_proj'. Finer than --quant-exempt-layers and "
            "composes with it. v_proj and o_proj must be exempted together "
            "when the value-head fold is on, since the fold pairs them."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument(
        "--latent-master",
        action="store_true",
        help="Initialize QAT masters from BF16 weights projected into GPTQ cells.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.gptq_wikitext_fraction < 1.0:
        raise ValueError("gptq_wikitext_fraction must be in [0, 1)")
    if args.gptq_calibration_data is None and args.gptq_wikitext_fraction:
        raise ValueError(
            "gptq_wikitext_fraction requires gptq_calibration_data"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = QADJsonlDataset(args.validation_data)
    collator = AssistantOnlyDataCollator(
        tokenizer,
        max_length=128,
        max_answer_tokens=32,
    )
    parity_batch = {
        key: value.to(args.device)
        for key, value in collator([dataset[0]]).items()
    }

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    if args.weight_quant == "rtn":
        # RTN reads nothing but the weights, so there is no calibration set to
        # build and no per-layer Hessian to accumulate: the ~490 s GPTQ spends
        # walking 36 layers disappears entirely.
        calibration = None
        calibration_source = None
    elif args.gptq_calibration_data is None:
        calibration = calibration_tokens(
            tokenizer,
            nsamples=args.gptq_nsamples,
            seqlen=args.gptq_seqlen,
            seed=args.gptq_calibration_seed,
            corpus=args.gptq_calibration_corpus,
        )
        calibration_source = {
            "dataset": {
                "wikitext2": "Salesforce/wikitext:wikitext-2-raw-v1:train",
                "c4": "allenai/c4:en:train (streamed, shuffled)",
                "redpajama": "data/qad_redpajama/train.jsonl (raw document text)",
            }[args.gptq_calibration_corpus],
            "corpus": args.gptq_calibration_corpus,
            "seed": args.gptq_calibration_seed,
        }
    else:
        wikitext_samples = round(
            args.gptq_nsamples * args.gptq_wikitext_fraction
        )
        jsonl_samples = args.gptq_nsamples - wikitext_samples
        calibration_parts = [
            calibration_tokens_from_jsonl(
                tokenizer,
                args.gptq_calibration_data,
                nsamples=jsonl_samples,
                seqlen=args.gptq_seqlen,
                seed=args.seed,
                max_record_tokens=args.gptq_max_record_tokens,
            )
        ]
        if wikitext_samples:
            calibration_parts.append(
                calibration_tokens(
                    tokenizer,
                    nsamples=wikitext_samples,
                    seqlen=args.gptq_seqlen,
                    seed=args.gptq_calibration_seed,
                )
            )
        calibration = torch.cat(calibration_parts, dim=0)
        generator = torch.Generator().manual_seed(args.seed)
        calibration = calibration[
            torch.randperm(calibration.shape[0], generator=generator)
        ]
        calibration_source = {
            "dataset": str(args.gptq_calibration_data),
            "sha256": sha256(args.gptq_calibration_data),
            "seed": args.seed,
            "category_balanced": True,
            "max_record_tokens": args.gptq_max_record_tokens,
            "jsonl_samples": jsonl_samples,
            "wikitext_samples": wikitext_samples,
            "wikitext_fraction": args.gptq_wikitext_fraction,
        }
    exempt_layers = tuple(
        int(part) for part in args.quant_exempt_layers.split(",") if part.strip()
    )
    if len(set(exempt_layers)) != len(exempt_layers):
        raise ValueError(
            f"--quant-exempt-layers lists a layer twice: {args.quant_exempt_layers!r}"
        )
    exempt_modules = tuple(
        part.strip()
        for part in args.quant_exempt_modules.split(",")
        if part.strip()
    )
    if len(set(exempt_modules)) != len(exempt_modules):
        raise ValueError(
            f"--quant-exempt-modules lists one twice: {args.quant_exempt_modules!r}"
        )
    qk_matmul_exempt_layers = tuple(
        int(part)
        for part in args.qk_matmul_exempt_layers.split(",")
        if part.strip()
    )
    pv_matmul_exempt_layers = tuple(
        int(part)
        for part in args.pv_matmul_exempt_layers.split(",")
        if part.strip()
    )
    if len(set(qk_matmul_exempt_layers)) != len(qk_matmul_exempt_layers):
        raise ValueError("--qk-matmul-exempt-layers lists one twice")
    if len(set(pv_matmul_exempt_layers)) != len(pv_matmul_exempt_layers):
        raise ValueError("--pv-matmul-exempt-layers lists one twice")
    qk_matmul_mxfp8_layers = tuple(
        int(part)
        for part in args.qk_matmul_mxfp8_layers.split(",")
        if part.strip()
    )
    pv_matmul_mxfp8_layers = tuple(
        int(part)
        for part in args.pv_matmul_mxfp8_layers.split(",")
        if part.strip()
    )
    if len(set(qk_matmul_mxfp8_layers)) != len(qk_matmul_mxfp8_layers):
        raise ValueError("--qk-matmul-mxfp8-layers lists one twice")
    if len(set(pv_matmul_mxfp8_layers)) != len(pv_matmul_mxfp8_layers):
        raise ValueError("--pv-matmul-mxfp8-layers lists one twice")
    scale_search = tuple(
        int(part) for part in args.gptq_hif4_scale_search.split(",") if part.strip()
    )
    damp_overrides = tuple(
        (name.strip(), float(value))
        for item in args.gptq_damp_overrides.split(",")
        if item.strip()
        for name, separator, value in (item.partition("="),)
        if separator
    )
    if args.gptq_damp_overrides and len(damp_overrides) != len(
        [item for item in args.gptq_damp_overrides.split(",") if item.strip()]
    ):
        raise ValueError(
            "--gptq-damp-overrides entries must use projection=value"
        )
    config = qwen3_hif4_qad_config(
        rotation=args.rotation,
        value_head_rotation=args.value_head_rotation,
        hisq_down_proj_block_size=args.hisq_down_proj_block_size,
        gptq_nsamples=args.gptq_nsamples,
        gptq_seqlen=args.gptq_seqlen,
        seed=args.seed,
        capture_latent_master=args.latent_master,
        weight_quant=args.weight_quant,
        quant_exempt_layers=exempt_layers,
        quant_exempt_modules=exempt_modules,
        qk_matmul_exempt_layers=qk_matmul_exempt_layers,
        pv_matmul_exempt_layers=pv_matmul_exempt_layers,
        qk_matmul_mxfp8_layers=qk_matmul_mxfp8_layers,
        pv_matmul_mxfp8_layers=pv_matmul_mxfp8_layers,
        gptq_damp=args.gptq_damp,
        gptq_mxfp8_hessian_fraction=args.gptq_mxfp8_hessian_fraction,
        gptq_mxfp8_hessian_mode=args.gptq_mxfp8_hessian_mode,
        gptq_damp_overrides=damp_overrides,
        hif4_scale_search_steps=scale_search,
        sequential_groups=args.gptq_sequential_groups,
        act_order_within_group=args.gptq_act_order_within_group,
        deployed_attention=args.calibration_deployed_attention,
        qk_smooth_scales=args.calibration_qk_smooth_scales,
    )
    prepare_model(model, config, calibration_input_ids=calibration)
    with torch.no_grad():
        inference_logits = model(
            input_ids=parity_batch["input_ids"],
            attention_mask=parity_batch["attention_mask"],
            use_cache=False,
        ).logits
    with torch.inference_mode(False):
        qat = enable_hif4_qat(model)
    model.eval()
    with torch.no_grad():
        qat_logits = model(
            input_ids=parity_batch["input_ids"],
            attention_mask=parity_batch["attention_mask"],
            use_cache=False,
        ).logits
    max_abs = float(
        (inference_logits.float() - qat_logits.float()).abs().max().item()
    )
    if max_abs != 0.0:
        raise RuntimeError(f"QAD initialization parity failed: max_abs={max_abs}")
    tensor_stats = materialize_training_tensors(model)
    master_grid_stats = collect_hif4_qat_master_grid_stats(model)
    # Which scale each block ended up on. Without this the only readout is
    # whether the shards hash differently from the reference build, which says
    # nothing about how often the search moved or in which direction.
    scale_step_histogram: dict[str, int] = {}
    for layer_stats in getattr(model, "faquant_gptq_stats", {}).values():
        for step, count in getattr(layer_stats, "scale_step_counts", {}).items():
            key = str(step)
            scale_step_histogram[key] = scale_step_histogram.get(key, 0) + count
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)

    shards = sorted(args.output_dir.glob("model-*.safetensors"))
    if not shards:
        shards = sorted(args.output_dir.glob("model.safetensors"))
    receipt = {
        "model": args.model,
        "rotation": args.rotation,
        "value_head_rotation": args.value_head_rotation,
        # Read back by load_qad_student_checkpoint: the saved permutation and
        # signs look identical for every block size, so this is the only record
        # of which Hadamard the runtime has to apply.
        "hisq_down_proj_block_size": args.hisq_down_proj_block_size,
        "quant_exempt_layers": list(exempt_layers),
        "quant_exempt_modules": list(exempt_modules),
        "qk_matmul_exempt_layers": list(qk_matmul_exempt_layers),
        "pv_matmul_exempt_layers": list(pv_matmul_exempt_layers),
        "qk_matmul_mxfp8_layers": list(qk_matmul_mxfp8_layers),
        "pv_matmul_mxfp8_layers": list(pv_matmul_mxfp8_layers),
        "gptq_damp": args.gptq_damp,
        "gptq_mxfp8_hessian_fraction": args.gptq_mxfp8_hessian_fraction,
        "gptq_mxfp8_hessian_mode": args.gptq_mxfp8_hessian_mode,
        "gptq_damp_overrides": {
            name: value for name, value in damp_overrides
        },
        "gptq_hif4_scale_search": list(scale_search),
        "gptq_sequential_groups": args.gptq_sequential_groups,
        "gptq_hif4_scale_step_histogram": scale_step_histogram,
        "gptq_act_order_within_group": args.gptq_act_order_within_group,
        "calibration_deployed_attention": args.calibration_deployed_attention,
        "calibration_qk_smooth_scales": args.calibration_qk_smooth_scales,
        "seed": args.seed,
        "latent_master": args.latent_master,
        "weight_quant": args.weight_quant,
        # None rather than an empty object, so a reader cannot mistake an RTN
        # init for a GPTQ one whose calibration record went missing.
        "gptq": None
        if calibration_source is None
        else {
            **calibration_source,
            "nsamples": args.gptq_nsamples,
            "seqlen": args.gptq_seqlen,
            # From the config, not a literal: this said 0.01 through an entire
            # damping sweep, so every receipt in it recorded the wrong value.
            "damp": config.gptq_damp,
            "mxfp8_hessian_fraction": config.gptq_mxfp8_hessian_fraction,
            "mxfp8_hessian_mode": config.gptq_mxfp8_hessian_mode,
            "damp_overrides": {
                name: value for name, value in config.gptq_damp_overrides
            },
            "hif4_scale_search_steps": list(config.gptq_hif4_scale_search_steps),
            "act_order_within_group": config.gptq_act_order_within_group,
            "sequential_groups": config.gptq_sequential_groups,
        },
        # Read from the config rather than restated as literals. These were
        # hardcoded, so a checkpoint calibrated through the deployed attention
        # core still reported qk/pv as unquantized -- provenance that says the
        # opposite of what was run is worse than none.
        "quantization": {
            "format": config.quant_format,
            "weight_bits": config.bits,
            "activation_bits": config.bits,
            "group_size": config.weight_group_size,
            "attention_kernel": config.attention_kernel,
            "attention_input_quant": config.attention_input_quant,
            "attention_output_quant": config.attention_output_quant,
            "qk_matmul_quant": config.qk_matmul_quant,
            "pv_matmul_quant": config.pv_matmul_quant,
            "qk_matmul_exempt_layers": list(qk_matmul_exempt_layers),
            "pv_matmul_exempt_layers": list(pv_matmul_exempt_layers),
            "qk_matmul_mxfp8_layers": list(qk_matmul_mxfp8_layers),
            "pv_matmul_mxfp8_layers": list(pv_matmul_mxfp8_layers),
            "post_rope_qk_rotation": config.post_rope_qk_rotation,
            "pv_normalizer_mode": config.pv_normalizer_mode,
            "qat_linears": qat.converted,
        },
        "step0_parity_max_abs": max_abs,
        "training_tensors": asdict(tensor_stats),
        "master_grid": asdict(master_grid_stats),
        "elapsed_seconds": time.time() - started,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "shards": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in shards
        },
    }
    (args.output_dir / "qad_init.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
