#!/usr/bin/env python3
"""Diagnose post-RoPE Q/K channel imbalance and calibrate Smooth-QK scales.

The enhanced recipe's remaining attention-core loss is dominated by QK, not PV:
on the frozen step-0 checkpoint the MMLU split is QK-only -0.7406 pp, PV-only
-0.3133 pp, both -1.9727 pp, so the QK/PV interaction alone is +0.9188 pp -- more
than either operand on its own. That makes the Q/K operands the highest-leverage
target, and their known weakness is a channel imbalance: Qwen3's k_norm affine
weight is reported to carry one channel near 34, which post-RoPE turns into a
head-invariant K outlier that inflates every HiF4 group it lands in.

Smooth-QK migrates that imbalance into Q, where there is headroom, using a
diagonal scale that leaves the scores untouched:

    S = (Q diag(s)) (K diag(s)^-1)^T = Q K^T

applied after RoPE and before the Hadamard, so it composes exactly with the
existing rotation. The scale follows SmoothQuant's form on post-RoPE per-channel
maxima, s_c = |K_max,c|^alpha / |Q_max,c|^(1-alpha); alpha=0.5 equalizes the two
operands' channel maxima at their geometric mean.

This script only measures and calibrates. It writes the per-layer statistics that
decide whether Smooth-QK applies at all (the gate is a K channel outlier ratio of
roughly 6 or more together with K-over-Q dominance of 2 or more) and the scale
vectors themselves, so the evaluation path can load a frozen artifact instead of
recalibrating.

Statistics are collected with the deployed linears, i.e. Q/K carry their W4A4
error when --checkpoint is given, because those are the maxima the quantizer will
actually see.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from faquant.gptq import calibration_tokens
from faquant.qad_checkpoint import load_qad_student_checkpoint
from faquant.qad_quantization import (
    convert_hif4_qat_to_inference,
    set_hif4_qat_metadata_mode,
)
from faquant.qad_checkpoint import (
    recorded_quant_exempt_layers,
    recorded_quant_exempt_modules,
)
from faquant.quantization import FakeQuantLinear
from faquant.qwen3 import DEFAULT_MODEL

EXPECTED_QUANTIZED_LINEARS = 252
GATE_K_OUTLIER_RATIO = 6.0
GATE_K_OVER_Q_DOMINANCE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--alphas", default="0.25,0.5,0.75")
    parser.add_argument(
        "--statistic",
        choices=("max", "rms"),
        default="max",
        help=(
            "Which per-channel statistic drives the scale. 'max' is SmoothQuant's "
            "form. 'rms' minimizes the product of operand norms, which is what "
            "bounds the score error once the Hadamard has spread the channels: "
            "|dS| <~ 2*eps*||q~||*||k~||, minimized at s_c = (E[k_c^2]/E[q_c^2])^(1/4), "
            "i.e. this same expression with alpha=0.5 on RMS statistics."
        ),
    )
    parser.add_argument(
        "--scale-clamp",
        type=float,
        default=32.0,
        help="Clamp scales to [1/clamp, clamp] so near-dead channels cannot blow up.",
    )
    parser.add_argument(
        "--key-mean",
        action="store_true",
        help=(
            "Also calibrate a per-(kv head, channel) mean subtracted from post-RoPE "
            "K. Free and exact: it shifts every score in a row by the same "
            "constant, which softmax ignores, while removing the token-shared "
            "component that inflates the max of every HiF4 group K lands in. The "
            "scale is then fitted on the residual."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scales-output",
        type=Path,
        help="Where to save the calibrated scale tensors (.pt).",
    )
    parser.add_argument(
        "--scales-alpha",
        type=float,
        default=0.5,
        help="Which alpha to save in --scales-output.",
    )
    return parser.parse_args()


def _load_model(args: argparse.Namespace):
    source = args.checkpoint if args.checkpoint is not None else args.model
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    if args.checkpoint is None:
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = False
        return model, tokenizer, {"linear_quantization": "none (BF16)"}

    loaded = load_qad_student_checkpoint(
        model,
        args.checkpoint,
        rotation="hisq1024",
        seed=0,
        hisq_block_size=1024,
        hisq_seed=17,
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={loaded.missing_keys}, "
            f"unexpected={loaded.unexpected_keys}"
        )
    set_hif4_qat_metadata_mode(model, "fixed")
    model.requires_grad_(False)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    converted = convert_hif4_qat_to_inference(model)
    quantized = sum(isinstance(module, FakeQuantLinear) for module in model.modules())
    # A whole-layer exemption drops that layer's seven projections from the
    # quantized set, so the expected count is no longer a constant.
    protected = 7 * len(recorded_quant_exempt_layers(args.checkpoint)) + len(
        recorded_quant_exempt_modules(args.checkpoint)
    )
    expected = EXPECTED_QUANTIZED_LINEARS - protected
    if (
        loaded.qat.converted != expected
        or converted.converted != expected
        or quantized != expected
    ):
        raise RuntimeError(
            f"expected {expected} quantized linears (252 minus {protected} "
            f"protected projections), got "
            f"loaded={loaded.qat.converted}, converted={converted.converted}, "
            f"modules={quantized}"
        )
    if args.device.startswith("cuda:"):
        torch.cuda.empty_cache()
    # Step-0 checkpoints carry qad_init.json; the ones QAD training writes carry
    # training_receipt.json and no shard digests.  Recording provenance must not
    # decide whether the calibration can run, since the scales have to be
    # recalibrated precisely when the weights have moved.
    provenance: dict[str, object] = {
        "linear_quantization": "hif4 W4A4, GPTQ, HiSQ1024 input rotation",
        "checkpoint": str(args.checkpoint),
    }
    init_receipt = args.checkpoint / "qad_init.json"
    training_receipt = args.checkpoint / "training_receipt.json"
    if init_receipt.exists():
        receipt = json.loads(init_receipt.read_text(encoding="utf-8"))
        provenance["checkpoint_shard_sha256"] = {
            name: shard["sha256"] for name, shard in receipt["shards"].items()
        }
    elif training_receipt.exists():
        receipt = json.loads(training_receipt.read_text(encoding="utf-8"))
        provenance["training_receipt"] = {
            key: receipt.get(key)
            for key in ("student_init", "rotation", "step", "global_step")
            if receipt.get(key) is not None
        }
    return model, tokenizer, provenance


class _ChannelMaxCollector:
    """Recompute post-RoPE Q/K for one layer and accumulate per-channel maxima.

    Mirrors the adapter's own pre-Hadamard path (q_proj/q_norm, k_proj/k_norm,
    then RoPE) so the statistics are in the basis the scale acts on.
    """

    def __init__(
        self,
        attention: torch.nn.Module,
        head_dim: int,
        device: str,
        *,
        key_value_heads: int,
        key_offset: torch.Tensor | None = None,
    ) -> None:
        self._attention = attention
        self._key_offset = key_offset
        self.query_max = torch.zeros(head_dim, dtype=torch.float64, device=device)
        self.key_max = torch.zeros(head_dim, dtype=torch.float64, device=device)
        self.query_sq_sum = torch.zeros(head_dim, dtype=torch.float64, device=device)
        self.key_sq_sum = torch.zeros(head_dim, dtype=torch.float64, device=device)
        # Per KV head, because the offset only has to be constant along the key
        # axis to cancel in softmax, and a per-head mean removes strictly more.
        self.key_head_sum = torch.zeros(
            key_value_heads, head_dim, dtype=torch.float64, device=device
        )
        self.key_head_elements = 0
        self.query_elements = 0
        self.key_elements = 0

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = args[0]
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("position_embeddings must be passed as a keyword")
        attention = self._attention
        shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(
            attention.q_proj(hidden_states).view(shape)
        ).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(
            1, 2
        )
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key_by_head = key.double()
        self.key_head_sum += key_by_head.sum(dim=(0, 2))
        self.key_head_elements += key_by_head.shape[0] * key_by_head.shape[2]
        if self._key_offset is not None:
            # Second pass: report the statistics the scale will actually be fitted
            # on, i.e. after the offset has already removed the shared component.
            key = key - self._key_offset
        # Channels live on the last axis; fold everything else so the statistic is
        # head-invariant, matching the k_norm-induced outlier's own structure.
        query = query.reshape(-1, query.shape[-1]).double()
        key = key.reshape(-1, key.shape[-1]).double()
        self.query_max = torch.maximum(self.query_max, query.abs().amax(dim=0))
        self.key_max = torch.maximum(self.key_max, key.abs().amax(dim=0))
        self.query_sq_sum += query.pow(2).sum(dim=0)
        self.key_sq_sum += key.pow(2).sum(dim=0)
        self.query_elements += query.shape[0]
        self.key_elements += key.shape[0]
        return None


def _outlier_ratio(channel_max: torch.Tensor) -> float:
    median = float(channel_max.median())
    if median <= 0:
        return float("inf")
    return float(channel_max.max()) / median


def main() -> None:
    args = parse_args()
    alphas = tuple(float(item) for item in args.alphas.split(",") if item.strip())
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alphas must be in [0, 1]")
    if args.scales_alpha not in alphas:
        raise ValueError("--scales-alpha must be one of --alphas")

    model, tokenizer, provenance = _load_model(args)
    head_dim = model.model.layers[0].self_attn.head_dim
    key_value_heads = model.config.num_key_value_heads
    windows = calibration_tokens(
        tokenizer,
        nsamples=args.samples,
        seqlen=args.sequence_length,
        seed=args.seed,
    )

    def collect(
        offsets: list[torch.Tensor] | None,
    ) -> list[_ChannelMaxCollector]:
        collectors = []
        handles = []
        for index, layer in enumerate(model.model.layers):
            collector = _ChannelMaxCollector(
                layer.self_attn,
                head_dim,
                args.device,
                key_value_heads=key_value_heads,
                key_offset=None if offsets is None else offsets[index],
            )
            collectors.append(collector)
            handles.append(
                layer.self_attn.register_forward_pre_hook(collector, with_kwargs=True)
            )
        try:
            with torch.no_grad():
                for index in range(args.samples):
                    model(
                        input_ids=windows[index : index + 1].to(args.device),
                        use_cache=False,
                    )
        finally:
            for handle in handles:
                handle.remove()
        return collectors

    raw_collectors = collect(None)
    collectors = raw_collectors
    key_offsets: dict[str, torch.Tensor] = {}
    if args.key_mean:
        # The offset only cancels in softmax if it is constant along the key axis,
        # so it is a per-(layer, kv head, channel) vector, and the scale must then
        # be fitted on the residual rather than on raw K.
        offsets = [
            (collector.key_head_sum / collector.key_head_elements).float()[:, None, :]
            for collector in raw_collectors
        ]
        for index, offset in enumerate(offsets):
            key_offsets[f"model.layers.{index}.self_attn"] = (
                offset[:, 0, :].to(torch.float32).cpu()
            )
        collectors = collect(offsets)

    layers = []
    scales: dict[str, torch.Tensor] = {}
    for index, collector in enumerate(collectors):
        attention = model.model.layers[index].self_attn
        query_max = collector.query_max
        key_max = collector.key_max
        query_rms = (collector.query_sq_sum / collector.query_elements).sqrt()
        key_rms = (collector.key_sq_sum / collector.key_elements).sqrt()
        entry = {
            "layer": index,
            "k_norm_weight_max": float(attention.k_norm.weight.abs().max()),
            "k_norm_weight_median": float(attention.k_norm.weight.abs().median()),
            "q_norm_weight_max": float(attention.q_norm.weight.abs().max()),
            "q_norm_weight_median": float(attention.q_norm.weight.abs().median()),
            "key_channel_max": float(key_max.max()),
            "key_outlier_ratio": _outlier_ratio(key_max),
            "key_argmax_channel": int(key_max.argmax()),
            "query_channel_max": float(query_max.max()),
            "query_outlier_ratio": _outlier_ratio(query_max),
            "key_over_query_dominance": float(key_max.max() / query_max.max()),
            "key_rms_max": float(key_rms.max()),
            "query_rms_max": float(query_rms.max()),
            "alphas": {},
        }
        raw = raw_collectors[index]
        key_head_mean = raw.key_head_sum / raw.key_head_elements
        entry["key_head_mean_abs_max"] = float(key_head_mean.abs().max())
        entry["raw_key_channel_max"] = float(raw.key_max.max())
        # How much of K's worst channel is a constant the softmax cannot see. A
        # large fraction is what would make mean subtraction worth its keep.
        entry["key_head_mean_over_raw_max"] = float(
            key_head_mean.abs().max() / raw.key_max.max().clamp_min(1e-12)
        )
        if args.key_mean:
            entry["key_channel_max_after_mean"] = float(key_max.max())
            entry["key_outlier_ratio_after_mean"] = _outlier_ratio(key_max)
        entry["gate_passes"] = bool(
            entry["key_outlier_ratio"] >= GATE_K_OUTLIER_RATIO
            and entry["key_over_query_dominance"] >= GATE_K_OVER_Q_DOMINANCE
        )
        floor = torch.finfo(torch.float32).tiny
        key_stat = key_max if args.statistic == "max" else key_rms
        query_stat = query_max if args.statistic == "max" else query_rms
        for alpha in alphas:
            scale = key_stat.clamp_min(floor).pow(alpha) / query_stat.clamp_min(
                floor
            ).pow(1.0 - alpha)
            # A per-channel scale is only meaningful up to a global constant: the
            # scores are invariant to s -> c*s only if Q and K move oppositely,
            # which they do, so normalize to keep both operands near their
            # original overall magnitude and make the clamp meaningful.
            scale = scale / scale.median()
            scale = scale.clamp(1.0 / args.scale_clamp, args.scale_clamp)
            smoothed_key = key_max / scale
            smoothed_query = query_max * scale
            entry["alphas"][f"{alpha:g}"] = {
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "smoothed_key_channel_max": float(smoothed_key.max()),
                "smoothed_key_outlier_ratio": _outlier_ratio(smoothed_key),
                "smoothed_query_channel_max": float(smoothed_query.max()),
                "smoothed_query_outlier_ratio": _outlier_ratio(smoothed_query),
                # The product of the two operands' channel maxima is what a
                # per-group quantizer has to cover; balancing should lower the
                # worse of the two without raising this product.
                "max_operand_channel_max": float(
                    max(smoothed_key.max(), smoothed_query.max())
                ),
            }
            if alpha == args.scales_alpha:
                scales[f"model.layers.{index}.self_attn"] = scale.to(torch.float32).cpu()
        layers.append(entry)

    gated = [entry["layer"] for entry in layers if entry["gate_passes"]]
    payload = {
        "calibration": "qk_smooth_post_rope_channel_scales",
        "model": args.model,
        "device": args.device,
        "sequence_length": args.sequence_length,
        "samples": args.samples,
        "seed": args.seed,
        "statistic": args.statistic,
        "scale_clamp": args.scale_clamp,
        "calibration_source": {
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "note": "generic text, not LongBench prompts",
        },
        "provenance": provenance,
        "gate": {
            "key_outlier_ratio_min": GATE_K_OUTLIER_RATIO,
            "key_over_query_dominance_min": GATE_K_OVER_Q_DOMINANCE,
            "layers_passing": gated,
            "num_layers_passing": len(gated),
            "num_layers": len(layers),
        },
        "interpretation": {
            "statistics_basis": "post-RoPE, pre-Hadamard, per head_dim channel",
            "scale_applies_as": "Q *= s, K /= s (exact: scores unchanged)",
            "outlier_ratio": "channel max over the median channel max",
            "smoothed_ratios_use": (
                "max statistics regardless of --statistic, so the reported "
                "outlier ratios stay comparable across calibration variants"
            ),
        },
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.scales_output is not None:
        args.scales_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "alpha": args.scales_alpha,
                "statistic": args.statistic,
                "scale_clamp": args.scale_clamp,
                "key_mean": args.key_mean,
                "head_dim": head_dim,
                "basis": "post-RoPE pre-Hadamard per-channel, shared across heads",
                "calibration": {
                    "model": args.model,
                    "checkpoint": provenance.get("checkpoint"),
                    "sequence_length": args.sequence_length,
                    "samples": args.samples,
                    "seed": args.seed,
                },
                "scales": scales,
                "key_offsets": key_offsets if args.key_mean else None,
            },
            args.scales_output,
        )

    worst = max(layers, key=lambda entry: entry["key_outlier_ratio"])
    print(
        f"k_norm weight max over layers: "
        f"{max(entry['k_norm_weight_max'] for entry in layers):.4f}"
    )
    print(
        f"worst K channel outlier ratio: layer {worst['layer']}, "
        f"ratio {worst['key_outlier_ratio']:.2f}, "
        f"K/Q dominance {worst['key_over_query_dominance']:.2f}"
    )
    print(
        f"gate passes on {len(gated)}/{len(layers)} layers: {gated}"
    )
    for alpha in alphas:
        key_ratios = [entry["alphas"][f"{alpha:g}"]["smoothed_key_outlier_ratio"] for entry in layers]
        print(
            f"alpha={alpha:g}: worst smoothed K outlier ratio "
            f"{max(key_ratios):.2f} (was {worst['key_outlier_ratio']:.2f})"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
