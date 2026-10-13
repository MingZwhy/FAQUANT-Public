#!/usr/bin/env python3
"""HiF4 GPTQ weights for the block Linears of the Wan2.2 DiT experts (the Linear part of the Qwen recipe).

collect:  one calibration prompt (HiSQRot4's calibration set, disjoint from VBench) through the
          build the weights are for (HiF4 RTN Linears with HiF4 QK/PV, Q/K Hadamard and P-Reordering: run A),
          summing for one expert and a range of its blocks the Hessian X^T X of the HiF4-quantized inputs that
          the block Linears multiply (q/k/v share one), over every --stride-th token of both CFG branches of
          every step that expert runs. As in the Qwen GPTQ, the inputs are the quantized ones the deployed
          layer sees, coming through quantized upstream layers.
merge:    sums the collect files of one expert and block range into one Hessian file per block, keeping the
          last prompt's Hessian also on its own.
gptq:     GPTQ with the settings of the Qwen recipe (HiF4 groups of 64, damp 0.01, column blocks of 128, no
          act-order, no scale search) from the bf16 weights the RTN build quantizes, on all prompts. Reports
          per Linear the Hessian-weighted relative error of RTN and GPTQ, in-sample and on the held-out
          prompt (for a GPTQ fitted without it). Cross-attention k and v keep RTN:
          their input is the text context, of which a few calibration prompts span far fewer than its 5,120
          dimensions, so their Hessian is rank-deficient and GPTQ would fit those prompts.
assemble: <expert>.pt for both experts and manifest.json, for wan22_generate.py --hif4-weights.

  python scripts/calibrate_wan22_gptq.py collect --ckpt-dir CKPT --prompt 0 --expert high_noise_model \
      --blocks 0:14 --out DIR/high_noise_model_b00-14_p0.pt
  python scripts/calibrate_wan22_gptq.py merge --inputs 'DIR/high_noise_model_b00-14_p*.pt' --out HESS
  python scripts/calibrate_wan22_gptq.py gptq --ckpt-dir CKPT --hessians HESS --expert high_noise_model \
      --blocks 0:2 --out PARTS/high_noise_model_b00-02.pt
  python scripts/calibrate_wan22_gptq.py assemble --parts PARTS --out WEIGHTS
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn

# The thirty calibration prompts of the HiSQRot4 release, disjoint from VBench.
CALIBRATION_PROMPTS = Path("data/prompts/OpenS2V-5M_to_mm_calib_30.json")
TAGS = ("high_noise_model", "low_noise_model")
BLOCKS = 40
# the Linear whose input is recorded for each Hessian
HESSIAN_INPUTS = {
    "self_attn.q": "qkv", "self_attn.o": "o", "cross_attn.q": "cq", "cross_attn.o": "co",
    "ffn.0": "ffn0", "ffn.2": "ffn2",
}
# None: RTN
LINEAR_HESSIAN = {
    "self_attn.q": "qkv", "self_attn.k": "qkv", "self_attn.v": "qkv", "self_attn.o": "o",
    "cross_attn.q": "cq", "cross_attn.k": None, "cross_attn.v": None, "cross_attn.o": "co",
    "ffn.0": "ffn0", "ffn.2": "ffn2",
}
GPTQ_SETTINGS = {"quant_format": "hif4", "bits": 4, "group_size": 64, "symmetric": True, "block_size": 128}


def block_range(text: str) -> range:
    first, _, last = text.partition(":")
    return range(int(first), int(last))


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def save(obj, path: Path) -> None:
    """Written aside and renamed, so that an existing file is a complete one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(obj, partial)
    os.replace(partial, path)


class Done(Exception):
    """Raised to leave WanT2V.generate once the calibrated expert has run its last step."""


class HessianBank:
    """X^T X per (block, input), summed on the GPU while a block runs and then into fp32 host memory."""

    def __init__(self, stride: int) -> None:
        self.stride = stride
        self.host: dict[str, torch.Tensor] = {}
        self.tokens: dict[str, int] = {}
        self.device: dict[str, torch.Tensor] = {}
        self.block: int | None = None

    def collector(self, block: int, name: str):
        def collect(_module: nn.Module, x: torch.Tensor, start: int) -> None:
            if self.block != block:
                self.flush()
                self.block = block
            rows = x[(-start) % self.stride::self.stride].float()
            if not rows.shape[0]:
                return
            if name not in self.device:
                self.device[name] = torch.zeros(rows.shape[1], rows.shape[1], device=rows.device)
            self.device[name].addmm_(rows.T, rows)
            key = f"{block}.{name}"
            self.tokens[key] = self.tokens.get(key, 0) + rows.shape[0]
        return collect

    def flush(self) -> None:
        if self.block is None:
            return
        for name, total in self.device.items():
            key = f"{self.block}.{name}"
            if key in self.host:
                self.host[key] += total.cpu()
            else:
                self.host[key] = total.to("cpu", copy=True)
            total.zero_()
        self.block = None


def collect(args) -> None:
    from faquant.workflows import wan22_generate as gen
    from faquant.wan22_suite import video_seed

    prompts = args.prompts or args.hisqrot4_root / CALIBRATION_PROMPTS
    item = json.loads(prompts.read_text())[args.prompt]
    ns = argparse.Namespace(
        wan_root=args.wan_root, quant="hif4", ckpt_dir=args.ckpt_dir, convert_model_dtype=1,
        size="1280*720", hisqrot4_root=args.hisqrot4_root, hif4_levels="123",
        hif4_input_backend="quant_cy", hif4_weights=None, hif4_attention="qkpv", hif4_pv_normalizer="quantized",
        hif4_qk_rotation=True, hif4_qk_smooth=None, hif4_vo_rotation=False, hif4_attn_bf16="",
    )
    pipeline, _, size, _, _, quant_info = gen.build_pipeline(ns)
    blocks = block_range(args.blocks)
    bank = HessianBank(args.stride)
    model = getattr(pipeline, args.expert)
    for index in blocks:
        for path, name in HESSIAN_INPUTS.items():
            model.blocks[index].get_submodule(path).collector = bank.collector(index, name)

    def stop(*_args, **_kwargs):
        raise Done

    if args.expert == "high_noise_model":
        pipeline.low_noise_model.register_forward_pre_hook(stop)
    else:
        pipeline.vae.decode = stop
    seed = video_seed(item["prompt"], 0)
    started = time.time()
    try:
        pipeline.generate(item["prompt"], size=size, frame_num=81, shift=12.0, sample_solver="unipc",
                          sampling_steps=args.steps, guide_scale=(3.0, 4.0), seed=seed, offload_model=True)
    except Done:
        pass
    else:
        raise RuntimeError("generation ended without reaching the stop point")
    bank.flush()
    expected = {f"{index}.{name}" for index in blocks for name in HESSIAN_INPUTS.values()}
    if set(bank.host) != expected:
        raise RuntimeError(f"Hessians missing: {sorted(expected - set(bank.host))[:8]}")
    save({
        "hessians": bank.host, "tokens": bank.tokens, "expert": args.expert, "blocks": [blocks.start, blocks.stop],
        "stride": args.stride, "prompt_id": item["id"], "seed": seed, "attention": quant_info["attention"],
        "minutes": round((time.time() - started) / 60, 1),
    }, args.out)
    print(json.dumps({"out": str(args.out), "minutes": round((time.time() - started) / 60, 1),
                      "tokens_ffn0_first_block": bank.tokens[f"{blocks.start}.ffn0"]}))


def merge(args) -> None:
    files = sorted(glob.glob(args.inputs))
    if len(files) < 2:
        sys.exit(f"need at least two collect files, {args.inputs} matches {len(files)}")
    heldout_file = max(files, key=lambda f: int(re.search(r"_p(\d+)\.pt$", f).group(1)))
    total: dict[str, torch.Tensor] = {}
    tokens: dict[str, int] = {}
    heldout: dict[str, torch.Tensor] = {}
    heldout_tokens: dict[str, int] = {}
    meta = []
    for path in files:
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for key, hessian in data["hessians"].items():
            if key in total:
                total[key] += hessian
            else:
                total[key] = hessian.clone()
            tokens[key] = tokens.get(key, 0) + data["tokens"][key]
            if path == heldout_file:
                heldout[key], heldout_tokens[key] = hessian.clone(), data["tokens"][key]
        meta.append({k: data[k] for k in ("expert", "blocks", "stride", "prompt_id", "seed")})
        if path == heldout_file:
            heldout_prompt = data["prompt_id"]
        del data
    if len({(m["expert"], tuple(m["blocks"]), m["stride"]) for m in meta}) != 1:
        raise RuntimeError(f"collect files disagree: {meta}")
    if len({m["prompt_id"] for m in meta}) != len(meta):
        raise RuntimeError("a prompt was collected twice")
    expert, (first, last) = meta[0]["expert"], meta[0]["blocks"]
    out = args.out / expert
    for index in range(first, last):
        names = HESSIAN_INPUTS.values()
        save({
            "hessians": {name: total[f"{index}.{name}"] for name in names},
            "tokens": {name: tokens[f"{index}.{name}"] for name in names},
            "heldout": {name: heldout[f"{index}.{name}"] for name in names},
            "heldout_tokens": {name: heldout_tokens[f"{index}.{name}"] for name in names},
            "prompts": [m["prompt_id"] for m in meta], "heldout_prompt": heldout_prompt,
            "stride": meta[0]["stride"],
        }, out / f"block{index:02d}.pt")
    print(json.dumps({"expert": expert, "blocks": [first, last], "prompts": [m["prompt_id"] for m in meta]}))


def relative_error(delta: torch.Tensor, hessian: torch.Tensor, weight: torch.Tensor) -> float:
    """tr(dW H dW^T) / tr(W H W^T): the layer's output error on the calibration inputs, relative."""
    return float(((delta @ hessian) * delta).sum() / ((weight @ hessian) * weight).sum())


def solve(source: torch.Tensor, hessian: torch.Tensor, samples: int, damp: float):
    """HiF4 GPTQ of one weight; returns the quantized weight (fp32) and GPTQ's stats."""
    from faquant.gptq import GPTQ

    layer = nn.Linear(source.shape[1], source.shape[0], bias=False, device=source.device)
    layer.weight.data.copy_(source)
    solver = GPTQ(layer)
    solver.hessian, solver.samples = hessian, samples
    stats = solver.quantize(damp=damp, **GPTQ_SETTINGS)
    return layer.weight.detach(), stats


def gptq(args) -> None:
    from safetensors import safe_open

    from faquant.hif4 import fake_quantize_hif4
    from faquant.wan22_hif4 import BLOCK_LINEARS

    device = torch.device(args.device)
    shards = args.ckpt_dir / args.expert
    weight_map = json.loads((shards / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    weights: dict[str, torch.Tensor] = {}
    report: dict[str, dict] = {}
    for index in block_range(args.blocks):
        saved = torch.load(args.hessians / args.expert / f"block{index:02d}.pt", map_location="cpu",
                           weights_only=True)
        for path in BLOCK_LINEARS:
            key = f"blocks.{index}.{path}"
            with safe_open(str(shards / weight_map[f"{key}.weight"]), "pt", device="cpu") as handle:
                # the bf16 parameters the RTN build quantizes (convert_model_dtype casts the fp32 checkpoint)
                source = handle.get_tensor(f"{key}.weight").to(torch.bfloat16).to(device, torch.float32)
            rtn = fake_quantize_hif4(source, levels="123")
            name = LINEAR_HESSIAN[path]
            if name is None:
                weights[key] = rtn.to(torch.bfloat16).cpu()
                report[key] = {"rtn": None, "gptq": None, "kept": "rtn"}
                continue
            hessian, tokens = saved["hessians"][name].to(device), saved["tokens"][name]
            heldout, heldout_tokens = saved["heldout"][name].to(device), saved["heldout_tokens"][name]
            quantized, stats = solve(source, hessian, tokens, args.damp)
            fitted, _ = solve(source, hessian - heldout, tokens - heldout_tokens, args.damp)
            stored = quantized.to(torch.bfloat16)
            if not torch.equal(stored.float(), quantized):
                raise RuntimeError(f"{key}: GPTQ weight is not exactly representable in bf16")
            heldout /= heldout_tokens
            report[key] = {
                "rtn": relative_error(source - rtn, hessian / tokens, source),
                "gptq": relative_error(source - quantized, hessian / tokens, source),
                "heldout_rtn": relative_error(source - rtn, heldout, source),
                "heldout_gptq": relative_error(source - fitted, heldout, source),
                "damp_multiplier": stats.damp_multiplier, "tokens": tokens,
            }
            weights[key] = stored.cpu()
            del hessian, heldout
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.out.with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    save(weights, args.out)
    fitted = [r for r in report.values() if r["gptq"] is not None]
    print(json.dumps({"out": str(args.out), "linears": len(report),
                      "gptq_over_rtn_median": median([r["gptq"] / r["rtn"] for r in fitted]),
                      "heldout_gptq_over_rtn_median": median([r["heldout_gptq"] / r["heldout_rtn"] for r in fitted])}))


def assemble(args) -> None:
    from faquant.wan22_hif4 import BLOCK_LINEARS

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for part in sorted(args.parts.glob("*.json")):
        expert = part.stem.split("_b")[0]
        report.update({f"{expert}.{key}": row for key, row in json.loads(part.read_text()).items()})
    digests = {}
    for tag in TAGS:
        merged: dict[str, torch.Tensor] = {}
        for part in sorted(args.parts.glob(f"{tag}_b*.pt")):
            merged.update(torch.load(part, map_location="cpu", weights_only=True))
        expected = {f"blocks.{index}.{path}" for index in range(BLOCKS) for path in BLOCK_LINEARS}
        if set(merged) != expected:
            raise RuntimeError(f"{tag}: {len(expected - set(merged))} Linears missing, {len(set(merged) - expected)} extra")
        target = args.out / f"{tag}.pt"
        save(merged, target)
        digest = hashlib.sha256()
        with open(target, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        digests[tag] = digest.hexdigest()
    fitted = [r for r in report.values() if r["gptq"] is not None]
    hessians = sorted(args.hessians.glob("*/block*.pt"))
    first = torch.load(hessians[0], map_location="cpu", weights_only=True)
    error = {"linears": len(fitted)}
    for label, num, den in (("", "gptq", "rtn"), ("heldout_", "heldout_gptq", "heldout_rtn")):
        ratios = sorted(r[num] / r[den] for r in fitted)
        error |= {f"{label}gptq_over_rtn_median": ratios[len(ratios) // 2],
                  f"{label}gptq_over_rtn_p10": ratios[len(ratios) // 10],
                  f"{label}gptq_over_rtn_p90": ratios[9 * len(ratios) // 10]}
    manifest = {
        "gptq": {**GPTQ_SETTINGS, "damp": args.damp, "act_order_within_group": False, "scale_search": False,
                 "source_weights": "bf16 (convert_model_dtype)", "rtn_kept": ["cross_attn.k", "cross_attn.v"]},
        "calibration": {"prompts": first["prompts"], "heldout_prompt": first["heldout_prompt"],
                        "stride": first["stride"], "build": "hif4 rtn + qkpv + qk_rotation + quantized normalizer",
                        "inputs": "HiF4-quantized Linear inputs, both CFG branches"},
        "sha256": digests,
        "error": error,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (args.out / "error_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(manifest["error"] | {"sha256": digests}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt-dir", type=Path, required=True)
    c.add_argument("--wan-root", type=Path, required=True, help="Checkout of github.com/Wan-Video/Wan2.2.")
    c.add_argument("--hisqrot4-root", type=Path, required=True,
                   help="Checkout of github.com/GCC-HiFloat/HiSQRot4_Wan2.2 (its HiF4 input kernel).")
    c.add_argument("--prompts", type=Path,
                   help="Calibration prompt list (default: <hisqrot4-root>/" f"{CALIBRATION_PROMPTS}).")
    c.add_argument("--prompt", type=int, required=True, help="Index into the calibration prompt list.")
    c.add_argument("--expert", choices=TAGS, required=True)
    c.add_argument("--blocks", required=True, help="first:last (last excluded)")
    c.add_argument("--stride", type=int, default=4, help="Record every n-th token.")
    c.add_argument("--steps", type=int, default=40, help="Denoising steps (fewer only for a smoke test).")
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--inputs", required=True, help="Glob of the collect files of one expert and block range.")
    m.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gptq")
    g.add_argument("--ckpt-dir", type=Path, required=True)
    g.add_argument("--hessians", type=Path, required=True)
    g.add_argument("--expert", choices=TAGS, required=True)
    g.add_argument("--blocks", required=True)
    g.add_argument("--damp", type=float, default=0.01)
    g.add_argument("--device", default="cuda")
    g.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--parts", type=Path, required=True)
    a.add_argument("--hessians", type=Path, required=True, help="Any merged Hessian directory (for the manifest).")
    a.add_argument("--damp", type=float, default=0.01)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"collect": collect, "merge": merge, "gptq": gptq, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
