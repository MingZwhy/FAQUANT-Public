#!/usr/bin/env python3
"""Smooth-QK scales for the Wan2.2 DiT, with the statistic of the Qwen recipe.

collect: one calibration prompt (HiSQRot4's calibration set, disjoint from
         VBench) through all 40 steps and both CFG branches, on the build the scales are for
         (HiF4 Linear by default; attention in bf16), recording per expert and block the
         per-(head, channel) max |Q| and |K| after RoPE.
merge:   max over prompts; s_c = Kmax_c^alpha / Qmax_c^(1-alpha) over head_dim channels (shared by
         the heads, as in Qwen; per head kept too), divided by its median, clamped to [1/32, 32].

  python scripts/calibrate_wan22_qk_smooth.py collect --ckpt-dir CKPT --prompt 0 --out DIR/stats_0.pt
  python scripts/calibrate_wan22_qk_smooth.py merge --stats DIR --out OUT/qk_smooth.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# The thirty calibration prompts of the HiSQRot4 release, disjoint from VBench.
CALIBRATION_PROMPTS = Path("data/prompts/OpenS2V-5M_to_mm_calib_30.json")
TAGS = ("high_noise_model", "low_noise_model")


def abs_max(x):
    """Per (head, channel) max |x| over batch and tokens, without a full-size temporary."""
    return torch.maximum(x.amax(dim=(0, 1)), -x.amin(dim=(0, 1))).float()


def collect(args) -> None:
    from faquant.workflows import wan22_generate as gen
    from faquant.wan22_attention import patch_attention_modules
    from faquant.wan22_suite import video_seed

    prompts = args.prompts or args.hisqrot4_root / CALIBRATION_PROMPTS
    item = json.loads(prompts.read_text())[args.prompt]
    ns = argparse.Namespace(
        wan_root=args.wan_root, quant=args.quant, ckpt_dir=args.ckpt_dir, convert_model_dtype=1,
        size="1280*720", hisqrot4_root=args.hisqrot4_root, hif4_attention="none",
        hif4_levels="123", hif4_input_backend="quant_cy", hif4_vo_rotation=args.vo_rotation,
        hif4_weights=None, hif4_qk_rotation=False, hif4_qk_smooth=None, hif4_attn_bf16="",
    )
    pipeline, _, size, _, _, quant_info = gen.build_pipeline(ns)
    import wan.modules.model as wan_model

    where = {id(block.self_attn): (tag, i) for tag in TAGS for i, block in enumerate(getattr(pipeline, tag).blocks)}
    stats: dict = {}

    def collector(module, q, k):
        key = where[id(module)]
        qm, km = abs_max(q), abs_max(k)
        if key in stats:
            qm, km = torch.maximum(stats[key][0], qm), torch.maximum(stats[key][1], km)
        stats[key] = (qm, km)

    patch_attention_modules(wan_model, collector=collector)
    seed = video_seed(item["prompt"], 0)
    started = time.time()
    pipeline.generate(item["prompt"], size=size, frame_num=81, shift=12.0, sample_solver="unipc",
                      sampling_steps=40, guide_scale=(3.0, 4.0), seed=seed, offload_model=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stats": {tag: [tuple(t.cpu() for t in stats[(tag, i)]) for i in range(len(getattr(pipeline, tag).blocks))]
                  for tag in TAGS},
        "prompt_id": item["id"], "seed": seed, "quant": quant_info["method"], "vo_rotation": args.vo_rotation,
        "minutes": round((time.time() - started) / 60, 1),
    }, args.out)


def merge(args) -> None:
    files = sorted(args.stats.glob("stats_*.pt"))
    loaded = [torch.load(f, map_location="cpu") for f in files]
    if not loaded:
        sys.exit(f"no stats_*.pt in {args.stats}")
    floor = torch.finfo(torch.float32).tiny
    out = {"per_head": {}}
    for tag in TAGS:
        shared, per_head = [], []
        for block in range(len(loaded[0]["stats"][tag])):
            qmax = torch.stack([d["stats"][tag][block][0] for d in loaded]).amax(0)
            kmax = torch.stack([d["stats"][tag][block][1] for d in loaded]).amax(0)
            for target, (qm, km) in ((shared, (qmax.amax(0), kmax.amax(0))), (per_head, (qmax, kmax))):
                s = km.clamp_min(floor).pow(args.alpha) / qm.clamp_min(floor).pow(1 - args.alpha)
                s = s / s.median(dim=-1, keepdim=True).values
                target.append(s.clamp(1 / args.clamp, args.clamp))
        out[tag] = shared
        out["per_head"][tag] = per_head
    out["meta"] = {
        "alpha": args.alpha, "statistic": "max", "clamp": args.clamp, "basis": "post-RoPE, pre-Hadamard",
        "scale": "shared by heads (per_head kept separately)", "prompts": [d["prompt_id"] for d in loaded],
        "build": sorted({f"{d['quant']}{'+vo' if d['vo_rotation'] else ''}" for d in loaded}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    spread = [float(s.max() / s.min()) for tag in TAGS for s in out[tag]]
    print(json.dumps({**out["meta"], "files": len(files), "scale_max_over_min": [min(spread), max(spread)]}))


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
    c.add_argument("--quant", choices=("none", "hif4"), default="hif4")
    c.add_argument("--vo-rotation", action="store_true")
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--stats", type=Path, required=True)
    m.add_argument("--alpha", type=float, default=0.5)
    m.add_argument("--clamp", type=float, default=32.0)
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args) if args.cmd == "collect" else merge(args)


if __name__ == "__main__":
    main()
