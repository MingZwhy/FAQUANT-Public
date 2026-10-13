#!/usr/bin/env python3
"""Generate Wan2.2-T2V-A14B videos for a job list, one GPU per process.

The model loads once; the worker then claims jobs from --jobs-file through
O_EXCL claim files in --claim-dir (local disk) until none are left. A video
is written outside the videos directory and renamed into place, so the
directory VBench reads only ever holds finished MP4 files. Per-video metadata
(seed, settings, timings, FlashAttention calls) goes to <run>/meta.

--quant hisqrot4 replaces every Linear of both DiT experts with the released
HiSQRot4 (GCC-HiFloat) W4A4 HiFloat4 path, loaded from its Stage-2 artifacts
exactly as its own launcher does. --quant hif4 puts plain HiF4 W4A4 (RTN weights,
dynamic inputs, no smoothing/rotation/clipping) on the same 400 block Linears per
expert (faquant.wan22_hif4). Either can add --hif4-attention qkpv, HiF4 Q/K/P/V in
every DiT attention (faquant.wan22_attention).

  python scripts/wan22_generate.py --run-dir RUN --jobs-file jobs.json --claim-dir claims \
      --ckpt-dir /path/to/Wan2.2-T2V-A14B --wan-root /path/to/Wan2.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

from faquant.wan22_suite import OFFICIAL, meta_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--claim-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True,
                        help="Checkout of github.com/Wan-Video/Wan2.2.")
    parser.add_argument("--quant", choices=("none", "hisqrot4", "hif4"), default="none")
    parser.add_argument("--hisqrot4-root", type=Path,
                        help="Checkout of github.com/GCC-HiFloat/HiSQRot4_Wan2.2; needed by --quant hisqrot4 "
                             "and the quant_cy input kernel.")
    parser.add_argument("--hisqrot4-artifacts", type=Path)
    parser.add_argument("--hisqrot4-chunk-rows", type=int, default=16384)
    parser.add_argument("--hif4-levels", choices=("123", "12", "13"), default="123",
                        help="HiF4 scale levels for weights and inputs (123: official three-level format).")
    parser.add_argument("--hif4-input-backend", choices=("faquant", "quant_cy"), default="faquant",
                        help="Input quantizer: faquant.hif4 (torch) or the quant_cy CUDA kernel (123 only).")
    parser.add_argument("--hif4-weights", type=Path,
                        help="HiF4 block weights to use instead of RTN: a directory with <expert>.pt and "
                             "manifest.json (scripts/calibrate_wan22_gptq.py assemble).")
    parser.add_argument("--hif4-attention", choices=("none", "qkpv"), default="none",
                        help="qkpv: also HiF4 Q/K/P/V in every DiT attention (faquant.wan22_attention, 123 only), "
                             "with --quant hif4 or hisqrot4.")
    parser.add_argument("--hif4-pv-normalizer", choices=("unquantized", "quantized"), default="unquantized",
                        help="Softmax denominator from the unquantized P or the quantized one (P-Reordering).")
    parser.add_argument("--hif4-qk-rotation", action="store_true",
                        help="Head_dim Hadamard on Q and K after RoPE, before they are quantized.")
    parser.add_argument("--hif4-qk-smooth", type=Path,
                        help="Smooth-QK scales (scripts/calibrate_wan22_qk_smooth.py), per expert and block.")
    parser.add_argument("--hif4-vo-rotation", action="store_true",
                        help="Fold a per-head Hadamard through self-attention v and o before quantization (--quant hif4).")
    parser.add_argument("--hif4-attn-bf16", default="",
                        help="Attention kept in bf16, e.g. 'low:33,34,38;cross' (experts high, low, both).")
    parser.add_argument("--size", default=OFFICIAL.size)
    parser.add_argument("--frame-num", type=int, default=OFFICIAL.frame_num)
    parser.add_argument("--sample-steps", type=int, default=OFFICIAL.sample_steps)
    parser.add_argument("--sample-shift", type=float, default=OFFICIAL.sample_shift)
    parser.add_argument("--guide-scale", type=float, nargs=2, default=OFFICIAL.guide_scale)
    parser.add_argument(
        "--convert-model-dtype",
        type=int,
        choices=(0, 1),
        default=1,
        help="Cast DiT parameters to bf16 (official single-GPU command; FSDP computes in bf16 too).",
    )
    parser.add_argument("--offload-model", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-jobs", type=int, default=0, help="Stop after this many videos (0: no limit).")
    parser.add_argument("--stop-file", type=Path, help="Exit cleanly after the current video once this file exists.")
    return parser.parse_args()


def claim(claim_dir: Path, key: str) -> bool:
    try:
        fd = os.open(claim_dir / f"{key}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def build_pipeline(args: argparse.Namespace):
    uses_kernels = args.quant == "hisqrot4" or (
        args.quant == "hif4" and args.hif4_input_backend == "quant_cy"
    )
    if uses_kernels and args.hisqrot4_root is None:
        raise ValueError("--hisqrot4-root is required for HiSQRot4 and the quant_cy kernel")
    if args.quant == "hisqrot4" and args.hisqrot4_artifacts is None:
        raise ValueError("--quant hisqrot4 needs --hisqrot4-artifacts (its Stage-2 output)")
    sys.path.insert(0, str(args.wan_root))
    import wan.modules.model as wan_model
    import wan.text2video as wan_text2video
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.modules import attention as wan_attention

    calls = {"flash_attention": 0}
    attention = {"impl": wan_model.flash_attention}
    bf16_attention = wan_model.flash_attention

    def counted(*a, **k):
        calls["flash_attention"] += 1
        return attention["impl"](*a, **k)

    wan_model.flash_attention = counted

    quant_info: dict = {"method": "none"}
    if args.quant == "hisqrot4":
        sys.path.insert(0, str(args.hisqrot4_root))
        from hifx4_linear_quant import Hifx4Config, PTQPreparedLinearWrapper, patch_want2v_configure_model

        from faquant.wan22_hisqrot4 import patch_wrapper

        patch_wrapper(PTQPreparedLinearWrapper, chunk_rows=args.hisqrot4_chunk_rows)

        artifacts = args.hisqrot4_artifacts
        manifests = {
            tag: json.loads((artifacts / tag / "hifx4/manifest.json").read_text())
            for tag in ("low_noise_model", "high_noise_model")
        }
        # The artifact signature hashes the fp32 Linear dtypes of HiSQRot4's own runs;
        # with bf16 DiT parameters (as in the BF16 baseline) it cannot match, so its
        # check is off. Coverage is still enforced: replace_wan_linear_with_hifx4 asserts
        # every block Linear was wrapped, and the wrapped count is recorded below.
        cfg = Hifx4Config(
            qtype="hifx4",
            quant_method="ptq",
            stage="infer",
            ptq_artifact_root=str(artifacts),
            ptq_validate_artifact=False,
            timestep_count=args.sample_steps,
            act_quant_mode="lookup",
        )
        patch_want2v_configure_model(
            wan_text2video_module=wan_text2video,
            hifloat4_root=str(args.hisqrot4_root / "hifloat4"),
            cfg=cfg,
        )
        quant_info = {"method": "hisqrot4", "artifacts": str(artifacts), "manifests": manifests}

    config = WAN_CONFIGS[OFFICIAL.task]
    pipeline = wan_text2video.WanT2V(
        config=config,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=bool(args.convert_model_dtype),
    )
    if args.quant == "hisqrot4":
        wrapped = {
            tag: sum(isinstance(m, PTQPreparedLinearWrapper) for m in getattr(pipeline, tag).modules())
            for tag in ("low_noise_model", "high_noise_model")
        }
        if any(count != manifests[tag]["prepared_layers"] for tag, count in wrapped.items()):
            raise RuntimeError(f"HiSQRot4 wrapped {wrapped}, manifests list {manifests}")
        quant_info["wrapped_linears"] = wrapped
    if args.quant == "hif4":
        from faquant.hif4 import HIF4_REFERENCE_COMMIT
        from faquant.wan22_hif4 import faquant_qdq, quant_cy_qdq, quantize_dit_blocks

        if (args.hif4_input_backend == "quant_cy" or args.hif4_attention != "none") and args.hif4_levels != "123":
            raise ValueError("the quant_cy and attention kernels implement only the three-level format")
        weight_qdq = faquant_qdq(args.hif4_levels)
        input_qdq = (
            quant_cy_qdq(str(args.hisqrot4_root / "hifloat4"))
            if args.hif4_input_backend == "quant_cy" else faquant_qdq(args.hif4_levels)
        )
        if args.hif4_vo_rotation:
            from faquant.wan22_hif4 import fold_value_head_hadamard

            for tag in ("low_noise_model", "high_noise_model"):
                fold_value_head_hadamard(getattr(pipeline, tag))
        manifest = None
        if args.hif4_weights:
            if args.hif4_levels != "123" or args.hif4_vo_rotation:
                raise ValueError("--hif4-weights are three-level HiF4 of the unrotated weights")
            manifest = json.loads((args.hif4_weights / "manifest.json").read_text())
        wrapped = {}
        for tag in ("low_noise_model", "high_noise_model"):
            weights = (torch.load(args.hif4_weights / f"{tag}.pt", map_location="cpu", mmap=True, weights_only=True)
                       if args.hif4_weights else None)
            wrapped[tag] = quantize_dit_blocks(getattr(pipeline, tag), weight_qdq=weight_qdq, input_qdq=input_qdq,
                                               weights=weights)
            del weights
        torch.cuda.empty_cache()
        quant_info = {
            "method": "hif4_gptq" if manifest else "hif4_rtn", "levels": args.hif4_levels,
            "input_backend": args.hif4_input_backend,
            "weights_from": str(next(pipeline.low_noise_model.parameters()).dtype),
            "hif4_reference_commit": HIF4_REFERENCE_COMMIT, "wrapped_linears": wrapped,
        }
        if manifest:
            quant_info["weights"] = {"dir": str(args.hif4_weights),
                                     **{k: manifest[k] for k in ("gptq", "calibration", "sha256")}}
    if args.hif4_vo_rotation and args.quant != "hif4":
        raise ValueError("--hif4-vo-rotation folds into the weights before HiF4 RTN (--quant hif4)")
    if args.hif4_weights and args.quant != "hif4":
        raise ValueError("--hif4-weights replace the RTN weights of --quant hif4")
    quant_info["attention"] = {"qkpv": False}
    if args.hif4_attention == "qkpv":
        if args.quant == "none":
            raise ValueError("--hif4-attention qkpv goes with --quant hif4 or hisqrot4")
        from faquant.wan22_attention import KEY_TILE, make_flash_attention, parse_bf16_attention, patch_attention_modules
        from faquant.wan22_hif4 import faquant_qdq, quant_cy_qdq

        # Q/K/V use the input quantizer of the Linear path (HiSQRot4's is the quant_cy kernel)
        attention_qdq = (
            quant_cy_qdq(str(args.hisqrot4_root / "hifloat4"))
            if args.quant == "hisqrot4" or args.hif4_input_backend == "quant_cy" else faquant_qdq("123")
        )
        attention["impl"] = make_flash_attention(
            attention_qdq, quantized_normalizer=args.hif4_pv_normalizer == "quantized",
            qk_rotation=args.hif4_qk_rotation, fallback=bf16_attention)
        quant_info["attention"] = {
            "qkpv": True, "qkv_axis": "head_dim", "p_key_tile": KEY_TILE, "normalizer": args.hif4_pv_normalizer,
            "qk_rotation": args.hif4_qk_rotation, "vo_rotation": args.hif4_vo_rotation,
        }
        if args.hif4_qk_smooth or args.hif4_attn_bf16:
            patch_attention_modules(wan_model)
        if args.hif4_qk_smooth:
            smooth = torch.load(args.hif4_qk_smooth, map_location="cpu")
            for tag in ("low_noise_model", "high_noise_model"):
                for block, scale in zip(getattr(pipeline, tag).blocks, smooth[tag]):
                    block.self_attn.register_buffer("faquant_qk_smooth_scale", scale.float(), persistent=False)
            quant_info["attention"]["qk_smooth"] = {"file": str(args.hif4_qk_smooth), **smooth.get("meta", {})}
        if args.hif4_attn_bf16:
            keep = parse_bf16_attention(args.hif4_attn_bf16)
            for tag in ("low_noise_model", "high_noise_model"):
                for index, block in enumerate(getattr(pipeline, tag).blocks):
                    block.self_attn.faquant_attention_bf16 = index in keep[tag]
                    block.cross_attn.faquant_attention_bf16 = bool(keep["cross"])
            quant_info["attention"]["bf16"] = args.hif4_attn_bf16
    elif args.hif4_qk_rotation or args.hif4_qk_smooth or args.hif4_attn_bf16:
        raise ValueError("the attention options need --hif4-attention qkpv")
    backend = {
        "flash_attn_3": bool(wan_attention.FLASH_ATTN_3_AVAILABLE),
        "flash_attn_2": bool(wan_attention.FLASH_ATTN_2_AVAILABLE),
    }
    try:
        import flash_attn

        backend["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        pass
    return pipeline, config, SIZE_CONFIGS[args.size], calls, backend, quant_info


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", stream=sys.stdout)

    videos = args.run_dir / "videos"
    meta = args.run_dir / "meta"
    tmp = args.run_dir / "tmp"
    for path in (videos, meta, tmp, args.claim_dir):
        path.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs_file.read_text())

    t0 = time.time()
    pipeline, config, size, calls, backend, quant_info = build_pipeline(args)
    load_seconds = time.time() - t0
    logging.info("pipeline ready in %.0fs (quant=%s backend=%s)", load_seconds, args.quant, backend)

    from wan.utils.utils import save_video

    settings = {
        "task": OFFICIAL.task,
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "guide_scale": list(args.guide_scale),
        "sample_solver": OFFICIAL.sample_solver,
        "fps": config.sample_fps,
        "negative_prompt": config.sample_neg_prompt,
        "convert_model_dtype": bool(args.convert_model_dtype),
        "offload_model": bool(args.offload_model),
        "ckpt_dir": str(args.ckpt_dir),
        "wan_root": str(args.wan_root),
    }
    done = 0
    for position, job in enumerate(jobs):
        if args.stop_file and args.stop_file.exists():
            logging.info("stop file %s present, exiting", args.stop_file)
            break
        final = videos / job["name"]
        if final.exists() or not claim(args.claim_dir, str(position)):
            continue
        calls["flash_attention"] = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        video = pipeline.generate(
            job["prompt"],
            size=size,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=OFFICIAL.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=tuple(args.guide_scale),
            seed=job["seed"],
            offload_model=bool(args.offload_model),
        )
        generate_seconds = time.time() - start
        partial = tmp / f"{os.getpid()}-{position}.mp4"
        save_video(
            tensor=video[None], save_file=str(partial), fps=config.sample_fps,
            nrow=1, normalize=True, value_range=(-1, 1),
        )
        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError(f"save_video wrote nothing for {job['name']}")
        os.replace(partial, final)
        record = {
            **job,
            "settings": settings,
            "quant": quant_info,
            "frames": int(video.shape[1]),
            "height": int(video.shape[2]),
            "width": int(video.shape[3]),
            "flash_attention_calls": calls["flash_attention"],
            "attention_backend": backend,
            "generate_seconds": round(generate_seconds, 1),
            "load_seconds": round(load_seconds, 1),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "time": time.strftime("%F %T"),
        }
        (meta / meta_name(job)).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        done += 1
        del video
        logging.info(
            "done %s (%.0fs, peak %.1f GB, fa calls %d)",
            job["name"], generate_seconds, record["peak_memory_gb"], record["flash_attention_calls"],
        )
        if args.max_jobs and done >= args.max_jobs:
            break
    logging.info("worker finished: %d videos", done)


if __name__ == "__main__":
    main()
