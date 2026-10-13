#!/usr/bin/env python3
"""Official VBench (standard prompt suite, 16 dimensions) on a generated video directory.

Follows the VBench repository flow: temporal flickering is evaluated on the
static videos picked by static_filter.py from the 25 samples per prompt; every
other dimension runs evaluate.py in vbench_standard mode on <run>/videos with
the local checkpoints. Dimensions run in parallel, one process per GPU (or a
torchrun group with --gpus-per-dim). Scores are aggregated with VBench's own
normalization and weights (scripts/constant.py of the VBench repository) into
Quality, Semantic and Total.

  python scripts/wan22_vbench_eval.py --run-dir RUN --vbench-root VBENCH [--gpus 0,1,2,3] [--dims ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from faquant.wan22_suite import DIMENSIONS, suite_jobs

# VBench scripts/constant.py
NORMALIZE = {
    "subject_consistency": (0.1462, 1.0), "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0), "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0), "aesthetic_quality": (0.0, 1.0), "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0), "multiple_objects": (0.0, 1.0), "human_action": (0.0, 1.0),
    "color": (0.0, 1.0), "spatial_relationship": (0.0, 1.0), "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855), "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
}
WEIGHT = {dim: (0.5 if dim == "dynamic_degree" else 1.0) for dim in DIMENSIONS}
QUALITY = ("subject_consistency", "background_consistency", "temporal_flickering",
           "motion_smoothness", "aesthetic_quality", "imaging_quality", "dynamic_degree")
SEMANTIC = ("object_class", "multiple_objects", "human_action", "color", "spatial_relationship",
            "scene", "appearance_style", "temporal_style", "overall_consistency")
# Wan2.2-T2V-A14B (w/o prompt-optimized), sampled and evaluated by the VBench team, 2026-02-18.
OFFICIAL_WAN22 = {
    "subject_consistency": 97.46, "background_consistency": 96.67, "temporal_flickering": 98.92,
    "motion_smoothness": 97.93, "dynamic_degree": 68.24, "aesthetic_quality": 63.52,
    "imaging_quality": 71.69, "object_class": 85.04, "multiple_objects": 74.61,
    "human_action": 83.40, "color": 88.93, "spatial_relationship": 80.22, "scene": 34.38,
    "appearance_style": 20.27, "temporal_style": 23.13, "overall_consistency": 24.68,
    "quality_score": 85.03, "semantic_score": 72.92, "total_score": 82.61,
}


def aggregate(scores: dict[str, float]) -> dict[str, float]:
    norm = {d: (scores[d] - lo) / (hi - lo) * WEIGHT[d] for d, (lo, hi) in NORMALIZE.items()}
    quality = sum(norm[d] for d in QUALITY) / sum(WEIGHT[d] for d in QUALITY)
    semantic = sum(norm[d] for d in SEMANTIC) / sum(WEIGHT[d] for d in SEMANTIC)
    return {"quality_score": quality, "semantic_score": semantic, "total_score": (4 * quality + semantic) / 5}


def dim_result(out: Path, dim: str):
    hits = sorted(out.glob("*_eval_results.json"))
    for path in reversed(hits):
        data = json.loads(path.read_text())
        if dim in data:
            return data[dim]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True,
                        help="Checkout of github.com/Vchitect/VBench with its evaluation checkpoints.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter of the VBench environment.")
    parser.add_argument("--cache-dir", type=Path,
                        help="Put VBENCH_CACHE_DIR, HF_HOME and TORCH_HOME under this directory.")
    parser.add_argument("--out", type=Path, help="Default <run-dir>/vbench")
    parser.add_argument("--dims", nargs="+", default=list(DIMENSIONS))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--gpus-per-dim", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=29700, help="Distinct per concurrent eval driver.")
    parser.add_argument("--allow-missing", action="store_true", help="Evaluate an incomplete suite (smoke tests).")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help=(
            "Evaluate only these sample indices (e.g. 0 = one video per prompt) through a "
            "directory of symlinks. Not the official protocol: temporal flickering then "
            "filters only the samples present."
        ),
    )
    args = parser.parse_args()
    vbench = args.vbench_root

    videos = args.run_dir / "videos"
    tag = "" if not args.indices else "_idx" + "-".join(map(str, args.indices))
    out = args.out or args.run_dir / f"vbench{tag}"
    out.mkdir(parents=True, exist_ok=True)
    expected = suite_jobs(vbench / "vbench/VBench_full_info.json")
    stray = sorted(set(os.listdir(videos)) - {job["name"] for job in expected})
    if stray:
        sys.exit(f"{len(stray)} unexpected files in {videos}, e.g. {stray[:3]}")
    if args.indices:
        expected = [job for job in expected if job["index"] in args.indices]
        subset = out / "videos"
        subset.mkdir(exist_ok=True)
        for job in expected:
            link = subset / job["name"]
            # Relative targets: some network filesystems truncate symlink targets longer
            # than 255 bytes, which absolute paths to long VBench prompts exceed.
            target = os.path.relpath(videos / job["name"], subset)
            if (videos / job["name"]).exists() and not link.exists():
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(target)
        videos = subset
    missing = [job["name"] for job in expected if not (videos / job["name"]).exists()]
    if missing and not args.allow_missing:
        sys.exit(f"{len(missing)} of {len(expected)} videos missing, e.g. {missing[:3]}")

    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        PYTHONPATH=str(vbench), TOKENIZERS_PARALLELISM="false",
    )
    if args.cache_dir:
        # TORCH_HOME holds the DINO weights that the local dino hubconf would otherwise download.
        env.update(VBENCH_CACHE_DIR=str(args.cache_dir / "vbench_cache"),
                   HF_HOME=str(args.cache_dir / "hf_home"), TORCH_HOME=str(args.cache_dir / "torch_home"))
    gpus = [g for g in args.gpus.split(",") if g]
    groups = [gpus[i:i + args.gpus_per_dim] for i in range(0, len(gpus) - args.gpus_per_dim + 1, args.gpus_per_dim)]
    logs = out / "logs"
    logs.mkdir(exist_ok=True)

    def command(dim: str, gpu_group: list[str], videos_path: Path) -> list[str]:
        evaluate = [
            str(vbench / "evaluate.py"), "--videos_path", str(videos_path), "--dimension", dim,
            "--output_path", str(out / dim), "--load_ckpt_from_local", "True", "--mode", "vbench_standard",
        ]
        if len(gpu_group) == 1:
            return [args.python, *evaluate]
        return [args.python, "-m", "torch.distributed.run", f"--nproc_per_node={len(gpu_group)}",
                f"--master_port={args.port_base + int(gpu_group[0])}", *evaluate]

    def process_env(gpu_group: list[str]) -> dict:
        # VBench's dist_init() opens a process group even for one process, on port
        # 29500 unless MASTER_PORT is set; give every concurrent process its own.
        return {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpu_group),
                "MASTER_PORT": str(args.port_base + int(gpu_group[0]))}

    pending = [d for d in args.dims if dim_result(out / d, d) is None]
    filtered = out / "temporal_flickering_static" / "filtered_videos"
    queue = [d for d in pending if d != "temporal_flickering"]
    need_static = "temporal_flickering" in pending
    running: dict[str, tuple[subprocess.Popen, list[str]]] = {}
    free = list(groups)
    static_proc = None
    if need_static and not (out / "temporal_flickering_static" / "filtered_static_video.json").exists():
        group = free.pop(0)
        static_proc = subprocess.Popen(
            [args.python, str(vbench / "static_filter.py"), "--videos_path", str(videos),
             "--result_path", str(out / "temporal_flickering_static")],
            cwd=vbench, env={**env, "CUDA_VISIBLE_DEVICES": group[0]},
            stdout=(logs / "static_filter.log").open("w"), stderr=subprocess.STDOUT,
        )
        running["static_filter"] = (static_proc, group)
    elif need_static:
        queue.append("temporal_flickering")

    start = time.time()
    failed = []
    while queue or running:
        for name, (proc, group) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            free.append(group)
            if proc.returncode != 0:
                failed.append(f"{name}:exit{proc.returncode}")
            elif name == "static_filter":
                queue.append("temporal_flickering")
            print(f"[{time.strftime('%T')}] {name} finished rc={proc.returncode}", flush=True)
        while queue and free:
            dim = queue.pop(0)
            group = free.pop(0)
            path = filtered if dim == "temporal_flickering" else videos
            proc = subprocess.Popen(
                command(dim, group, path), cwd=vbench, env=process_env(group),
                stdout=(logs / f"{dim}.log").open("w"), stderr=subprocess.STDOUT,
            )
            running[dim] = (proc, group)
            print(f"[{time.strftime('%T')}] start {dim} on GPUs {','.join(group)}", flush=True)
        time.sleep(20)

    scores, per_video = {}, {}
    for dim in args.dims:
        result = dim_result(out / dim, dim)
        if result is not None:
            scores[dim] = float(result[0])
            per_video[dim] = result[1]
    summary = {
        "run_dir": str(args.run_dir), "videos_missing": len(missing), "failed": failed,
        "minutes": round((time.time() - start) / 60, 1),
        "scores": {d: round(100 * s, 4) for d, s in scores.items()},
    }
    if len(scores) == len(DIMENSIONS):
        summary.update({k: round(100 * v, 4) for k, v in aggregate(scores).items()})
        summary["official_wan22_wo_prompt_extend"] = OFFICIAL_WAN22
        summary["delta_vs_official"] = {
            k: round(summary.get(k, summary["scores"].get(k)) - v, 4)
            for k, v in OFFICIAL_WAN22.items()
        }
    static_info = out / "temporal_flickering_static" / "filtered_static_video.json"
    if static_info.exists():
        counts = [v["static_count"] for v in json.loads(static_info.read_text()).values()]
        summary["temporal_flickering_static"] = {
            "prompts": len(counts), "videos": sum(counts), "prompts_below_5": sum(c < 5 for c in counts),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "per_video.json").write_text(json.dumps(per_video) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "official_wan22_wo_prompt_extend"}, indent=2))


if __name__ == "__main__":
    main()
