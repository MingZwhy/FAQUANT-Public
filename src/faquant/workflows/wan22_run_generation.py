#!/usr/bin/env python3
"""Generate a Wan2.2-T2V-A14B video suite on one machine, one worker per GPU.

The scheduler builds the job list, keeps its share of it (--shard K/N lets N
machines write into one shared run directory; videos all cost the same), and
starts one single-GPU worker per GPU (faquant.workflows.wan22_generate),
staggered so that the fp32 checkpoint loads do not all overlap in host memory.
A crashed worker is restarted twice, and STATUS_<shard>.json in the run
directory is rewritten once a minute.

Suites:
  vbench             VBench standard prompt suite, 6,220 videos (faquant.wan22_suite)
  hisqrot4_vbench30  the 30 prompts HiSQRot4 reports on, named as its own scripts name them

  python scripts/run_wan22_generation.py --run-dir RUN --ckpt-dir /path/to/Wan2.2-T2V-A14B \\
      --wan-root /path/to/Wan2.2 --vbench-root /path/to/VBench --gpus 0,1,2,3 --indices 0 \\
      [--hisqrot4-root /path/to/HiSQRot4_Wan2.2] [--worker-args "--quant hif4"]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from faquant.wan22_suite import suite_jobs, video_seed


def suite(args: argparse.Namespace) -> list[dict]:
    if args.suite == "vbench":
        if args.vbench_root is None:
            raise SystemExit("--suite vbench needs --vbench-root")
        return suite_jobs(args.vbench_root / "vbench/VBench_full_info.json")
    if args.hisqrot4_root is None:
        raise SystemExit("--suite hisqrot4_vbench30 needs --hisqrot4-root")
    items = json.loads((args.hisqrot4_root / "data/prompts/OpenS2V-5M_to_mm_vbench_30.json").read_text())
    return [
        {
            "prompt": item["prompt"],
            "index": 0,
            "seed": video_seed(item["prompt"], 0),
            "name": os.path.basename(item["path"]).replace("/", "_"),
            "id": item["id"],
        }
        for item in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True, help="Checkout of github.com/Wan-Video/Wan2.2.")
    parser.add_argument("--hisqrot4-root", type=Path,
                        help="Checkout of github.com/GCC-HiFloat/HiSQRot4_Wan2.2, passed on to the workers.")
    parser.add_argument("--vbench-root", type=Path, help="Checkout of github.com/Vchitect/VBench (--suite vbench).")
    parser.add_argument("--suite", choices=("vbench", "hisqrot4_vbench30"), default="vbench")
    parser.add_argument("--shard", default="0/1", help="K/N: keep every N-th job starting at job K.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Work the job list from its end; two groups of machines can then share one list from both ends.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="Start at this position of the job list and wrap around; a third group can start in its middle.",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--stagger", type=float, default=20.0, help="Seconds between worker starts.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N jobs of the suite.")
    parser.add_argument("--indices", type=int, nargs="+", help="Only these sample indices (0 = one video per prompt).")
    parser.add_argument("--work-dir", type=Path,
                        help="Directory for the job list and the O_EXCL claim files; it must be on a filesystem "
                             "with atomic exclusive create (default: <run-dir>/work/<shard>).")
    parser.add_argument("--python", default=sys.executable, help="Interpreter of the generation environment.")
    parser.add_argument("--worker-args", default="", help="Extra arguments passed to every worker.")
    args = parser.parse_args()

    shard, shards = map(int, args.shard.split("/"))
    if not 0 <= shard < shards:
        parser.error(f"--shard {args.shard} is not K/N with 0 <= K < N")
    tag = f"shard{shard}of{shards}"
    jobs = suite(args)
    if args.indices:
        jobs = [job for job in jobs if job["index"] in args.indices]
    if args.limit:
        jobs = jobs[: args.limit]
    if args.rotate:
        jobs = jobs[args.rotate:] + jobs[: args.rotate]
    if args.reverse:
        jobs = jobs[::-1]
    mine = jobs[shard::shards]

    work = args.work_dir or args.run_dir / "work" / tag
    claims = work / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    jobs_file = work / "jobs.json"
    jobs_file.write_text(json.dumps(mine, ensure_ascii=False, indent=1) + "\n")
    logs = args.run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    videos = args.run_dir / "videos"
    # No worker of this shard is alive yet, so a claim without a video is stale.
    for stale in claims.glob("*.claim"):
        if not (videos / mine[int(stale.stem)]["name"]).exists():
            stale.unlink()

    manifest = args.run_dir / "RUN.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "suite": args.suite, "indices": args.indices, "videos": len(jobs), "shards": shards,
            "worker_args": args.worker_args, "created": time.strftime("%F %T"),
        }, indent=2) + "\n")

    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    # Touch <run>/STOP_<shard> to drain: workers finish their current video and
    # exit 0, and the scheduler does not restart them.
    stop = args.run_dir / f"STOP_{tag}"
    command = [
        args.python, "-m", "faquant.workflows.wan22_generate",
        "--run-dir", str(args.run_dir), "--jobs-file", str(jobs_file),
        "--claim-dir", str(claims), "--ckpt-dir", str(args.ckpt_dir),
        "--wan-root", str(args.wan_root),
        *(["--hisqrot4-root", str(args.hisqrot4_root)] if args.hisqrot4_root else []),
        "--stop-file", str(stop),
        *shlex.split(args.worker_args),
    ]
    gpus = [gpu for gpu in args.gpus.split(",") if gpu]

    def launch(gpu: str) -> subprocess.Popen:
        log = (logs / f"{tag}_gpu{gpu}.log").open("a")
        return subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            env={**env, "CUDA_VISIBLE_DEVICES": gpu},
        )

    workers: dict[str, subprocess.Popen] = {}
    for gpu in gpus:
        workers[gpu] = launch(gpu)
        time.sleep(args.stagger)
    restarts = {gpu: 0 for gpu in gpus}
    failed: list[str] = []
    started = time.time()
    while workers:
        for gpu, process in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            del workers[gpu]
            if code != 0:
                failed.append(f"gpu{gpu}:exit{code}")
                for owned in claims.glob("*.claim"):
                    if owned.read_text().strip() == str(process.pid) and not (
                        videos / mine[int(owned.stem)]["name"]
                    ).exists():
                        owned.unlink()
                pending = any(not (videos / job["name"]).exists() for job in mine)
                if pending and restarts[gpu] < 2:
                    restarts[gpu] += 1
                    workers[gpu] = launch(gpu)
        done = sum((videos / job["name"]).exists() for job in mine)
        status = args.run_dir / f"STATUS_{tag}.json"
        partial = status.with_suffix(".tmp")
        partial.write_text(json.dumps({
            "time": time.strftime("%F %T"),
            "elapsed_hours": round((time.time() - started) / 3600, 2),
            "videos_done": done, "videos_total": len(mine),
            "workers_alive": sorted(workers, key=int), "failed": failed,
            "draining": stop.exists(),
        }, indent=2) + "\n")
        os.replace(partial, status)
        time.sleep(60)
    print("WAN22_GENERATION_DONE")


if __name__ == "__main__":
    main()
