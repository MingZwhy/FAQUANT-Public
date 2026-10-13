#!/usr/bin/env python3
"""Paired comparison of two VBench runs of the same prompt suite and seeds.

Reads <run>/vbench/per_video.json of a reference and a candidate run (written
by scripts/wan22_vbench_eval.py), pairs videos by file name, and reports per
dimension the score gap (reference - candidate, in points), and for Quality,
Semantic and Total a paired bootstrap 95% interval: prompts are resampled
with their videos, and the same draw is used for both runs and for every
dimension, so the interval of the aggregate keeps the correlation between
dimensions. Per-video scores of the dimensions VBench scores per prompt
(object class, color, ...) are averaged the same way VBench does.

  python scripts/compare_wan22_vbench.py --reference RUN_A --candidate RUN_B --output gap.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from faquant.wan22_suite import DIMENSIONS
from faquant.workflows.wan22_vbench_eval import aggregate


def prompt_of(path: str) -> str:
    return re.sub(r"-\d+$", "", Path(path).stem)


def video_scores(per_video: dict) -> dict[str, dict[str, float]]:
    """dimension -> video file name -> score in [0, 1].

    Every dimension score is the mean of its per-video results, except that
    imaging quality reports per-video MUSIQ on a 0-100 scale. Color only lists
    the videos in which the object was detected, and so does its mean.
    """

    out: dict[str, dict[str, float]] = {}
    for dim, rows in per_video.items():
        scale = 100.0 if dim == "imaging_quality" else 1.0
        scores = {}
        for row in rows:
            value = row.get("video_results")
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                scores[Path(row["video_path"]).name] = float(value) / scale
        out[dim] = scores
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", default="", help="Read <run>/vbench<suffix>, e.g. _idx0.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sub = f"vbench{args.suffix}"
    ref_summary = json.loads((args.reference / sub / "summary.json").read_text())
    cand_summary = json.loads((args.candidate / sub / "summary.json").read_text())
    ref = video_scores(json.loads((args.reference / sub / "per_video.json").read_text()))
    cand = video_scores(json.loads((args.candidate / sub / "per_video.json").read_text()))

    report = {"reference": str(args.reference), "candidate": str(args.candidate), "dimensions": {}}
    per_prompt: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for dim in DIMENSIONS:
        r_dim, c_dim = ref.get(dim, {}), cand.get(dim, {})
        shared = sorted(set(r_dim) & set(c_dim))
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for name in shared:
            groups[prompt_of(name)].append((r_dim[name], c_dim[name]))
        per_prompt[dim] = groups
        report["dimensions"][dim] = {
            "reference": ref_summary["scores"].get(dim),
            "candidate": cand_summary["scores"].get(dim),
            "gap": (
                round(ref_summary["scores"][dim] - cand_summary["scores"][dim], 4)
                if dim in ref_summary["scores"] and dim in cand_summary["scores"] else None
            ),
            "paired_videos": len(shared),
            "per_video_scores": bool(shared),
        }
    for key in ("quality_score", "semantic_score", "total_score"):
        if key in ref_summary and key in cand_summary:
            report[key] = {
                "reference": ref_summary[key], "candidate": cand_summary[key],
                "gap": round(ref_summary[key] - cand_summary[key], 4),
            }

    # Bootstrap over prompts, only meaningful when every dimension has per-video scores.
    if all(per_prompt[d] for d in DIMENSIONS) and len(ref_summary["scores"]) == len(DIMENSIONS):
        rng = random.Random(args.seed)
        prompts = sorted({p for d in DIMENSIONS for p in per_prompt[d]})
        draws = defaultdict(list)
        base = {d: ref_summary["scores"][d] / 100 for d in DIMENSIONS}
        cand_base = {d: cand_summary["scores"][d] / 100 for d in DIMENSIONS}
        for _ in range(args.bootstrap):
            pick = [rng.choice(prompts) for _ in prompts]
            r_scores, c_scores = {}, {}
            for d in DIMENSIONS:
                pairs = [v for p in pick for v in per_prompt[d].get(p, [])]
                if not pairs:
                    r_scores[d], c_scores[d] = base[d], cand_base[d]
                    continue
                r_scores[d] = sum(a for a, _ in pairs) / len(pairs)
                c_scores[d] = sum(b for _, b in pairs) / len(pairs)
            ra, ca = aggregate(r_scores), aggregate(c_scores)
            for key in ra:
                draws[key].append(100 * (ra[key] - ca[key]))
        for key, values in draws.items():
            values.sort()
            report[key]["ci95"] = [
                round(values[int(0.025 * len(values))], 4),
                round(values[int(0.975 * len(values)) - 1], 4),
            ]
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
