#!/usr/bin/env python
"""Merge independently evaluated MMLU task shards with strict alignment checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GROUPS = {
    "mmlu_stem": (
        "mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy",
        "mmlu_college_biology", "mmlu_college_chemistry",
        "mmlu_college_computer_science", "mmlu_college_mathematics",
        "mmlu_college_physics", "mmlu_computer_security",
        "mmlu_conceptual_physics", "mmlu_electrical_engineering",
        "mmlu_elementary_mathematics", "mmlu_high_school_biology",
        "mmlu_high_school_chemistry", "mmlu_high_school_computer_science",
        "mmlu_high_school_mathematics", "mmlu_high_school_physics",
        "mmlu_high_school_statistics", "mmlu_machine_learning",
    ),
    "mmlu_other": (
        "mmlu_business_ethics", "mmlu_clinical_knowledge",
        "mmlu_college_medicine", "mmlu_global_facts", "mmlu_human_aging",
        "mmlu_management", "mmlu_marketing", "mmlu_medical_genetics",
        "mmlu_miscellaneous", "mmlu_nutrition",
        "mmlu_professional_accounting", "mmlu_professional_medicine",
        "mmlu_virology",
    ),
    "mmlu_social_sciences": (
        "mmlu_econometrics", "mmlu_high_school_geography",
        "mmlu_high_school_government_and_politics",
        "mmlu_high_school_macroeconomics", "mmlu_high_school_microeconomics",
        "mmlu_high_school_psychology", "mmlu_human_sexuality",
        "mmlu_professional_psychology", "mmlu_public_relations",
        "mmlu_security_studies", "mmlu_sociology", "mmlu_us_foreign_policy",
    ),
    "mmlu_humanities": (
        "mmlu_formal_logic", "mmlu_high_school_european_history",
        "mmlu_high_school_us_history", "mmlu_high_school_world_history",
        "mmlu_international_law", "mmlu_jurisprudence",
        "mmlu_logical_fallacies", "mmlu_moral_disputes",
        "mmlu_moral_scenarios", "mmlu_philosophy", "mmlu_prehistory",
        "mmlu_professional_law", "mmlu_world_religions",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        help=(
            "Full-MMLU output whose sample fingerprints every shard must match. "
            "Without it the shards are checked only against each other."
        ),
    )
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-linears",
        type=int,
        default=252,
        help=(
            "Quantized linears every shard must report: 7 per decoder layer. "
            "0 merges unquantized faquant-eval shards, which carry no QAD "
            "runtime receipt or weight hash."
        ),
    )
    return parser.parse_args()


def sample_fingerprint(sample: dict[str, object]) -> tuple[object, ...]:
    return (
        sample.get("doc_id"), sample.get("doc_hash"), sample.get("prompt_hash"),
        sample.get("target_hash"), sample.get("target"), sample.get("filter"),
    )


def aggregate(samples: dict[str, list[dict[str, object]]], tasks: tuple[str, ...]) -> dict[str, object]:
    scores = [float(row["acc"]) for task in tasks for row in samples[task]]
    if any(score not in (0.0, 1.0) for score in scores):
        raise RuntimeError("MMLU samples must have binary acc values")
    count = len(scores)
    accuracy = math.fsum(scores) / count
    stderr = math.sqrt(accuracy * (1.0 - accuracy) / max(count - 1, 1))
    return {
        "sample_len": count,
        "acc,none": accuracy,
        "acc_stderr,none": stderr,
        "sample_count": {"acc,none": count},
    }


def main() -> None:
    args = parse_args()
    reference = (
        json.loads(args.reference.read_text(encoding="utf-8"))
        if args.reference is not None else None
    )
    expected_tasks = (
        set(reference["samples"]) if reference is not None
        else {task for tasks in GROUPS.values() for task in tasks}
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.shard]
    samples: dict[str, list[dict[str, object]]] = {}
    results: dict[str, object] = {}
    merged_fields = {
        name: {} for name in (
            "configs", "versions", "n-shot", "higher_is_better", "n-samples"
        )
    }
    runtimes = []
    for path, payload in zip(args.shard, payloads, strict=True):
        if args.expected_linears == 0:
            if "faquant_qad_runtime" in payload:
                raise RuntimeError(f"{path}: quantized shard in an unquantized merge")
        else:
            runtime = payload.get("faquant_qad_runtime")
            if not isinstance(runtime, dict):
                raise RuntimeError(f"{path}: missing QAD runtime receipt")
            if (
                runtime.get("conversion_parity_max_abs") != 0.0
                or runtime.get("qat_linears_loaded") != args.expected_linears
                or runtime.get("inference_linears") != args.expected_linears
            ):
                raise RuntimeError(f"{path}: failed deployment parity audit")
            runtimes.append(runtime)
        for task, rows in payload.get("samples", {}).items():
            if task in samples:
                raise RuntimeError(f"duplicate task across shards: {task}")
            samples[task] = rows
            results[task] = payload["results"][task]
        for name, target in merged_fields.items():
            for key, value in payload.get(name, {}).items():
                if key in target and target[key] != value:
                    raise RuntimeError(f"conflicting {name} value for {key}")
                target[key] = value

    if set(samples) != expected_tasks:
        raise RuntimeError(
            f"task mismatch: missing={sorted(expected_tasks - set(samples))}, "
            f"extra={sorted(set(samples) - expected_tasks)}"
        )
    for task in sorted(expected_tasks):
        candidate_rows = samples[task]
        if reference is not None:
            reference_rows = reference["samples"][task]
            if len(reference_rows) != len(candidate_rows):
                raise RuntimeError(f"{task}: sample count mismatch")
            for left, right in zip(reference_rows, candidate_rows, strict=True):
                if sample_fingerprint(left) != sample_fingerprint(right):
                    raise RuntimeError(f"{task}: sample fingerprint mismatch")
        observed = math.fsum(float(row["acc"]) for row in candidate_rows) / len(candidate_rows)
        reported = float(results[task]["acc,none"])
        if not math.isclose(observed, reported, abs_tol=1e-12, rel_tol=0.0):
            raise RuntimeError(f"{task}: sample and reported accuracy differ")

    group_results = {}
    for group, tasks in GROUPS.items():
        values = aggregate(samples, tasks)
        alias = group.removeprefix("mmlu_").replace("_", " ")
        group_results[group] = {"name": group, "alias": alias, **values}
    all_tasks = tuple(sorted(expected_tasks))
    overall = aggregate(samples, all_tasks)
    group_results["mmlu"] = {"name": "mmlu", "alias": "mmlu", **overall}
    results.update(group_results)

    merged = dict(payloads[0])
    merged["results"] = results
    merged["groups"] = group_results
    merged["group_subtasks"] = {
        "mmlu": list(GROUPS), **{name: list(tasks) for name, tasks in GROUPS.items()}
    }
    merged["samples"] = samples
    merged.update(merged_fields)
    if args.expected_linears:
        hash_runtimes = [
            runtime for runtime in runtimes
            if not runtime.get("quantized_weight_hash_skipped")
        ]
        if len(hash_runtimes) != 1 or not hash_runtimes[0].get(
            "quantized_weight_sha256"
        ):
            raise RuntimeError(
                "exactly one shard must contain the all-linear weight hash"
            )
        runtime = dict(hash_runtimes[0])
        runtime["parallel_mmlu_shards"] = len(payloads)
        runtime["shard_outputs"] = [str(path) for path in args.shard]
        merged["faquant_qad_runtime"] = runtime
    merged.setdefault("config", {})["parallel_mmlu_shards"] = len(payloads)
    if overall["sample_len"] != 14_042:
        raise RuntimeError(f"expected 14042 MMLU samples, got {overall['sample_len']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(group_results, indent=2))


if __name__ == "__main__":
    main()
