import json
import sys

import pytest

from faquant.workflows import summarize_longbench as summarize


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _sample(doc_id: int, score: float) -> dict:
    return {
        "doc_id": doc_id,
        "doc_hash": f"d{doc_id}",
        "prompt_hash": f"p{doc_id}",
        "target_hash": f"t{doc_id}",
        "score": score,
    }


def test_summary_reports_loss_and_paired_bootstrap_interval(tmp_path, monkeypatch) -> None:
    shards = 2
    for task, count in summarize.TASKS.items():
        key = f"faquant_longbench_{task}"
        _write(
            tmp_path / "bf16" / f"{key}_full_bf16_native.json",
            {"results": {key: {"score,none": 0.5}}, "samples": {key: [_sample(i, 0.5) for i in range(count)]}},
        )
        for index in range(shards):
            start, end = index * count // shards, (index + 1) * count // shards
            candidate = [_sample(i, 0.5 if i % 2 else 0.4) for i in range(start, end)]
            _write(
                tmp_path / "run" / "shards" / f"{key}_shard_{index}_of_{shards}.json",
                {
                    "results": {key: {"score,none": sum(s["score"] for s in candidate) / len(candidate)}},
                    "samples": {key: candidate},
                    "faquant_qad_runtime": {
                        "quantization_scope": "prefill_only",
                        "decode_reuses_quantized_prefill_kv_cache": True,
                        "checkpoint": "checkpoints/example",
                    },
                },
            )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "summarize_longbench", "--result-dir", str(tmp_path / "run"), "--bf16-dir", str(tmp_path / "bf16"),
        "--num-shards", str(shards), "--output", str(output), "--bootstrap", "200",
    ])
    summarize.main()

    summary = json.loads(output.read_text())
    assert summary["loss_pp"] == pytest.approx(5.0)
    low, high = summary["loss_ci95"]
    assert low <= 5.0 <= high and high - low < 2.0
    for category in summary["categories"].values():
        assert category["loss_pp"] == pytest.approx(5.0)
        assert category["loss_ci95"][0] <= 5.0 <= category["loss_ci95"][1]
