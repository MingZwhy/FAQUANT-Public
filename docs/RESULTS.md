# Results

All losses below are absolute score differences:

```text
loss = BF16 score - candidate score
```

No answer-aware post-processing is used. Raw prompts, benchmark samples, and
model generations are intentionally not redistributed.

## MMLU

| Setting | BF16 | Candidate | Loss |
|---|---:|---:|---:|
| 0-shot, 57 subjects, 14,042 questions | 72.9597% | **71.9627%** | **0.9970 pp** |

## LongBench

The candidate quantizes the full-prompt prefill and first generated token.
Subsequent autoregressive decoding is BF16 and reuses the KV cache produced by
the quantized prefill.

| Setting | BF16 | Candidate | Loss |
|---|---:|---:|---:|
| Official 21-task macro, 4,750 samples | 49.4352 | **48.3529** | **1.0824 pp** |

| Category | BF16 | Candidate | Loss |
|---|---:|---:|---:|
| Single-document QA | 47.8000 | 45.3800 | 2.4200 |
| Multi-document QA | 41.4025 | 39.0325 | 2.3700 |
| Summarization | 24.1200 | 23.8575 | 0.2625 |
| Few-shot learning | 63.5100 | 63.5725 | -0.0625 |
| Synthetic | 66.5000 | 66.0000 | 0.5000 |
| Code | 65.6550 | 65.0200 | 0.6350 |

Machine-readable task-level values are in
[`results/longbench_prefill_only.json`](../results/longbench_prefill_only.json).

## Reproducibility boundary

The published hashes identify the evaluated checkpoint artifact. Given that
artifact, model loading, protection metadata, and evaluation are deterministic
under the pinned stack. Retraining with the same seed is not claimed to
reproduce byte-identical weights.
