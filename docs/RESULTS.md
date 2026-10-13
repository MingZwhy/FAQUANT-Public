# Results

All losses below are absolute score differences:

```text
loss = BF16 score - candidate score
```

No answer-aware post-processing is used. Raw prompts, benchmark samples, and
model generations are intentionally not redistributed.

## Qwen3-8B

### MMLU

| Setting | BF16 | Candidate | Loss |
|---|---:|---:|---:|
| 0-shot, 57 subjects, 14,042 questions | 72.9597% | **71.9627%** | **0.9970 pp** |

### LongBench

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

## Qwen3-32B

PTQ with whole-layer BF16 protection, no QAD; see [Qwen3-32B](QWEN3_32B.md)
for the recipe and protocol and [Rotation](ROTATION.md) for the rotations.

### MMLU

0-shot, 57 subjects, 14,042 questions, no chat template, 16 fixed shards with
batch size 4. BF16: 80.8859%.

| HiF4 scale | Rotation | BF16 layers | MMLU | Loss | KL |
|---|---|---|---:|---:|---:|
| three-level | HiSQ1024 | none | 79.5756% | 1.3104 pp | 0.1067 |
| three-level | HiSQ1024 | 35, 41, 42, 45, 46 | 80.1239% | 0.7620 pp | 0.0818 |
| three-level | `rbhn_r64` | 39, 40, 42, 45, 46 | 80.0527% | 0.8332 pp | 0.0813 |
| two-level | HiSQ1024 | none | 79.2195% | 1.6664 pp | 0.1337 |
| two-level | HiSQ1024 | 39, 42, 43, 45, 46, 63 | 79.7963% | 1.0896 pp | 0.1000 |
| two-level | `rbh_r64` | 41, 42, 43, 44, 45, 46 | 79.8747% | 1.0113 pp | 0.1090 |

KL is KL(BF16 ‖ candidate) over the four answer choices, averaged over
questions. Per-group accuracies are in
[`results/qwen3_32b_mmlu_0shot.json`](../results/qwen3_32b_mmlu_0shot.json).

### LongBench

Same prefill-only protocol as Qwen3-8B. BF16: 44.9224. Intervals are 95% from
a paired bootstrap over samples within each task.

| Configuration | Macro | Loss (95% CI) | Single-doc QA | Multi-doc QA | Summarization | Few-shot | Synthetic | Code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| three-level, HiSQ1024, layers 35, 41, 42, 45, 46 | 45.1224 | −0.20 [−0.75, +0.33] | 0.58 | −0.33 | −0.02 | −1.11 | 0.67 | −1.34 |
| three-level, `rbhn_r64`, layers 39, 40, 42, 45, 46 | 44.9348 | −0.01 [−0.53, +0.52] | 0.45 | 0.75 | −0.06 | −1.80 | 1.50 | −1.06 |
| two-level, HiSQ1024, layers 39, 42, 43, 45, 46, 63 | 45.0224 | −0.10 [−0.62, +0.47] | −0.31 | −0.27 | 0.11 | −1.07 | 2.00 | −0.97 |
| two-level, `rbh_r64`, layers 41–46 | 45.1205 | −0.20 [−0.77, +0.36] | 0.74 | 0.53 | 0.01 | −2.23 | 0.83 | −1.41 |

Category columns are losses. Task-level values and category intervals are in
[`results/qwen3_32b_longbench_prefill_only.json`](../results/qwen3_32b_longbench_prefill_only.json).

## Wan2.2-T2V-A14B

Full VBench standard suite (944 prompts, 16 dimensions), sample 0 of every
prompt, paired with BF16 by prompt and seed; intervals are 95% from a bootstrap
over prompts. See [Wan2.2](WAN22.md) for the protocol and quantization paths.

| Configuration | Quality | Semantic | Total | Loss (95% CI) |
|---|---:|---:|---:|---:|
| BF16 | 84.55 | 72.81 | 82.20 | — |
| HiF4 Linear | 84.14 | 71.29 | 81.57 | 0.63 [−0.04, 1.32] |
| HiSQRot4 Linear | 84.32 | 71.64 | 81.78 | 0.42 [−0.25, 1.14] |
| HiF4 Linear + QK/PV | 81.48 | 66.47 | 78.48 | 3.72 [2.96, 4.60] |
| HiSQRot4 Linear + QK/PV | 81.54 | 67.85 | 78.80 | 3.40 [2.52, 4.27] |
| A: HiF4 Linear + QK/PV + Q/K Hadamard + P-Reordering | 84.19 | 71.21 | 81.60 | 0.60 [−0.03, 1.32] |
| A + Smooth-QK | 83.95 | 70.56 | 81.27 | 0.93 [0.30, 1.65] |
| A + V/O per-head Hadamard | 84.29 | 70.70 | 81.57 | 0.63 [0.12, 1.38] |
| A + GPTQ | 84.21 | 71.47 | 81.66 | 0.54 [−0.12, 1.19] |
| HiSQRot4 Linear + QK/PV + Q/K Hadamard + P-Reordering | 84.37 | 70.85 | 81.66 | 0.54 [−0.19, 1.25] |

Samples 0 and 1 of every prompt: BF16 82.13; HiF4 Linear 81.89, loss 0.25
[−0.23, 0.78]; HiSQRot4 Linear 81.95, loss 0.18 [−0.29, 0.73]. All 16
dimensions and every paired comparison are in
[`results/wan22_vbench.json`](../results/wan22_vbench.json).

## Reproducibility boundary

The published hashes identify the evaluated Qwen3-8B checkpoint artifact.
Given that artifact, model loading, protection metadata, and evaluation are
deterministic under the pinned stack. Retraining with the same seed is not
claimed to reproduce byte-identical weights. The Qwen3-32B builds use fixed
seeds and were rebuilt bit for bit on the same software and hardware.
