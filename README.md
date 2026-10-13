# FAQUANT-Public

[English](README.md) | [简体中文](README_zh-CN.md)

Reproduction code and aggregate results for HiF4 W4A4 quantization of
Qwen3-8B (PTQ, QAD, post-QAD layer protection), Qwen3-32B (PTQ with layer
protection), and Wan2.2-T2V-A14B (all DiT Linears and attention QK/PV,
evaluated on the full VBench suite).

This repository intentionally excludes model weights, checkpoints, datasets,
raw benchmark samples, generations, and private experiment history.

## Headline results

### Qwen3-8B

| Benchmark | BF16 | Candidate | Loss |
|---|---:|---:|---:|
| MMLU 0-shot, 14,042 questions | 72.9597% | **71.9627%** | **0.9970 pp** |
| LongBench v1, 21-task macro | 49.4352 | **48.3529** | **1.0824 pp** |

The LongBench setting quantizes the complete prompt prefill and first generated
token. Later decode tokens use BF16 while reusing the quantized-prefill KV
cache.

### Qwen3-32B

PTQ only, no QAD. BF16: MMLU 80.8859%, LongBench 44.9224.

| HiF4 scale | Linear input rotation | BF16 layers | MMLU loss | LongBench loss (95% CI) |
|---|---|---|---:|---:|
| three-level | HiSQ1024, 7 online rotations per layer | 35, 41, 42, 45, 46 | 0.7620 pp | −0.20 [−0.75, +0.33] |
| three-level | `rbhn_r64`, 4 online block Hadamards per layer | 39, 40, 42, 45, 46 | 0.8332 pp | −0.01 [−0.53, +0.52] |
| two-level | HiSQ1024 | 39, 42, 43, 45, 46, 63 | 1.0896 pp | −0.10 [−0.62, +0.47] |
| two-level | `rbh_r64`, 4 online block Hadamards per layer | 41, 42, 43, 44, 45, 46 | 1.0113 pp | −0.20 [−0.77, +0.36] |

### Wan2.2-T2V-A14B

Full VBench standard suite, one sample per prompt, paired with BF16 by prompt
and seed. More than 99.99% of the DiT matmul FLOPs run in 4-bit.

| Configuration | VBench Total | Loss (95% CI) |
|---|---:|---:|
| BF16 | 82.20 | — |
| HiF4 Linear + QK/PV | 78.48 | 3.72 [2.96, 4.60] |
| A: + post-RoPE Q/K Hadamard + P-Reordering | 81.60 | 0.60 [−0.03, 1.32] |
| A + GPTQ | 81.66 | 0.54 [−0.12, 1.19] |

See [Method](docs/METHOD.md), [Reproduction](docs/REPRODUCE.md),
[Results](docs/RESULTS.md), [Qwen3-32B](docs/QWEN3_32B.md),
[Rotation](docs/ROTATION.md), and [Wan2.2](docs/WAN22.md).

## Qwen3-8B candidate

- Base model: `Qwen/Qwen3-8B`
- Format: HiF4 W4A4, group size 64
- PTQ: GPTQ + HiSQ1024 + value-head rotation + Smooth-QK
- Attention: post-RoPE Q/K Hadamard + tiled HiF4 QK/PV + P-Reordering
- QAD: 250 exact-attention steps + 125 deployed-attention steps
- Post-QAD protection: layers 16, 17, and 18 restored to BF16
- Quantized projections: 231 / 252
- QK MXFP8 layers: none

The evaluated checkpoint is identified by SHA-256 values in
[`results/final_metrics.json`](results/final_metrics.json). The weights are
distributed separately on
[Hugging Face](https://huggingface.co/MingZwhy/qwen3-8b-hif4-l16-17-18).

## Installation

Python 3.11 is recommended. Install PyTorch for your CUDA version first, then:

```bash
python -m pip install -e ".[test]"
```

An example environment for 80 GB GPUs is provided:

```bash
bash scripts/setup_env.sh
```

FlashAttention is optional for setup and BF16 reference runs. The quantized
QK/PV simulator for Qwen is implemented in PyTorch. The Wan2.2 workflows also
need checkouts of Wan2.2, HiSQRot4_Wan2.2, and VBench; see
[Wan2.2](docs/WAN22.md#reproduce).

## Reproduction outline

```text
Qwen3-8B BF16
  -> HiF4 PTQ initialization
  -> Smooth-QK calibration
  -> QAD: exact 250 + deployed 125
  -> materialize BF16 protection for layers 16/17/18
  -> MMLU and LongBench evaluation

Qwen3-32B BF16
  -> HiF4 PTQ with protected layers excluded (one GPU, layer offload)
  -> Smooth-QK calibration
  -> MMLU (16 shards) and LongBench evaluation

Wan2.2-T2V-A14B
  -> optional GPTQ weights and Smooth-QK scales
  -> VBench video generation, one worker per GPU
  -> VBench evaluation and paired comparison with BF16
```

All commands accept explicit model, data, checkpoint, scale, and output paths.
No machine-specific path is assumed.

## Repository layout

```text
configs/   final Qwen3-8B recipe, Qwen3-32B MMLU shards, environment example
docs/      method, reproduction, results, Qwen3-32B, rotation, Wan2.2, citations
results/   compact aggregate metrics only
scripts/   thin command-line entry points
src/       Qwen3/Wan2.2 HiF4 implementation and workflows
tests/     CPU/synthetic numerical checks
```

## Reproducibility scope

The checkpoint hashes identify the evaluated Qwen3-8B artifact. Given that
artifact, the materialized protection and evaluation path are reproducible.
Distributed QAD retraining is not claimed to produce byte-identical weights
from the same seed. The Qwen3-32B builds use fixed seeds and were reproduced
bit for bit on the same software and hardware. Generated Wan2.2 videos depend
on the GPU kernels, so a quantized run should be compared with a BF16 run
generated in the same environment.

## License

FAQUANT-Public is licensed under Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream licenses and
citations.
