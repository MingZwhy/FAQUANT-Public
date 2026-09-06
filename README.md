# FAQUANT-Public

[English](README.md) | [简体中文](README_zh-CN.md)

Minimal reproduction code and aggregate results for Qwen3-8B HiF4 W4A4 PTQ,
QAD, post-QAD layer protection, and prefill-only LongBench evaluation.

This repository intentionally excludes model weights, checkpoints, datasets,
raw benchmark samples, generations, and private experiment history.

## Headline results

| Benchmark | BF16 | Candidate | Loss |
|---|---:|---:|---:|
| MMLU 0-shot, 14,042 questions | 72.9597% | **71.9627%** | **0.9970 pp** |
| LongBench v1, 21-task macro | 49.4352 | **48.3529** | **1.0824 pp** |

The LongBench setting quantizes the complete prompt prefill and first generated
token. Later decode tokens use BF16 while reusing the quantized-prefill KV
cache.

See [Method](docs/METHOD.md), [Reproduction](docs/REPRODUCE.md), and
[Results](docs/RESULTS.md).

## Candidate

- Base model: `Qwen/Qwen3-8B`
- Format: HiF4 W4A4, group size 64
- PTQ: GPTQ + HiSQ1024 + value-head rotation + Smooth-QK
- Attention: post-RoPE Q/K Hadamard + tiled HiF4 QK/PV + P-Reordering
- QAD: 250 exact-attention steps + 125 deployed-attention steps
- Post-QAD protection: layers 16, 17, and 18 restored to BF16
- Quantized projections: 231 / 252
- QK MXFP8 layers: none

The evaluated checkpoint is identified by SHA-256 values in
[`results/final_metrics.json`](results/final_metrics.json). The weights are not
part of this initial release.

## Installation

Python 3.11 is recommended. Install PyTorch for your CUDA version first, then:

```bash
python -m pip install -e ".[test]"
```

An A800 80GB example is provided:

```bash
bash scripts/setup_a800.sh
```

FlashAttention is optional for setup and BF16 reference runs. The quantized
QK/PV simulator is implemented in PyTorch.

## Reproduction outline

```text
Qwen3-8B BF16
  -> HiF4 PTQ initialization
  -> Smooth-QK calibration
  -> QAD: exact 250 + deployed 125
  -> materialize BF16 protection for layers 16/17/18
  -> MMLU and LongBench evaluation
```

All commands accept explicit model, data, checkpoint, scale, and output paths.
No machine-specific path is assumed.

## Repository layout

```text
configs/   final recipe
docs/      method, reproduction, results, citations
results/   compact aggregate metrics only
scripts/   thin command-line entry points
src/       Qwen/HiF4 implementation and workflows
tests/     CPU/synthetic numerical checks
```

## Reproducibility scope

The checkpoint hashes identify the evaluated artifact. Given that artifact,
the materialized protection and evaluation path are reproducible. Distributed
QAD retraining is not claimed to produce byte-identical weights from the same
seed.

## License

FAQUANT-Public is licensed under Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream licenses and
citations.
