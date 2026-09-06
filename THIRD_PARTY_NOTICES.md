# Third-party notices

FAQUANT-Public is distributed under Apache-2.0. Portions of the implementation,
task definitions, and evaluation workflow are derived from or interoperate with
the projects below.

## GCC-HiFloat / HiFloat4-Quantization_Library

- Repository: https://github.com/GCC-HiFloat/HiFloat4-Quantization_Library
- Reference commit: `6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4`
- License: Apache License 2.0

`faquant.hif4` is a PyTorch implementation aligned to the public HiF4
reference semantics.

## EleutherAI lm-evaluation-harness

- Repository: https://github.com/EleutherAI/lm-evaluation-harness
- Reference commit: `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`
- License: MIT
- Copyright: EleutherAI

The MMLU integration and model-evaluation interface use lm-evaluation-harness.

## THUDM LongBench

- Repository: https://github.com/THUDM/LongBench
- Reference commit: `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
- License: MIT
- Copyright: THU-KEG and Zhipu AI

The bundled LongBench task prompts, generation lengths, and metric functions
follow this reference.

## FlashAttention

- Repository: https://github.com/Dao-AILab/flash-attention
- License: BSD 3-Clause

FlashAttention is an optional runtime dependency for native BF16 attention.

## Qwen3

- Model: https://huggingface.co/Qwen/Qwen3-8B
- Project: https://github.com/QwenLM/Qwen3

Users must comply with the model's license and usage terms. Model weights are
not distributed by this repository.

## RedPajama

- Dataset: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample

Training data are downloaded by users and are not redistributed here.

## Academic references

- GPTQ: Frantar et al., “GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers,” arXiv:2210.17323.
- MMLU: Hendrycks et al., “Measuring Massive Multitask Language
  Understanding,” ICLR 2021.
- LongBench: Bai et al., “LongBench: A Bilingual, Multitask Benchmark for
  Long Context Understanding,” ACL 2024.
- QuaRot: Ashkboos et al., “QuaRot: Outlier-Free 4-Bit Inference in Rotated
  LLMs,” arXiv:2404.00456.
