# Wan2.2-T2V-A14B on VBench

HiF4 W4A4 on every block Linear of both DiT experts of Wan2.2-T2V-A14B, plus
HiF4 on both attention matrix multiplications (`QKᵀ` and `PV`), evaluated on
the complete VBench standard prompt suite at the official 1280×720, 81-frame
setting. With attention included, more than 99.99% of the DiT matrix-multiply
FLOPs run in 4-bit.

## Results

One sample per prompt (944 prompts, 16 dimensions), with the same prompt and
seed as BF16. Intervals are 95% from a bootstrap over prompts.

| Configuration | 4-bit share of DiT matmul FLOPs | Total | Loss vs BF16 (95% CI) |
|---|---:|---:|---:|
| BF16 | — | 82.20 | — |
| HiF4 Linear (RTN) | 27.7% | 81.57 | 0.63 [−0.04, 1.32] |
| HiSQRot4 Linear | 27.7% | 81.78 | 0.42 [−0.25, 1.14] |
| HiF4 Linear + QK/PV | > 99.99% | 78.48 | 3.72 [2.96, 4.60] |
| HiSQRot4 Linear + QK/PV | > 99.99% | 78.80 | 3.40 [2.52, 4.27] |
| **A**: HiF4 Linear + QK/PV + post-RoPE Q/K Hadamard + P-Reordering | > 99.99% | 81.60 | 0.60 [−0.03, 1.32] |
| A + Smooth-QK | > 99.99% | 81.27 | 0.93 [0.30, 1.65] |
| A + V/O per-head Hadamard | > 99.99% | 81.57 | 0.63 [0.12, 1.38] |
| **A + GPTQ** Linear weights | > 99.99% | 81.66 | 0.54 [−0.12, 1.19] |
| HiSQRot4 Linear + QK/PV + post-RoPE Q/K Hadamard + P-Reordering | > 99.99% | 81.66 | 0.54 [−0.19, 1.25] |

With two samples per prompt (1,888 videos per configuration), BF16 scores
82.13; HiF4 Linear 81.89, loss 0.25 [−0.23, 0.78]; HiSQRot4 Linear 81.95, loss
0.18 [−0.29, 0.73].

- Quantizing QK/PV directly costs more than 3 points, spread over all kinds of
  dimensions. The post-RoPE Q/K Hadamard with P-Reordering recovers it: on the
  same sample, A loses −0.03 [−0.49, 0.30] more than HiF4 on the Linears
  alone. The two transforms do the same for HiSQRot4 (2.86 [2.22, 3.60]
  better than HiSQRot4 + QK/PV).
- On top of A, neither Smooth-QK (+0.33 [−0.11, 0.75]) nor the V/O per-head
  Hadamard (+0.03 [−0.47, 0.52]) helps measurably, and HiSQRot4's Linear
  processing (SmoothQuant, rotation, clipping) is indistinguishable from plain
  HiF4 (0.07 [−0.60, 0.70] better than A) while adding an online rotation to
  every Linear input.
- The remaining loss of about 0.5–0.6 is on the Linear side. GPTQ lowers the
  per-layer output error on held-out prompts to 0.41 of RTN (median), but
  A + GPTQ is only 0.07 [−0.54, 0.74] better than A with one sample per
  prompt. GPTQ is purely offline.
- One sample per prompt gives intervals of about ±0.65, so the 1% line (0.82)
  lies inside the intervals of A, A + GPTQ, and HiSQRot4 with the attention
  transforms; resolving it needs more samples.
- Among single dimensions, aesthetic quality shows what HiSQRot4's Linear
  processing buys: over two samples it loses 1.08 against 1.83 for plain HiF4.

Machine-readable values, including all 16 dimensions and every paired
comparison: [`results/wan22_vbench.json`](../results/wan22_vbench.json).

## Protocol

### Generation

The settings match the VBench team's own sampling for the leaderboard entry
"Wan2.2-T2V-A14B (w/o prompt-optimized)" (Total 82.61, Quality 85.03,
Semantic 72.92):

| Item | Setting |
|---|---|
| Model | `Wan-AI/Wan2.2-T2V-A14B`, high-noise and low-noise DiT experts |
| Code | Wan2.2 commit `81e6735`, `WanT2V.generate` called unchanged |
| Resolution, frames, fps | 1280×720, 81, 16 |
| Sampling | UniPC, 40 steps, shift 12, guidance 3.0 (low noise) / 4.0 (high noise), boundary 0.875, default negative prompt |
| Prompt extension | none (the leaderboard's extended-prompt entry, Total 84.23, is not comparable) |
| Precision | DiT parameters in bf16 (`--convert_model_dtype` of the official single-GPU command), T5 bf16 |
| Video files | official `save_video` |

The prompt suite is `VBench_full_info.json` (946 prompts, 16 dimensions):
five samples per prompt and 25 for the 75 temporal-flickering prompts. Every
video has its own seed derived from the prompt and sample index
(`faquant.wan22_suite.video_seed`), independent of job order and of the number
of GPUs. Two prompts occur twice (for different dimensions) and map to one file
name, so the full suite has 6,220 videos. Quantized configurations are compared
on the first one or two samples of every prompt.

### Evaluation

VBench commit `45e79ec`, `evaluate.py --mode vbench_standard
--load_ckpt_from_local True`, one process per dimension. Aggregation follows
VBench's `scripts/constant.py`: each dimension is min/max normalized, dynamic
degree has weight 0.5, Quality is the weighted mean of seven quality
dimensions, Semantic of nine semantic dimensions, and
Total = (4 · Quality + Semantic) / 5. Recomputed from the 16 leaderboard
values this gives 85.03 / 72.92 / 82.61, identical to the leaderboard.

Details that change scores:

- Temporal flickering is evaluated on static videos only: `static_filter.py`
  (RAFT) keeps five of the 25 samples per prompt. Evaluating samples 0–4
  directly measures the flicker of moving videos. With fewer samples, the
  filter can only choose among those present.
- Imaging quality uses the default `longer` preprocessing of `evaluate.py`,
  which resizes the longer side to 512 (720p becomes 512×288) before MUSIQ;
  `shorter` gives different scores.
- Dynamic degree samples frames every `round(fps / 8)` and scales its
  threshold with resolution and frame count, so videos must be written at
  their true fps. Other frame counts or resolutions change dynamic degree,
  motion smoothness, and imaging quality.
- Prompt extension moves Semantic from 72.92 to 79.50 for the same model.
- In `custom_input` mode without `--prompt_file`, VBench takes the prompt from
  the file name.
- Standard mode finds videos by the extension of the first file in the
  directory, so the video directory may contain only MP4 files. The generator
  keeps metadata and partial files elsewhere and renames finished videos into
  place.
- The official download link of the RAFT checkpoint (`raft-things.pth`) is
  unreachable; two independent mirrors gave the same file (SHA-256
  `fcfa4125…`) with the original `module.*` key names.

### Pipeline checks

- **Evaluation.** The videos published by the VBench team for this entry
  (five per prompt, 4,720 videos, without the 25-sample temporal-flickering
  set) score 82.19 with this pipeline, 0.42 below the leaderboard. The
  difference sits in three dimensions computed from DINO/CLIP features:
  subject consistency (−0.41 of Total), background consistency (−0.19), and
  aesthetic quality (+0.33); the other 13 dimensions contribute −0.16
  together. On a 40-video probe, antialiased resizing (not the default of
  tensor `Resize` in current torchvision) moves aesthetic quality by −2.65
  and background consistency by +1.05, which closes those two gaps; subject
  consistency moves by only +0.22. These are fixed offsets of the evaluator
  and do not affect paired comparisons.
- **Generation.** BF16 sample 0 of this pipeline against sample 0 of the
  team's videos (same prompts, different seeds): 82.20 against 82.12,
  difference −0.07 [−1.21, 0.67].

Two samples per prompt narrow the paired interval from about ±0.7 to ±0.5.
With the same seed the quantized and BF16 videos still diverge (frame PSNR
about 16 dB), so pairing removes prompt-to-prompt variation but not seed
noise.

## Quantization paths

**HiSQRot4** (`--quant hisqrot4`). The
[GCC-HiFloat/HiSQRot4_Wan2.2](https://github.com/GCC-HiFloat/HiSQRot4_Wan2.2)
release (commit `84cdcf7`, alpha 0.9 Stage-2 artifacts) replaces the 400 block
Linears of each expert with W4A4 HiFloat4: SmoothQuant channel scaling, a
per-layer 1024-block Hadamard, clipping from a lookup table (per group of
denoising steps and CFG branch), HiF4 quantization of the input, and a GEMM
with prequantized weights. Six non-block Linears per expert (text and time
embeddings, output head), T5, the VAE, and attention stay unquantized.
`faquant.wan22_hisqrot4` changes only memory behaviour so that 720p, 81-frame
generation fits on one GPU: prequantized tensors move with `Module.to()`,
one bf16 copy of each weight is kept, and rows are processed in chunks. On
real layers of both experts, both CFG branches, and after GPU↔CPU round
trips, outputs are bit-identical to the unpatched wrapper. The artifact
signature hashes fp32 Linear dtypes and cannot match bf16 parameters, so its
check is disabled; coverage is enforced by the release's own assertion and a
400/400 count of wrapped Linears.

**Plain HiF4** (`--quant hif4`, `faquant.wan22_hif4`). The same 400 Linears
per expert without SmoothQuant, rotation, or clipping: three-level HiF4
(bit-identical to the GCC-HiFloat reference), weights rounded to nearest from
the bf16 parameters, inputs quantized dynamically per token in 64-value
groups. `--hif4-input-backend quant_cy` uses HiSQRot4's CUDA input kernel,
which differs from `faquant.hif4` only on values exactly halfway between 0 and
the smallest nonzero code (4 of 387 million values on a real layer).

**HiF4 QK/PV** (`--hif4-attention qkpv`, `faquant.wan22_attention`). At
1280×720×81 a DiT block spends 45.2 TFLOP in Linears, 117.1 TFLOP in the
self-attention `QKᵀ` and `PV`, and 0.8 TFLOP in cross-attention, so the
Linears alone are 27.7% of the matmul FLOPs. Attention is quantized as in the
Qwen attention simulator: Q, K, and V per token along `head_dim` in 64-value
HiF4 groups; keys in tiles of 128 with an FP32 online softmax; the
unnormalized `P = exp(score − running max)` in 64-key HiF4 groups before the
product with V; FP32 scores, maxima, denominator, and accumulator. The
denominator comes from the unquantized P, or from the quantized P with
`--hif4-pv-normalizer quantized` (P-Reordering). It is a Triton FlashAttention
kernel whose in-kernel HiF4 matches `faquant.hif4` bit for bit; on real Wan
self-attention (75,600 keys) its relative error to the PyTorch simulator is
4–7 × 10⁻⁴, against a quantization effect of 8–23%.

**Attention transforms**, each switchable:

- `--hif4-qk-rotation`: one 128-point Hadamard on Q and K after RoPE,
  computed in FP32 and rounded to bf16 once before quantization; `QKᵀ` is
  unchanged.
- `--hif4-qk-smooth`: Q multiplied and K divided by per-channel scales
  `s_c = Kmax_c^α / Qmax_c^(1−α)` (α = 0.5, max statistic, shared by the
  heads, normalized by the median, clamped to 32×), calibrated per expert and
  block with `scripts/calibrate_wan22_qk_smooth.py` on 16 calibration prompts
  with HiF4 Linears and BF16 attention. The calibrated scales vary only by
  1.2–2.1× within a block.
- `--hif4-vo-rotation`: a per-head Hadamard folded offline into the v output
  rows (with bias) and the o input columns before rounding; no runtime cost.
- `--hif4-attn-bf16`: keeps selected attention calls in BF16, for example
  `low:33,34,38;cross`.

**GPTQ** (`scripts/calibrate_wan22_gptq.py`, used with `--hif4-weights`).
Calibration runs prompts 0–7 of HiSQRot4's calibration set (disjoint from
VBench) through full generation on configuration A and accumulates `XᵀX` of
the HiF4-quantized input that each block Linear multiplies (q/k/v share one),
over all denoising steps of the expert and both CFG branches, from every
fourth token. GPTQ uses the Qwen settings: three-level HiF4 in 64-value
groups, damping 1%, column blocks of 128, no act-order and no scale search.
Cross-attention k and v keep RTN: their input is the text context, whose few
distinct tokens leave a rank-deficient Hessian (these two layers are about
0.5% of the FLOPs). Fitting on seven prompts and testing on the eighth, the
Hessian-weighted output error of GPTQ relative to RTN has median 0.41
(10th percentile 0.11, 90th 0.59) over 640 Linears, and no layer is worse than
RTN. The weights are exact HiF4 values stored in bf16, one file per expert,
with a manifest of the GPTQ settings, calibration prompts, SHA-256 values, and
error statistics; generation memory-maps them instead of rounding.

## Reproduce

Besides this package, the workflows need checkouts of
[Wan2.2](https://github.com/Wan-Video/Wan2.2) (`81e6735`, with its
requirements), [HiSQRot4_Wan2.2](https://github.com/GCC-HiFloat/HiSQRot4_Wan2.2)
(`84cdcf7`, for its `quant_cy` input kernel, its calibration prompts, and its
Stage-2 artifacts), and [VBench](https://github.com/Vchitect/VBench)
(`45e79ec`, with its evaluation checkpoints, usually in a separate Python
environment). The attention kernel uses Triton.

```bash
export WAN_CKPT=/path/to/Wan2.2-T2V-A14B
export WAN_ROOT=/path/to/Wan2.2
export HISQROT4_ROOT=/path/to/HiSQRot4_Wan2.2
export VBENCH_ROOT=/path/to/VBench
export VBENCH_PYTHON=/path/to/vbench-env/bin/python
export RUNS=/path/to/wan22-runs
```

Generate sample 0 of every prompt with one worker per GPU, then evaluate:

```bash
python scripts/run_wan22_generation.py \
  --run-dir "$RUNS/bf16" \
  --ckpt-dir "$WAN_CKPT" \
  --wan-root "$WAN_ROOT" \
  --vbench-root "$VBENCH_ROOT" \
  --gpus 0,1,2,3 \
  --indices 0

python scripts/wan22_vbench_eval.py \
  --run-dir "$RUNS/bf16" \
  --vbench-root "$VBENCH_ROOT" \
  --python "$VBENCH_PYTHON" \
  --gpus 0,1,2,3 \
  --indices 0
```

Several machines can share one run directory on a shared filesystem with
`--shard K/N`. Configuration A passes its options to every worker:

```bash
python scripts/run_wan22_generation.py \
  --run-dir "$RUNS/recipe_a" \
  --ckpt-dir "$WAN_CKPT" \
  --wan-root "$WAN_ROOT" \
  --hisqrot4-root "$HISQROT4_ROOT" \
  --vbench-root "$VBENCH_ROOT" \
  --gpus 0,1,2,3 \
  --indices 0 \
  --worker-args "--quant hif4 --hif4-levels 123 --hif4-input-backend quant_cy --hif4-attention qkpv --hif4-pv-normalizer quantized --hif4-qk-rotation"
```

The options of every configuration are listed under `worker_args` in
`results/wan22_vbench.json`. Evaluate as above, then compare with BF16:

```bash
python scripts/compare_wan22_vbench.py \
  --reference "$RUNS/bf16" \
  --candidate "$RUNS/recipe_a" \
  --suffix _idx0 \
  --output "$RUNS/gap_recipe_a_vs_bf16.json"
```

GPTQ weights. Collect Hessians with one job per expert, block range, and
calibration prompt (prompts 0–7, both experts, block ranges covering blocks
0–39), merge each expert and range over the prompts, solve, and assemble:

```bash
python scripts/calibrate_wan22_gptq.py collect \
  --ckpt-dir "$WAN_CKPT" --wan-root "$WAN_ROOT" --hisqrot4-root "$HISQROT4_ROOT" \
  --prompt 0 --expert high_noise_model --blocks 0:14 \
  --out "$RUNS/gptq/collect/high_noise_model_b00-14_p0.pt"

python scripts/calibrate_wan22_gptq.py merge \
  --inputs "$RUNS/gptq/collect/high_noise_model_b00-14_p*.pt" \
  --out "$RUNS/gptq/hessians"

python scripts/calibrate_wan22_gptq.py gptq \
  --ckpt-dir "$WAN_CKPT" --hessians "$RUNS/gptq/hessians" \
  --expert high_noise_model --blocks 0:2 \
  --out "$RUNS/gptq/parts/high_noise_model_b00-02.pt"

python scripts/calibrate_wan22_gptq.py assemble \
  --parts "$RUNS/gptq/parts" --hessians "$RUNS/gptq/hessians" \
  --out "$RUNS/gptq/weights"
```

A + GPTQ adds `--hif4-weights "$RUNS/gptq/weights"` to the worker options of
A. Smooth-QK scales come from `scripts/calibrate_wan22_qk_smooth.py collect`
(one job per calibration prompt) and `merge`, and are passed with
`--hif4-qk-smooth`.
