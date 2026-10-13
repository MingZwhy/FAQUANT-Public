# Qwen3-32B

HiF4 W4A4 post-training quantization of Qwen3-32B with the Qwen3-8B PTQ
recipe and whole-layer BF16 protection, without QAD. Two HiF4 formats are
covered: the standard three-level scale and a two-level variant. For both, the
online per-linear HiSQ rotation can be replaced by a partially offline block
rotation with four instead of seven online transforms per layer; see
[Rotation](ROTATION.md).

## Results

BF16: MMLU 80.8859%, LongBench 44.9224.

| HiF4 scale | Rotation | BF16 layers | GPTQ | MMLU | Loss | KL | LongBench loss (95% CI) |
|---|---|---|---|---:|---:|---:|---:|
| three-level | HiSQ1024 | none | damp 0.03 | 79.5756% | 1.3104 pp | 0.1067 | — |
| three-level | HiSQ1024 | 35, 41, 42, 45, 46 | damp 0.03 | 80.1239% | 0.7620 pp | 0.0818 | −0.20 [−0.75, +0.33] |
| three-level | `rbhn_r64` | 39, 40, 42, 45, 46 | damp 0.03 | 80.0527% | 0.8332 pp | 0.0813 | −0.01 [−0.53, +0.52] |
| two-level | HiSQ1024 | none | damp 0.3, act-order | 79.2195% | 1.6664 pp | 0.1337 | — |
| two-level | HiSQ1024 | 39, 42, 43, 45, 46, 63 | damp 0.3, act-order | 79.7963% | 1.0896 pp | 0.1000 | −0.10 [−0.62, +0.47] |
| two-level | `rbh_r64` | 41, 42, 43, 44, 45, 46 | damp 0.3, act-order | 79.8747% | 1.0113 pp | 0.1090 | −0.20 [−0.77, +0.36] |

- Loss is BF16 minus quantized; a negative LongBench loss means the quantized
  model scored higher. KL is KL(BF16 ‖ quantized) between the softmax
  distributions over the four MMLU answer log-likelihoods, averaged over
  questions.
- No configuration has a measurable LongBench loss: every interval contains 0.
- All builds use seed 0 for the model, the GPTQ calibration data, and the
  Smooth-QK calibration. Rebuilding the four protected configurations from
  scratch on the same software and hardware reproduced the checkpoints, the
  Smooth-QK scales, and every MMLU answer and log-likelihood bit for bit.
- Each protected layer set is the best of a search; see the selection notes
  below before comparing rows.

Machine-readable values:
[`results/qwen3_32b_mmlu_0shot.json`](../results/qwen3_32b_mmlu_0shot.json)
and
[`results/qwen3_32b_longbench_prefill_only.json`](../results/qwen3_32b_longbench_prefill_only.json).

## Recipe

- Model: `Qwen/Qwen3-32B`, revision
  `9216db5781bf21249d130ec9da846c4624c16137`; 64 decoder layers, 448 projection
  Linears.
- HiF4 W4A4 on the projections, 64-value groups; V/O per-head Hadamard folded
  into the weights; per-linear HiSQ1024 or a partially offline block rotation
  on the Linear inputs; post-RoPE Q/K Hadamard; Smooth-QK (alpha 0.5, max
  statistic, 8 × 4096 tokens, seed 0, recalibrated for every build); tiled
  HiF4 QK/PV with P-Reordering (`quantized_same` normalizer).
- GPTQ on WikiText-2, 128 × 2048 tokens, calibration seed 0, without
  sequential groups. Three-level: damping 0.03. Two-level: damping 0.3 and
  act-order within each 64-value group.
- A protected layer keeps its seven projections in BF16, and its attention
  core (QK/PV quantization, Smooth-QK, Q/K rotation) is not quantized either.
  Protected layers are excluded when the checkpoint is built, so GPTQ re-solves
  every later layer around them.

Builds fit on one GPU: `--device-map offload` keeps the model on the CPU and
moves one decoder layer at a time to the GPU, bit-identical to a multi-GPU
build. `--inference-load` loads the step-0 weights straight into inference
Linears for calibration and evaluation.

## Evaluation

**MMLU.** 57 subjects, 14,042 questions, 0-shot, without chat template,
batch size 4, evaluated as 16 fixed task shards
([`configs/qwen3_32b_mmlu_shards_16.json`](../configs/qwen3_32b_mmlu_shards_16.json))
with one GPU per shard. BF16 uses the same shards. Batch composition alone
moves BF16 by 0.13 pp (batch 16 on two GPUs gives 80.7577%), so losses are
only compared within one protocol.

**Noise.** Builds that differ only in GPTQ rounding differ by 0.3–0.5 pp in
accuracy, while KL is stable to about 0.003 (its paired standard error). KL is
the primary signal when comparing recipes, and accuracy differences below
about 0.5 pp are not resolved by single builds. Paired statistics are an exact
McNemar test on the questions that flip and paired means of KL and of the
probability of the correct answer.

**LongBench.** The prefill-only protocol of Qwen3-8B
([Method](METHOD.md#longbench)): the official 21 tasks and 4,750 samples,
official prompts, generation lengths and metrics, middle truncation to
40,960 tokens, and greedy decoding. The quantized model processes the prompt
and selects the first token; BF16 weights decode the rest from the quantized
prefill KV cache. Intervals come from a paired bootstrap that resamples samples
within each task. Each worker splits the layers over several GPUs
(`--gpus-per-job`), and the prompt is prefilled in 8,192-token chunks, which
gives the same logits and KV cache bit for bit as a single pass.

The two code tasks sit near their floor: the official code prompts use no chat
template, and Qwen3-32B in BF16 starts 914 of 1,000 code answers with
reasoning text instead of code. The two HiSQ1024 builds do so slightly less
often (871 and 903), which is why they score higher on code.

## Layer protection (three-level)

- Quantizing one layer at a time with all others in BF16 locates the
  sensitive band at layers 37–47; their KL is 4–8 times that of other layers
  (largest: 46, 42, 43, 45, 44).
- Protecting a single layer changes accuracy by at most 0.2 pp, less than the
  noise between builds, so single-layer screens cannot rank layers.
- 80 five-layer sets were built and evaluated in four rounds (balanced designs
  over layers 38–46 and 61–63, regression-guided neighbours, local search):
  median loss 1.21 pp, best 0.7620 pp (layers 35, 41, 42, 45, 46), second
  0.94 pp. The best value carries a selection effect; the typical gain from
  five protected layers is about 0.3 pp over a typical unprotected build.
- Unprotected, about 25 GPTQ configurations were built. Damping 0.03 gave
  1.3104 pp with calibration seed 0 and 1.62–1.65 pp with seeds 1 and 2: seed
  variation is as large as the differences between configurations, and all
  reported builds use fixed seeds.
- Removing HiSQ1024 from the best protected build costs 0.83 pp (McNemar
  p = 0.0007) and raises KL by 25%.

## Two-level HiF4

The three-level format stores, per 64 values, an E6M2 base scale and two
levels of 1-bit exponents, one per 8 values and one per 4 values (4.5 bits per
value). The two-level variant `12` drops the per-4-value exponent
(4.25 bits per value). With one exponent level left, the largest multiplier is
2 instead of 4, so the base scale is `max / 3.5`; the remaining level uses the
last-level rule of the reference encoder. `123` matches the GCC-HiFloat
reference encoder bit for bit. A variant `13`, which keeps the per-4-value
exponent instead, served as a control.

| Relative squared error | three-level | `12` | `13` |
|---|---:|---:|---:|
| Gaussian | 0.00689 | 0.00894 | 0.00702 |
| Student-t, 3 degrees of freedom | 0.00883 | 0.01559 | 0.01294 |
| 1% outliers × 30 | 0.01009 | 0.02020 | 0.01912 |

The variant is selected by environment variables that every build,
calibration, and evaluation process reads:

| Variable | Applies to | Default |
|---|---|---|
| `FAQUANT_HIF4_SCALE_LEVELS` | weights | `123` |
| `FAQUANT_HIF4_ACT_SCALE_LEVELS` | Linear inputs and attention operands | weights |
| `FAQUANT_HIF4_ATTN_SCALE_LEVELS` | attention Q/K/P/V only | Linear inputs |

The build records the variants in `qad_init.json`, and loading a checkpoint
under a different setting raises an error.

Findings for `12`:

- Building the same protected set in both formats (17 sets) costs 0.35 pp of
  MMLU (median) and 22% more KL, on every set.
- The loss is on the activation side: two-level activations with three-level
  weights are as bad as the full two-level model, and Linear inputs and
  attention operands contribute about half each.
- Damping 0.3 and act-order within the group are needed (they lower the
  unprotected KL by 0.007 and 0.004). HiF4 scale search, other Smooth-QK
  settings, sequential groups, C4 or RedPajama calibration text, 512
  calibration samples, 4,096-token calibration sequences, and calibration
  through the deployed attention did not help.
- In attention, almost all of the loss is in P: restoring only P to three
  levels recovers three quarters of the attention part, and Q, K, or V alone
  recover nothing. Even all four attention operands in three-level format
  lower KL by only 0.007 (about 0.1 pp).
- Protection has to be re-selected: the best three-level set gives 1.5738 pp
  in two-level format. Five layers (17 sets): best 1.1466 pp (layers 39, 42,
  43, 45, 46), median 1.43. Six layers (36 sets, about 10% of the layers):
  best 1.0896 pp, median 1.37. Against the unprotected three-level build,
  the median KL change improves from −0.0001 (five layers) to −0.0075 (six
  layers), while accuracy barely moves.

`13` loses only in the attention operands; with Q/K/P/V kept in three-level
format its KL equals the three-level model's.

## Reproduce

```bash
export MODEL_DIR=/path/to/Qwen3-32B
export DATA_DIR=/path/to/faquant-data       # step 1 of docs/REPRODUCE.md
export OUTPUT_DIR=/path/to/qwen3-32b-output
```

The examples build the three-level `rbhn_r64` configuration. For the other
rows, change `--rotation` (`hisq1024`, `rbhn_r64`, `rbh_r64`) and the layer
list, and use the same `--rotation` in every step. For two-level rows, export
the format in every step and use the two-level GPTQ settings:

```bash
export FAQUANT_HIF4_SCALE_LEVELS=12 FAQUANT_HIF4_ACT_SCALE_LEVELS=12 FAQUANT_HIF4_ATTN_SCALE_LEVELS=12
# build: --gptq-damp 0.3 --gptq-act-order-within-group
```

Build the checkpoint on one GPU. The validation file is used only for the
step-0 parity check, which reads its first record:

```bash
CKPT="$OUTPUT_DIR/rbhn_r64_l39-40-42-45-46"
python scripts/prepare_ptq.py \
  --model "$MODEL_DIR" \
  --output-dir "$CKPT" \
  --rotation rbhn_r64 \
  --value-head-rotation \
  --seed 0 \
  --gptq-calibration-seed 0 \
  --gptq-damp 0.03 \
  --quant-exempt-layers 39,40,42,45,46 \
  --validation-data "$DATA_DIR/qad_redpajama/validation.jsonl" \
  --device cuda:0 \
  --device-map offload
```

Calibrate Smooth-QK:

```bash
faquant-calibrate-smooth-qk \
  --model "$MODEL_DIR" \
  --checkpoint "$CKPT" \
  --rotation rbhn_r64 \
  --inference-load \
  --device cuda:0 \
  --sequence-length 4096 \
  --samples 8 \
  --alphas 0.5 \
  --statistic max \
  --seed 0 \
  --output "$CKPT/smooth_qk_report.json" \
  --scales-output "$CKPT/smooth_qk.pt" \
  --scales-alpha 0.5
```

Evaluate MMLU as 16 shards, one GPU each (all but one shard skip the weight
hash), and merge them. `--expected-linears` is 448 minus 7 per protected
layer:

```bash
for i in $(seq 0 15); do
  TASKS=$(python -c "import json; print(','.join(json.load(open('configs/qwen3_32b_mmlu_shards_16.json'))[$i]))")
  CUDA_VISIBLE_DEVICES=$i python scripts/eval_mmlu.py \
    --model "$MODEL_DIR" \
    --checkpoint "$CKPT" \
    --rotation rbhn_r64 \
    --inference-load \
    --tasks "$TASKS" \
    --num-fewshot 0 \
    --batch-size 4 \
    --qk-matmul-quant \
    --pv-matmul-quant \
    --post-rope-qk-rotation \
    --pv-normalizer-mode quantized_same \
    --qk-smooth-scales "$CKPT/smooth_qk.pt" \
    --attention-kernel simulated \
    $( [ "$i" -gt 0 ] && echo --skip-weight-hash ) \
    --output "$CKPT/mmlu_$i.json" &
done
wait

python scripts/merge_mmlu_shards.py \
  $(for i in $(seq 0 15); do echo --shard "$CKPT/mmlu_$i.json"; done) \
  --expected-linears 413 \
  --output "$CKPT/mmlu.json"
```

For BF16, run `faquant-eval-bf16 --tasks "$TASKS" --num-fewshot 0
--batch-size 4` on the same shards and merge them with
`--expected-linears 0`. Passing a merged BF16 file as `--reference` checks that
every quantized shard saw identical questions.

LongBench BF16 reference, one file per task. Use the chat template for all
tasks except TREC, TriviaQA, SAMSum, LSHT, LCC, and RepoBench-P:

```bash
LM_EVAL_TRUNCATION_MODE=middle CUDA_VISIBLE_DEVICES=0,1 faquant-eval-longbench \
  --bf16-reference \
  --model "$MODEL_DIR" \
  --rotation none \
  --layer-devices 2 \
  --tasks faquant_longbench_qasper \
  --batch-size 1 \
  --max-length 40960 \
  --apply-chat-template \
  --output "$OUTPUT_DIR/longbench_bf16/faquant_longbench_qasper_full_bf16_native.json"
```

Quantized model, three GPUs per worker:

```bash
python scripts/run_longbench.py \
  --model "$MODEL_DIR" \
  --checkpoint "$CKPT" \
  --scales "$CKPT/smooth_qk.pt" \
  --rotation rbhn_r64 \
  --inference-load \
  --expected-exempt-layers 39,40,42,45,46 \
  --gpus 0,1,2,3,4,5 \
  --gpus-per-job 3 \
  --prefill-chunk-tokens 8192 \
  --num-shards 4 \
  --truncation-mode middle \
  --result-dir "$CKPT/longbench"
# two-level checkpoints: add --hif4-scale-levels 12

python scripts/summarize_longbench.py \
  --result-dir "$CKPT/longbench" \
  --bf16-dir "$OUTPUT_DIR/longbench_bf16" \
  --num-shards 4 \
  --output "$CKPT/longbench_summary.json"
```
