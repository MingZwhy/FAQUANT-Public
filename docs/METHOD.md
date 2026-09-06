# Method

## 1. Target arithmetic

The target is Qwen3-8B with:

- HiF4 W4A4 for transformer projection matrices;
- 64-value weight and activation groups;
- tiled fake-HiF4 QK and PV matrix multiplications;
- BF16 accumulation in the surrounding model and FP32 online-softmax state.

The implementation follows the GCC-HiFloat HiF4 S1P2 reference: one E6M2 base
scale per 64 values plus two levels of one-bit micro-exponents.

## 2. PTQ initialization

PTQ is applied before QAD:

1. Fold a per-head Hadamard into V/O weights.
2. Apply deterministic per-linear HiSQ rotations with block size 1024.
3. Collect WikiText-2 calibration activations (128 sequences × 2048 tokens).
4. Run GPTQ with 1% base damping, 3% damping for `o_proj` and `down_proj`, and
   layer-internal sequential groups.
5. Calibrate frozen Smooth-QK scales.

The deployed attention geometry is:

```text
RoPE
  -> Smooth-QK
  -> shared Q/K Hadamard
  -> tiled HiF4 QK
  -> online softmax
  -> P-Reordering
  -> tiled HiF4 PV
```

## 3. QAD

The student begins from the PTQ checkpoint with fixed HiF4 metadata. Only the
252 latent projection weights are trainable. The BF16 teacher is frozen.

The loss combines:

- task cross-entropy, weight 0.05;
- entropy-aware KL distillation, weight 2.0;
- layer-adaptive feature distillation, weight 0.5 over three selected layers.

Training uses a global batch of 256, learning rate `4e-5`, cosine horizon 1250,
warmup ratio 0.0104, and seed 3407:

1. 250 steps with exact attention;
2. 125 steps with deployed quantized attention.

The selected artifact is step 375. The same nominal seed is not guaranteed to
reproduce byte-identical training trajectories; checkpoint hashes are the
artifact identity.

## 4. Post-QAD protection

After QAD, layers 16, 17, and 18 are restored from the original BF16 model:

- all seven projections in each layer are BF16;
- HiSQ, Smooth-QK, post-RoPE Q/K Hadamard, and QK/PV quantization are disabled
  in those layers;
- no additional optimization is performed.

The materialized checkpoint records `quant_exempt_layers: [16, 17, 18]`.
It contains 231 quantized projection modules and 21 BF16 projections.

## 5. Evaluation

### MMLU

MMLU is evaluated 0-shot over all 57 subjects with the deployed quantized
attention path. Since it is a log-likelihood benchmark, all relevant model
computation is prefill.

### LongBench

LongBench uses the official 21-task main benchmark, official prompts and
generation lengths, greedy decoding, and middle truncation at 40,960 tokens.

Only prefill is quantized:

1. the quantized model processes the full prompt and selects token 1;
2. its transformed K/V cache is retained;
3. a basis-compatible BF16 model generates tokens 2 onward using that cache.

The BF16 decoder keeps mathematically equivalent basis transforms so that its
queries and outputs are compatible with the quantized-prefill cache, but it
does not apply W4/A4 or QK/PV quantization.
