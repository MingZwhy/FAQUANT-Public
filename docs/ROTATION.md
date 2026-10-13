# Rotations for HiF4 Linear inputs

HiF4 quantizes every Linear input in 64-value groups. Orthogonal rotations
spread outlier channels across a group before quantization. This page covers
the evidence behind the per-linear HiSQ rotation on Qwen3-8B and a partially
offline replacement for it on Qwen3-32B that needs four instead of seven online
transforms per decoder layer.

All accuracy values are MMLU 0-shot over 14,042 questions. KL is
KL(BF16 ‖ quantized) between the softmax distributions over the four answer
log-likelihoods, averaged over questions.

## HiSQ per-linear rotation

HiSQ gives every Linear `i` its own orthogonal matrix

```text
R_i = D_i · P_i · H
W_i' = W_i · R_i        (offline)
x_i' = x_i · R_i        (online, before activation quantization)
```

with random signs `D_i`, a permutation `P_i` of all input channels, and a block
Hadamard `H`. Signs and permutations follow the construction of
[GCC-HiFloat/HiSQRot4_Wan2.2](https://github.com/GCC-HiFloat/HiSQRot4_Wan2.2)
and are keyed by the layer name and a seed (17). Because q/k/v and gate/up get
different matrices, GPTQ collects a separate Hessian for each of them.

## Qwen3-8B

Setting: HiF4 W4A4 on all 252 projections, GPTQ (WikiText-2, 128 × 2048
tokens, 1% damping), attention output quantized, QK/PV not quantized, no
Smooth-QK and no QAD. BF16 is 72.9811% under this setting; the released
Qwen3-8B recipe uses a different evaluation setting (chat template), so these
numbers are not comparable with its 72.9597%.

| Linear input rotation | MMLU | Loss |
|---|---:|---:|
| Global residual rotation folded into the weights (online Hadamard before down_proj) | 70.5597% | 2.4213 pp |
| None | 71.1793% | 1.8018 pp |
| HiSQ, block 4096 | 71.4499% | 1.5312 pp |
| **HiSQ, block 1024** | **71.5639%** | **1.4172 pp** |

- HiSQ1024 beats the global rotation by 1.00 pp (95% task-block bootstrap
  interval [+0.29, +1.72], exact McNemar p = 0.0009). The global rotation is
  worse than no rotation.
- HiSQ1024 against no rotation: +0.38 pp, interval [−0.37, +1.16], p = 0.18.
  On four full LongBench tasks it recovers 41% of the no-rotation loss
  (macro +1.16, interval [−0.54, +2.85]).
- Blocks 64 and 256 scored below no rotation on a 1,430-question MMLU subset
  and were not evaluated further. 1024 and 4096 are indistinguishable
  (+0.11 pp); 1024 is the smaller online transform.

With QK/PV also quantized, the post-RoPE Q/K Hadamard (one 128-point Hadamard
applied to Q and K after RoPE, so `QKᵀ` is unchanged) is the transform that
protects the QK product. Added loss is relative to the same model with QK/PV
unquantized:

| Attention matmuls quantized | Added loss, HiSQ1024 only | Added loss, + post-RoPE Q/K Hadamard |
|---|---:|---:|
| QK | 2.51 pp | 0.74 pp |
| PV | 1.25 pp | 0.31 pp |
| QK and PV | 3.37 pp | 1.97 pp |

When QK is not quantized, adding the Q/K Hadamard lowered accuracy by 0.42 pp
(McNemar p = 0.08); the simulator rounds its FP32 output to BF16 once more.
The transform is therefore enabled only together with QK quantization.

## Qwen3-32B: partially offline block rotation

HiSQ1024 rotates each of the seven Linear inputs of a layer online. The aim
here is to move as much of this as possible offline while keeping its benefit.
The V/O per-head Hadamard (offline) and the post-RoPE Q/K Hadamard (online)
are unchanged.

### Result

Two configurations replace HiSQ1024 with no loss in accuracy. Per layer and
token, only four block Hadamards remain online: one 64-block transform shared
by q/k/v, one shared by gate/up, and one 1024-block transform each for o_proj
and down_proj. All signs and permutations are folded offline or written by the
RMSNorm in permuted order.

| HiF4 scale | Rotation | BF16 layers | MMLU | Loss | KL | LongBench loss (95% CI) |
|---|---|---|---:|---:|---:|---:|
| three-level | HiSQ1024 | 35, 41, 42, 45, 46 | 80.1239% | 0.7620 pp | 0.0818 | −0.20 [−0.75, +0.33] |
| three-level | **`rbhn_r64`** | 39, 40, 42, 45, 46 | 80.0527% | 0.8332 pp | 0.0813 | −0.01 [−0.53, +0.52] |
| two-level | HiSQ1024 | 39, 42, 43, 45, 46, 63 | 79.7963% | 1.0896 pp | 0.1000 | −0.10 [−0.62, +0.47] |
| two-level | **`rbh_r64`** | 41, 42, 43, 44, 45, 46 | 79.8747% | 1.0113 pp | 0.1090 | −0.20 [−0.77, +0.36] |

BF16: MMLU 80.8859%, LongBench 44.9224 (prefill-only, see
[Qwen3-32B](QWEN3_32B.md)). Each layer set was selected separately for its
rotation; see the selection notes below.

| Per layer and token | HiSQ1024 | `rbh_r64` / `rbhn_r64` |
|---|---:|---:|
| Online Hadamard transforms | 7, all 1024-block | 4: two 64-block, two 1024-block |
| Online sign multiplications | 7 | 0 |
| Online permutations (gathers) | 7 | 0 |
| Quantizations of residual-stream readers | 5 | 2 |
| Rotation MACs as block matrix products | 60.8 M (12.5% of Linear MACs) | 35.3 M (7.2%) |

With a fast Walsh–Hadamard transform both need about 0.1% of the Linear
operation count (0.59 M against 0.40 M additions); the savings are mainly the
signs, the gathers, and the separate rotation and quantization of q/k/v and of
gate/up.

### Why not fully offline

A rotation can be folded entirely into the weights only if the whole residual
stream shares it. RMSNorm normalization commutes with such a rotation,
`(hR) / rms(hR) = (h / rms(h)) R`, but the per-channel gain does not:
`(nR) ⊙ γ ≠ (n ⊙ γ) R`. The gain `γ` therefore has to be folded into the next
weights, and the quantized activation becomes `nR` without `γ`.

On Qwen3-32B this is what fails. Most channels of every RMSNorm have `γ`
between 0.25× and 4× the median, but about five channels per norm have
`γ` below 1% of the median (279 such channels over the 128 norms, 648 below
5%). These small gains suppress massive activations; once `γ` moves into the
weights, the large values reach the activation quantizer unscaled.

One factor at a time (three-level, layers 35, 41, 42, 45, 46 in BF16, o_proj
and down_proj keep HiSQ):

| q/k/v and gate/up input | MMLU | KL |
|---|---:|---:|
| Per-linear HiSQ, seeds 17 / 18 / 19 | 80.12 / 79.90 / 79.21% | 0.0818 / 0.0858 / 0.0872 |
| One rotation per RMSNorm, online | 79.58% | 0.0825 |
| One global rotation, online, `γ` stays in the activation | 80.18% | 0.0835 |
| One global rotation, online, `γ` folded into the weights | 79.15% | 0.1156 |
| One global rotation, offline (`shared_hisq_od`) | 79.27% | 0.1096 |
| No rotation | 79.82% | 0.0975 |
| No rotation, `γ` folded into the weights | 78.78% | 0.1389 |

- Sharing is not the problem: one rotation per RMSNorm or one for the whole
  network is as good as one per Linear, as long as `γ` stays in the
  activation.
- Folding `γ` is the problem: with the same global rotation, KL goes from
  0.0835 to 0.1156 (z = 11.2); without any rotation, folding `γ` alone takes
  KL from 0.0975 to 0.1389. Folding `γ` raises the weight-side HiF4 error by
  only 0.6–48% depending on the layer, so the loss is on the activation side.
- Accuracy varies by 0.9 pp across the three HiSQ seeds while KL varies by
  0.005, so rotation choices are compared on KL first.

### What can move offline

In the online transform `x · D · P · H`:

- **Signs `D`.** q/k/v read `RMSNorm(h) ⊙ γ`, and `(n ⊙ γ) D = n ⊙ (γ ⊙ d)`, so
  the signs fold into `γ` (gate/up likewise). The o_proj input is
  `softmax(QKᵀ) V`; flipping V output channels flips its input channels, so
  the signs fold into the v_proj output rows. The down_proj input is
  `silu(gate) ⊙ up`; the signs fold into the up_proj output rows (silu is not
  odd, so not into gate). HiF4 is symmetric, so sign flips do not change the
  quantization error.
- **Permutation `P`.** One global permutation reorders the whole residual
  stream and folds into the embedding, the o_proj/down_proj output rows, every
  `γ`, the final norm, and the LM head. A per-RMSNorm permutation leaves the
  residual stream alone and has the RMSNorm write its output in permuted order
  at no extra cost. The down_proj permutation folds into the gate/up output
  rows because silu is elementwise. o_proj gets no permutation: a permutation
  across heads cannot be folded into v_proj.
- **Block Hadamard `H`.** This is the remaining online work, applied to a
  shared input for q/k/v and for gate/up. A 64-value block coincides with a
  HiF4 group, so the transform never crosses a group and can be done inside
  the quantization kernel.

| Projection | Online | Folded offline | `rbh_r64` | `rbhn_r64` |
|---|---|---|---|---|
| q/k/v | 64-block Hadamard, once | signs into `input_layernorm.γ` | global permutation, folded into the residual stream | per-norm permutation, written by the RMSNorm |
| gate/up | 64-block Hadamard, once | signs into `post_attention_layernorm.γ` | same | same |
| o_proj | 1024-block Hadamard | HiSQ signs into the v_proj output rows | no permutation | no permutation |
| down_proj | 1024-block Hadamard | HiSQ permutation into the gate/up rows, signs into the up_proj rows | HiSQ seed 17 | HiSQ seed 17 |

In `rbh_r64` the reader signs and permutation come from one global rotation
(source `global_residual`, seed 0); in `rbhn_r64` from each RMSNorm's name and
the HiSQ seed 17.

### Which parts matter

Three-level, same layer set as above:

- Signs and permutations do not matter for KL. With o_proj/down_proj on HiSQ,
  the readers give KL 0.0843 without the permutation and 0.0846 with only a
  block Hadamard, against 0.0835 with both. Reader blocks of 16, 32, 64, 256,
  and 1024 values give KL 0.082–0.085.
- Block Hadamards on all seven Linears (no signs, no permutations) give KL
  0.0864 (all 1024), 0.0831 (readers 64, o/down 1024), and 0.0873 (all 64),
  within the range of the three HiSQ seeds.

Two-level (HiF4 with the per-4-value exponent removed, six layers in BF16):

- Randomization is needed. Over nine six-layer sets, plain block Hadamards
  (readers 64, o/down 1024) are 0.47 pp less accurate than HiSQ seed 17 on
  average and lower on all nine (standard error 0.06). Adding back the signs
  and permutations that fold offline (`rbh_r64`) closes the gap: −0.01 pp
  (standard error 0.07), and higher than plain block Hadamards on all nine.
- o_proj/down_proj need 1024-value blocks: 64-value blocks raise KL by 0.0094
  (z = 3.3).
- KL and accuracy rank differently: plain block Hadamards have lower KL than
  `rbh_r64` yet 0.46 pp lower accuracy. Both are needed to compare rotations
  here.

### Selecting the layer sets

The BF16 layer set interacts with the rotation, so each configuration was
selected with its own builds.

- **Three-level.** `rbh_r64` rebuilt on nine of the best HiSQ1024 sets (six
  by accuracy, three by KL) is 0.34 pp below HiSQ seed 17 on average. On three sets, five fixed
  randomizations were compared (HiSQ seed 17; `rbh_r64` with global seeds 0,
  17, 18; `rbhn_r64`). The same set moves by 0.4–0.5 pp across
  randomizations. The best is `rbhn_r64` with layers 39, 40, 42, 45, 46:
  0.8332 pp, 0.385 pp better than HiSQ1024 on the same set (1.2178), with the
  same KL (0.0813 against 0.0811). Ten more `rbh_r64` sets gave at best
  1.0611.
- **Two-level.** `rbh_r64` on the nine six-layer sets: best 1.0113 with layers
  41–46, median 1.18 (HiSQ seed 17: median 1.15 on the same sets).

On LongBench the three-level `rbhn_r64` build has a measurable loss only on
the synthetic tasks (+1.50, [0.17, 2.83], mostly `passage_count`); for the
two-level `rbh_r64` build the synthetic loss of its HiSQ1024 counterpart
(+2.00, [0.50, 3.50]) is no longer significant (+0.83, [−0.50, +2.17]).

### Implementation

All modes are values of `--rotation` (`faquant.qad.QAD_ROTATIONS`). The build,
Smooth-QK calibration, and evaluation of a checkpoint must use the same value.

| Mode | q/k/v and gate/up | o_proj and down_proj |
|---|---|---|
| `hisq1024` | HiSQ per Linear | HiSQ per Linear |
| `rbh_r64` | global signs and permutation, 64-block Hadamard | HiSQ signs (o_proj), signs and permutation (down_proj), 1024-block Hadamard |
| `rbhn_r64` | per-RMSNorm signs and permutation, 64-block Hadamard | as `rbh_r64` |
| `shared_hisq_od` | one global rotation folded offline, `γ` folded | HiSQ |
| `hisq_od` | none | HiSQ |
| `hisq_norm` | HiSQ shared per RMSNorm, online | HiSQ |
| `hisq_global` | one HiSQ rotation shared by all readers, online | HiSQ |
| `hisq_global_fused` | as `hisq_global`, `γ` folded into the weights | HiSQ |
| `fused_od` | none, `γ` folded into the weights | HiSQ |
| `sbh_readers` | global signs, 1024-block Hadamard | HiSQ |
| `bh_readers`, `bh256_readers`, `bh64_readers`, `bh32_readers`, `bh16_readers` | block Hadamard (1024, 256, 64, 32, 16) | HiSQ |
| `bh_all`, `bh_all_r64`, `bh64_all` | block Hadamard (1024, 64, 64) | block Hadamard (1024, 1024, 64) |

The modes set these `ExperimentConfig` fields: `qwen_hisq_rotation_sharing`
(`linear`, `norm`, or `global`), `qwen_hisq_global_seed`,
`qwen_hisq_reader_construction` and `qwen_hisq_od_construction` (`full`,
`signs`, or `plain`), `qwen_hisq_o_construction`,
`qwen_hisq_reader_block_size`, and `qwen_fuse_reader_norms`.
`faquant.hisq_rotation.derive_qwen_hisq_rotation` generates every rotation for
both GPTQ and checkpoint loading. `--hisq-seed` and `--hisq-global-seed` of
`scripts/prepare_ptq.py` set the per-Linear and global seeds.

The simulator applies the rotation online in `FakeQuantLinear` (signs, then
permutation, then block Hadamard), which is mathematically identical to
folding the signs and permutations into the weights and norms. Checkpoints
store each Linear's signs and permutation; the block size follows from the
mode. `tests/test_qwen3_rotation_modes.py` checks which Linears share a
rotation in each mode, the seeds, block sizes and constructions, that folding
`γ` into the weights leaves the logits unchanged, and that conflicting options
are rejected.
