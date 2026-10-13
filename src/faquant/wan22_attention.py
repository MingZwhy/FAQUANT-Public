"""HiF4 QK/PV attention for the Wan2.2 DiT, as a Triton FlashAttention kernel.

Same quantization as ``faquant.simulated_flash_attention`` (the simulator of the
Qwen path), which is far too slow for 75,600-token video attention:

- Q, K and V are HiF4 per token along head_dim (two blocks of 64 for head_dim 128);
  done outside the kernel by ``input_qdq``.
- Keys are processed in tiles of 128 in order, with an fp32 online softmax. In each
  tile the unnormalized ``P = exp(score - running_max)`` is HiF4-quantized along the
  keys (blocks of 64, zero padding past the last key, which a block maximum ignores)
  and multiplied with V; scores, running max, denominator and accumulator stay fp32.
  As in FlashAttention, the exponential is a hardware exp2 on log2-scaled scores, so P
  can differ from the simulator's ``torch.exp`` by a few ulp.
- The softmax denominator sums the unquantized P (the simulator's default), or the
  quantized one with ``quantized_normalizer`` (its P-Reordering mode).

The in-kernel HiF4 quantizer is bit-exact with ``faquant.hif4`` on the same input:
exponents come from the fp32 bit pattern, the reciprocal is correctly rounded, and
every other step is an exact power-of-two scaling or the same single rounding.
HiF4 values are exact in bf16, so PV runs on bf16 tensor cores with fp32 accumulation.
"""

from __future__ import annotations

import contextvars
from typing import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .rotation import hadamard_transform

KEY_TILE = 128
R7 = float(torch.tensor(1.0 / 7.0).to(torch.bfloat16).float())  # bf16 1/7, as the reference rounds it
_BF16_ATTENTION = contextvars.ContextVar("faquant_bf16_attention", default=False)


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _exponent(x):
    # floor(log2(x)) of a positive normal fp32 value
    return ((x.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127


@triton.jit
def _pow2(e):
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _hif4_qdq_128(x, BM: tl.constexpr, R7: tl.constexpr, SIGNED: tl.constexpr):
    """Official three-level HiF4 QDQ of each 64-value block of a [BM, 128] fp32 tile."""
    if SIGNED:
        a = tl.reshape(tl.abs(x), (BM, 32, 4))
    else:
        a = tl.reshape(x, (BM, 32, 4))
    m3 = tl.max(a, axis=2)                                   # per 4 values
    m2 = tl.max(tl.reshape(m3, (BM, 16, 2)), axis=2)         # per 8 values
    m1 = tl.max(tl.reshape(m2, (BM, 2, 8)), axis=2)          # per 64 values
    s = _bf16(m1 * R7)
    s = tl.minimum(tl.maximum(s, 3.552713678800501e-15), 49152.0)   # [2^-48, 49152]
    e = _exponent(s)
    s = libdevice.rint(s * _pow2(2 - e)) * _pow2(e - 2)    # E6M2, round half to even
    rec = _bf16(libdevice.div_rn(1.0, s))
    s16 = tl.reshape(tl.broadcast_to(s[:, :, None], (BM, 2, 8)), (BM, 16))
    rec16 = tl.reshape(tl.broadcast_to(rec[:, :, None], (BM, 2, 8)), (BM, 16))
    up2 = tl.floor(tl.minimum(tl.maximum(m2 * rec16, 0.0), 4.0) * 0.25) > 0
    sc2 = tl.where(up2, 2.0, 1.0)
    inv2 = tl.where(up2, 0.5, 1.0)
    s32 = tl.reshape(tl.broadcast_to((s16 * sc2)[:, :, None], (BM, 16, 2)), (BM, 32))
    rec32 = tl.reshape(tl.broadcast_to((rec16 * inv2)[:, :, None], (BM, 16, 2)), (BM, 32))
    up3 = tl.floor(tl.minimum(tl.maximum(m3 * rec32, 0.0), 2.0) * 0.5) > 0
    qm = rec32 * tl.where(up3, 0.5, 1.0)
    ds = s32 * tl.where(up3, 2.0, 1.0)
    mant = tl.minimum(tl.floor(a * qm[:, :, None] * 4.0 + 0.5) * 0.25, 1.75)
    y = tl.reshape(mant * ds[:, :, None], (BM, 128))
    if SIGNED:
        y = tl.where(x < 0, -y, y)
    return y


@triton.jit
def _qdq_rows_kernel(X, Y, R, BM: tl.constexpr, R7: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    ptrs = rows[:, None] * 128 + cols[None, :]
    x = tl.load(X + ptrs, mask=rows[:, None] < R, other=0.0)
    tl.store(Y + ptrs, _hif4_qdq_128(x, BM, R7, True), mask=rows[:, None] < R)


def qdq_rows_128(x: torch.Tensor, block_rows: int = 64) -> torch.Tensor:
    """The in-kernel quantizer applied to an [R, 128] fp32 tensor (for tests)."""
    x = x.contiguous().float()
    y = torch.empty_like(x)
    _qdq_rows_kernel[(triton.cdiv(x.shape[0], block_rows),)](x, y, x.shape[0], BM=block_rows, R7=R7)
    return y


@triton.jit
def _attention_tile(acc, l_i, m_i, q, K, V, start, Lk, sk_l, sv_l, dims, cols, qk_scale,
                    BM: tl.constexpr, MASKED: tl.constexpr, QUANT_P: tl.constexpr,
                    QUANT_NORM: tl.constexpr, R7: tl.constexpr):
    keys = start + cols
    if MASKED:
        valid = keys < Lk
        kt = tl.load(K + keys[None, :] * sk_l + dims[:, None], mask=valid[None, :], other=0.0)
        v = tl.load(V + keys[:, None] * sv_l + dims[None, :], mask=valid[:, None], other=0.0)
        s = tl.where(valid[None, :], tl.dot(q, kt) * qk_scale, float("-inf"))
    else:
        kt = tl.load(K + keys[None, :] * sk_l + dims[:, None])
        v = tl.load(V + keys[:, None] * sv_l + dims[None, :])
        s = tl.dot(q, kt) * qk_scale
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    if QUANT_P:
        pq = _hif4_qdq_128(p, BM, R7, False)  # P >= 0
    else:
        pq = p
    if QUANT_NORM:
        l_i = alpha * l_i + tl.sum(pq, axis=1)
    else:
        l_i = alpha * l_i + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(pq.to(tl.bfloat16), v)
    return acc, l_i, m_new


@triton.jit
def _attention_kernel(
    Q, K, V, O, Lq, Lk,
    sq_l, sq_h, sk_l, sk_h, sv_l, sv_h, so_l, so_h,
    scale,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
    QUANT_P: tl.constexpr, QUANT_NORM: tl.constexpr, R7: tl.constexpr,
):
    head = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    cols = tl.arange(0, BN)
    q = tl.load(Q + rows[:, None] * sq_l + head * sq_h + dims[None, :], mask=rows[:, None] < Lq, other=0.0)
    K += head * sk_h
    V += head * sv_h
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    # scores in the log2 domain, so exp(score - max) is one fast exp2 as in FlashAttention
    qk_scale = scale * 1.4426950408889634
    full = (Lk // BN) * BN
    for start in range(0, full, BN):
        acc, l_i, m_i = _attention_tile(acc, l_i, m_i, q, K, V, start, Lk, sk_l, sv_l, dims, cols, qk_scale,
                                        BM, False, QUANT_P, QUANT_NORM, R7)
    if full < Lk:
        acc, l_i, m_i = _attention_tile(acc, l_i, m_i, q, K, V, full, Lk, sk_l, sv_l, dims, cols, qk_scale,
                                        BM, True, QUANT_P, QUANT_NORM, R7)
    o = acc / l_i[:, None]
    tl.store(O + rows[:, None] * so_l + head * so_h + dims[None, :], o.to(tl.bfloat16), mask=rows[:, None] < Lq)


def hif4_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, softmax_scale: float | None = None,
    quantize_p: bool = True, quantized_normalizer: bool = False, block_m: int = 128,
    num_warps: int = 8, num_stages: int = 1,
) -> torch.Tensor:
    """Non-causal attention of [L, H, D] bf16 tensors (Q/K/V already quantized); bf16 out."""
    lq, heads, dim = q.shape
    lk = k.shape[0]
    assert dim == 128 and k.shape == v.shape and k.shape[1:] == (heads, dim)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    scale = float(softmax_scale if softmax_scale is not None else dim ** -0.5)
    out = torch.empty_like(q)
    _attention_kernel[(triton.cdiv(lq, block_m), heads)](
        q, k, v, out, lq, lk,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        scale, BM=block_m, BN=KEY_TILE, D=dim, QUANT_P=quantize_p, QUANT_NORM=quantized_normalizer, R7=R7,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def rotate_heads(x: torch.Tensor, chunk_rows: int = 1 << 16) -> torch.Tensor:
    """Normalized Hadamard over head_dim in fp32, cast once to bf16 (as the Qwen path does)."""
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty(flat.shape, dtype=torch.bfloat16, device=x.device)
    for start in range(0, flat.shape[0], chunk_rows):
        out[start:start + chunk_rows] = hadamard_transform(flat[start:start + chunk_rows].float()).to(torch.bfloat16)
    return out.reshape(x.shape)


def make_flash_attention(
    input_qdq: Callable[[torch.Tensor], torch.Tensor], *, quantized_normalizer: bool = False,
    qk_rotation: bool = False, fallback: Callable | None = None,
) -> Callable:
    """A drop-in for ``wan.modules.attention.flash_attention`` with HiF4 Q/K/P/V.

    ``qk_rotation`` applies the same head_dim Hadamard to Q and K (after RoPE) before
    they are quantized; Q K^T is unchanged in exact arithmetic. Calls made for a module
    marked ``faquant_attention_bf16`` (see ``patch_attention_modules``) go to ``fallback``,
    the original bf16 FlashAttention."""

    def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                        causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
                        version=None):
        if fallback is not None and _BF16_ATTENTION.get():
            return fallback(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
                            q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic,
                            dtype=dtype, version=version)
        assert dropout_p == 0 and q_scale is None and not causal and tuple(window_size) == (-1, -1)
        batch, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        # Wan pads nothing at batch 1; a shorter length would need key masking.
        for lens, full in ((q_lens, lq), (k_lens, lk)):
            if lens is not None and any(int(n) != full for n in lens.tolist()):
                raise NotImplementedError("variable sequence lengths are not supported")
        if qk_rotation:
            q, k = rotate_heads(q), rotate_heads(k)
        q, k, v = (input_qdq(t.to(dtype)) for t in (q, k, v))
        out = torch.stack([
            hif4_attention(q[i], k[i], v[i], softmax_scale=softmax_scale, quantized_normalizer=quantized_normalizer)
            for i in range(batch)
        ])
        return out.type(out_dtype)

    return flash_attention


def patch_attention_modules(model_module, collector: Callable | None = None) -> None:
    """Per-module attention settings for ``wan.modules.model`` (same computation otherwise):

    - a ``faquant_qk_smooth_scale`` buffer (head_dim, or [heads, head_dim]) on a WanSelfAttention
      multiplies Q and divides K by it after RoPE (Q K^T unchanged), in fp32 ahead of the
      head_dim Hadamard and the single bf16 cast, as Smooth-QK does in the Qwen recipe;
    - ``faquant_attention_bf16 = True`` sends that module's attention to the bf16 fallback;
    - ``collector(module, q, k)`` sees the post-RoPE Q/K of every self-attention call.
    """
    rope_apply = model_module.rope_apply

    def attend(module, *args, **kwargs):
        token = _BF16_ATTENTION.set(bool(getattr(module, "faquant_attention_bf16", False)))
        try:
            return model_module.flash_attention(*args, **kwargs)
        finally:
            _BF16_ATTENTION.reset(token)

    def self_forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q, k = rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs)
        if collector is not None:
            collector(self, q, k)
        scale = getattr(self, "faquant_qk_smooth_scale", None)
        if scale is not None:
            q, k = q * scale, k / scale
        x = attend(self, q=q, k=k, v=v, k_lens=seq_lens, window_size=self.window_size)
        return self.o(x.flatten(2))

    def cross_forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = attend(self, q, k, v, k_lens=context_lens)
        return self.o(x.flatten(2))

    model_module.WanSelfAttention.forward = self_forward
    model_module.WanCrossAttention.forward = cross_forward


def parse_bf16_attention(spec: str) -> dict[str, set]:
    """'low:33,34,38;cross' -> self-attention blocks per expert kept in bf16, plus 'cross' for all
    cross-attention. Experts: high, low, both."""
    out = {"high_noise_model": set(), "low_noise_model": set(), "cross": set()}
    for entry in filter(None, (e.strip() for e in spec.split(";"))):
        if entry == "cross":
            out["cross"] = {"all"}
            continue
        expert, _, blocks = entry.partition(":")
        tags = {"high": ["high_noise_model"], "low": ["low_noise_model"],
                "both": ["high_noise_model", "low_noise_model"]}[expert]
        for tag in tags:
            out[tag] |= {int(b) for b in blocks.split(",")}
    return out
