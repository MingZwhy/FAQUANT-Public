"""Smooth-QK, the V/O head fold and bf16 protection leave Wan attention exact in fp32.

Needs a checkout of github.com/Wan-Video/Wan2.2 in WAN22_ROOT; skipped otherwise.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

if os.environ.get("WAN22_ROOT"):
    sys.path.insert(0, os.environ["WAN22_ROOT"])
if not torch.cuda.is_available():  # wan's t5.py reads the current device in a default argument
    torch.cuda.current_device = lambda: 0

wan_model = pytest.importorskip("wan.modules.model", reason="set WAN22_ROOT to a Wan2.2 checkout")

from faquant.wan22_attention import _BF16_ATTENTION, patch_attention_modules  # noqa: E402
from faquant.wan22_hif4 import fold_value_head_hadamard  # noqa: E402

seen_bf16 = []


def exact_attention(q, k, v, k_lens=None, window_size=(-1, -1), **_):
    seen_bf16.append(_BF16_ATTENTION.get())
    out = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
    return out.transpose(1, 2)


def module_namespace():
    return SimpleNamespace(WanSelfAttention=type("S", (wan_model.WanSelfAttention,), {}),
                           WanCrossAttention=type("C", (wan_model.WanCrossAttention,), {}),
                           rope_apply=lambda x, grid_sizes, freqs: x.float(), flash_attention=exact_attention)


def test_smooth_qk_and_value_fold_are_exact():
    torch.manual_seed(0)
    ns = module_namespace()
    patch_attention_modules(ns)
    attn = ns.WanSelfAttention(256, 2).double().float()
    x = torch.randn(1, 32, 256)
    ref = attn(x, None, None, None)
    attn.register_buffer("faquant_qk_smooth_scale", torch.rand(128) * 4 + 0.25, persistent=False)
    assert torch.allclose(attn(x, None, None, None), ref, atol=1e-5)
    folded = SimpleNamespace(blocks=[SimpleNamespace(self_attn=attn)])
    fold_value_head_hadamard(folded, device="cpu")
    assert torch.allclose(attn(x, None, None, None), ref, atol=1e-4)


def test_bf16_flag_reaches_the_attention_call():
    ns = module_namespace()
    patch_attention_modules(ns)
    attn = ns.WanCrossAttention(256, 2)
    x, ctx = torch.randn(1, 8, 256), torch.randn(1, 4, 256)
    seen_bf16.clear()
    attn(x, ctx, None)
    attn.faquant_attention_bf16 = True
    attn(x, ctx, None)
    assert seen_bf16 == [False, True]

