"""Wan2.2 HiF4 GPTQ pieces: given weights, the Hessian bank, and GPTQ against RTN."""

import torch
from torch import nn

from faquant.gptq import GPTQ
from faquant.hif4 import fake_quantize_hif4
from faquant.wan22_hif4 import HiF4Linear, faquant_qdq
from faquant.workflows.wan22_gptq import GPTQ_SETTINGS, HessianBank, relative_error


def rtn(weight):
    return fake_quantize_hif4(weight.float(), levels="123")


def gptq(weight, hessian, samples):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    layer.weight.data.copy_(weight.float())
    solver = GPTQ(layer)
    solver.hessian, solver.samples = hessian, samples
    solver.quantize(damp=0.01, **GPTQ_SETTINGS)
    return layer.weight.detach()


def test_given_rtn_weights_match_the_rtn_build():
    torch.manual_seed(0)
    linear = nn.Linear(256, 96)
    x = torch.randn(3, 40, 256)
    qdq = faquant_qdq("123")
    built = HiF4Linear(linear, weight_qdq=qdq, input_qdq=qdq, device="cpu")
    given = HiF4Linear(linear, weight_qdq=None, input_qdq=qdq, device="cpu", weight=rtn(linear.weight))
    assert torch.equal(built.weight, given.weight)
    assert torch.equal(built(x), given(x))


def test_bank_sums_every_stride_th_quantized_row_across_chunks():
    torch.manual_seed(0)
    linear = nn.Linear(128, 64)
    qdq = faquant_qdq("123")
    module = HiF4Linear(linear, weight_qdq=qdq, input_qdq=qdq, device="cpu", chunk_elements=128 * 37)
    bank = HessianBank(stride=4)
    module.collector = bank.collector(0, "x")
    x = torch.randn(2, 150, 128)
    module(x)
    module(x[:1])
    bank.flush()
    rows = qdq(x.reshape(-1, 128))
    expected = rows[::4].T @ rows[::4] + rows[:150][::4].T @ rows[:150][::4]
    assert bank.tokens["0.x"] == len(rows[::4]) + len(rows[:150][::4])
    torch.testing.assert_close(bank.host["0.x"], expected, rtol=1e-5, atol=1e-3)


def test_gptq_is_rtn_without_correlation_and_beats_it_with_correlation():
    torch.manual_seed(0)
    weight = torch.randn(64, 256).to(torch.bfloat16)
    assert torch.equal(gptq(weight, torch.eye(256) * 1000, 1000), rtn(weight))

    mix = torch.randn(256, 256) / 16 + torch.eye(256)
    inputs = torch.randn(4096, 256) @ mix
    hessian = inputs.T @ inputs
    quantized = gptq(weight, hessian.clone(), len(inputs))
    assert torch.equal(quantized.to(torch.bfloat16).float(), quantized)
    w = weight.float()
    assert relative_error(w - quantized, hessian, w) < 0.8 * relative_error(w - rtn(weight), hessian, w)
