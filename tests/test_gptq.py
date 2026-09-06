import copy
import pytest
import torch
from torch import nn

import faquant.gptq as gptq_module
from faquant.gptq import GPTQ, calibration_tokens_from_jsonl
from faquant.hif4 import quantize_hif4_with_compact_parameters
from faquant.quantization import fake_quantize


def test_gptq_quantizes_small_linear() -> None:
    torch.manual_seed(0)
    layer = nn.Linear(16, 8, bias=False)
    quantizer = GPTQ(layer)
    calibration = torch.randn(4, 12, 16)
    quantizer.add_batch(calibration)
    original = layer.weight.detach().clone()
    stats = quantizer.quantize(bits=4, group_size=8, damp=0.01, block_size=8)
    assert stats.samples == 48
    assert stats.mean_loss > 0
    assert stats.damp_multiplier >= 1
    assert torch.isfinite(layer.weight).all()
    assert not torch.equal(original, layer.weight)


def test_gptq_rejects_nonpositive_block_size_without_changing_weights() -> None:
    torch.manual_seed(2)
    layer = nn.Linear(64, 4, bias=False)
    quantizer = GPTQ(layer)
    quantizer.add_batch(torch.randn(2, 8, 64))
    original = layer.weight.detach().clone()
    with (
        torch.inference_mode(),
        pytest.raises(ValueError, match="block_size must be positive"),
    ):
        quantizer.quantize(
            quant_format="hif4",
            bits=4,
            group_size=64,
            block_size=-64,
        )
    torch.testing.assert_close(layer.weight, original, atol=0, rtol=0)


def test_gptq_retries_when_inverse_cholesky_is_ill_conditioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cholesky_ex = torch.linalg.cholesky_ex
    calls = 0

    def fail_first_inverse(
        matrix: torch.Tensor,
        *,
        upper: bool = False,
        check_errors: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        factor, info = original_cholesky_ex(
            matrix, upper=upper, check_errors=check_errors
        )
        if calls == 2:
            info = torch.ones_like(info)
        return factor, info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", fail_first_inverse)
    factor, multiplier = GPTQ._damped_inverse_cholesky(torch.eye(4), torch.tensor(0.01))
    assert calls == 4
    assert multiplier == 10.0
    assert torch.isfinite(factor).all()


def test_hif4_gptq_fits_metadata_once_per_64_column_group(
    monkeypatch,
) -> None:
    torch.manual_seed(4)
    layer = nn.Linear(128, 8, bias=False)
    quantizer = GPTQ(layer)
    calibration = torch.randn(4, 32, 128)
    quantizer.add_batch(calibration)
    original = layer.weight.detach().clone()
    target = calibration @ original.T
    rtn_weight = fake_quantize(original, quant_format="hif4", bits=4, group_size=64)
    rtn_loss = (calibration @ rtn_weight.T - target).square().mean()

    calls = 0
    fit_reference = gptq_module.fit_hif4_parameters

    def counted_fit(blocks: torch.Tensor):
        nonlocal calls
        calls += 1
        return fit_reference(blocks)

    monkeypatch.setattr(gptq_module, "fit_hif4_parameters", counted_fit)
    stats = quantizer.quantize(
        quant_format="hif4",
        bits=4,
        group_size=64,
        damp=0.01,
        block_size=128,
    )

    assert calls == 2
    assert stats.samples == 128
    assert stats.mean_loss > 0
    assert stats.damp_multiplier >= 1
    assert torch.isfinite(layer.weight).all()
    assert not torch.equal(original, layer.weight)
    gptq_loss = (calibration @ layer.weight.T - target).square().mean()
    assert gptq_loss < rtn_loss


def test_hif4_gptq_latent_master_preserves_deployed_codes() -> None:
    torch.manual_seed(6)
    layer = nn.Linear(128, 8, bias=False, dtype=torch.bfloat16)
    quantizer = GPTQ(layer)
    quantizer.add_batch(torch.randn(4, 32, 128, dtype=torch.bfloat16))
    quantizer.quantize(
        quant_format="hif4",
        bits=4,
        group_size=64,
        damp=0.01,
        block_size=128,
        capture_hif4_metadata=True,
        capture_latent_master=True,
    )
    master = layer.faquant_hif4_qat_master
    assert master.device.type == "cpu"
    grouped = master.reshape(8, 2, 64)
    deployed = quantize_hif4_with_compact_parameters(
        grouped,
        layer.faquant_hif4_qat_parameters,
    ).reshape_as(layer.weight)
    torch.testing.assert_close(deployed, layer.weight.cpu(), atol=0, rtol=0)
    assert torch.any(master.ne(layer.weight.cpu()))


def test_jsonl_calibration_is_deterministic_and_category_balanced(tmp_path) -> None:
    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return [int(messages[-1]["content"])] * 3

    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "".join(
            [
                '{"category":"general","messages":[{"role":"assistant","content":"1"}]}\n',
                '{"category":"general","messages":[{"role":"assistant","content":"2"}]}\n',
                '{"category":"reasoning","messages":[{"role":"assistant","content":"7"}]}\n',
            ]
        ),
        encoding="utf-8",
    )
    first = calibration_tokens_from_jsonl(
        Tokenizer(), path, nsamples=2, seqlen=8, seed=3, max_record_tokens=2
    )
    second = calibration_tokens_from_jsonl(
        Tokenizer(), path, nsamples=2, seqlen=8, seed=3, max_record_tokens=2
    )
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.shape == (2, 8)
    assert 7 in first
    assert 1 in first or 2 in first
    assert 99 in first


def _hif4_gptq_layer(seed: int = 0):
    """A layer wide enough for two HiF4 groups, with a skewed Hessian."""

    import torch
    from torch import nn

    torch.manual_seed(seed)
    layer = nn.Linear(128, 16, bias=False)
    # Column influence has to vary, or act-order has nothing to reorder.
    scale = torch.logspace(-1.5, 1.5, 128)
    activations = torch.randn(256, 128) * scale
    return layer, activations


def test_in_group_act_order_keeps_columns_inside_their_group():
    """Group membership must survive, or the deployed model reads a different
    set of 64 values than the one the parameters were fitted to."""

    import torch

    from faquant.gptq import _in_group_act_order_permutation

    torch.manual_seed(0)
    hessian = torch.diag(torch.rand(128) + 0.1)
    perm = _in_group_act_order_permutation(hessian, 64)

    assert sorted(perm.tolist()) == list(range(128))
    assert sorted(perm[:64].tolist()) == list(range(64))
    assert sorted(perm[64:].tolist()) == list(range(64, 128))
    # Within a group the order really is by descending influence.
    first = hessian.diagonal()[perm[:64]]
    assert torch.all(first[:-1] >= first[1:])


def test_in_group_act_order_is_a_no_op_when_the_hessian_is_already_sorted():
    """The strongest available check that the permutation round-trips.

    With a Hessian whose diagonal already descends inside every group the
    permutation is the identity, so the result has to match the unpermuted
    solve bit for bit. Any mistake in permuting H, W or the HiF4 parameters
    would show up here, and the doc this follows warns the failure is silent.
    """

    import torch

    from faquant.gptq import GPTQ

    layer, activations = _hif4_gptq_layer()
    # Descending within each 64-group, so argsort returns the identity.
    scale = torch.cat([torch.logspace(1.5, -1.5, 64)] * 2)
    activations = torch.randn(256, 128) * scale

    reference = GPTQ(copy.deepcopy(layer))
    reference.add_batch(activations)
    reference.quantize(quant_format="hif4", bits=4, group_size=64)

    reordered = GPTQ(copy.deepcopy(layer))
    reordered.add_batch(activations)
    reordered.quantize(
        quant_format="hif4", bits=4, group_size=64, act_order_within_group=True
    )

    torch.testing.assert_close(
        reordered.layer.weight, reference.layer.weight, atol=0.0, rtol=0.0
    )


def test_in_group_act_order_changes_the_solution_but_not_the_grid():
    """It must actually do something, and the result must stay HiF4-legal.

    Staying on the grid is what the build's step-0 parity check enforces
    downstream; catching a violation here is much cheaper than catching it
    after a nine-minute checkpoint build.
    """

    import torch

    from faquant.gptq import GPTQ
    from faquant.hif4 import fake_quantize_hif4

    layer, activations = _hif4_gptq_layer()

    reference = GPTQ(copy.deepcopy(layer))
    reference.add_batch(activations)
    reference.quantize(quant_format="hif4", bits=4, group_size=64)

    reordered = GPTQ(copy.deepcopy(layer))
    reordered.add_batch(activations)
    reordered.quantize(
        quant_format="hif4", bits=4, group_size=64, act_order_within_group=True
    )

    assert not torch.equal(reordered.layer.weight, reference.layer.weight)
    # Re-encoding a value already on the HiF4 grid returns it unchanged.
    weight = reordered.layer.weight.float()
    torch.testing.assert_close(fake_quantize_hif4(weight), weight, atol=0.0, rtol=0.0)


def test_in_group_act_order_preserves_captured_metadata_alignment():
    """Captured parameters are stored per position, so they must describe the
    weight in storage order rather than in the order the solver visited."""

    import torch

    from faquant.gptq import GPTQ
    from faquant.hif4 import quantize_hif4_with_compact_parameters

    layer, activations = _hif4_gptq_layer()
    solver = GPTQ(copy.deepcopy(layer))
    solver.add_batch(activations)
    solver.quantize(
        quant_format="hif4",
        bits=4,
        group_size=64,
        capture_hif4_metadata=True,
        act_order_within_group=True,
    )

    weight = solver.layer.weight.float()
    grouped = weight.reshape(weight.shape[0], weight.shape[1] // 64, 64)
    round_trip = quantize_hif4_with_compact_parameters(
        grouped, solver.layer.faquant_hif4_qat_parameters
    ).reshape_as(weight)
    torch.testing.assert_close(round_trip, weight, atol=0.0, rtol=0.0)
