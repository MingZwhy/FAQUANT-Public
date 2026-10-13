import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from faquant.config import ExperimentConfig
from faquant.qwen3 import prepare_model


def _tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        # Exercise the same 12 * 2^n generalized Hadamard path as Qwen3-8B's
        # 12288-dimensional FFN, not only the power-of-two fast path.
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(config).eval()


def _tiny_qwen3_hif4() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(config).eval()


def _tiny_qwen3_two_layer_hif4() -> Qwen3ForCausalLM:
    """Two layers, 64 wide: the narrowest model HiF4's 64-value groups allow."""

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    return Qwen3ForCausalLM(config).eval()


def _rotation_key(module: torch.nn.Module) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(module.input_rotation_signs.tolist()),
        tuple(module.input_rotation_permutation.tolist()),
    )


def test_shared_rotation_preserves_logits_when_width_is_not_a_power_of_two() -> None:
    from faquant.qad import qwen3_hif4_qad_config

    torch.manual_seed(0)
    baseline = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=48, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, attention_dropout=0.0,
    )).eval()
    rotated = copy.deepcopy(baseline)
    inputs = torch.randint(0, 64, (2, 11))
    config = qwen3_hif4_qad_config(rotation="shared_hisq_od")
    assert config.rotation == "hadamard" and not config.online_hadamard
    assert config.qwen_hisq_input_rotation
    assert config.qwen_hisq_rotation_projections == ("o_proj", "down_proj")
    with torch.inference_mode():
        expected = baseline(inputs, use_cache=False).logits
        prepare_model(rotated, ExperimentConfig(rotation="hadamard", online_hadamard=False))
        actual = rotated(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    assert rotated.faquant_global_rotation_permutation.numel() == 48
    assert not hasattr(rotated, "faquant_rotation_signs")
    assert not getattr(rotated.model.layers[0].mlp.down_proj, "faquant_online_hadamard", False)


@pytest.mark.parametrize("weight_quant", ["gptq", "rtn"])
def test_shared_rotation_keeps_hisq_on_o_and_down_proj_only(weight_quant: str) -> None:
    torch.manual_seed(10)
    model = _tiny_qwen3_hif4()
    calibration = torch.randint(0, 64, (2, 8))
    config = ExperimentConfig(
        rotation="hadamard",
        online_hadamard=False,
        quant_target="all",
        quant_format="hif4",
        weight_quant=weight_quant,
        qwen_hisq_input_rotation=True,
        qwen_hisq_rotation_block_size=16,
        qwen_hisq_rotation_projections=("o_proj", "down_proj"),
    )
    with torch.inference_mode():
        prepare_model(model, config, calibration_input_ids=calibration)
        logits = model(calibration[:1], use_cache=False).logits

    assert torch.isfinite(logits).all()
    layer = model.model.layers[0]
    rotated = {
        name: module.input_rotation_signs is not None
        for name, module in (
            ("q_proj", layer.self_attn.q_proj), ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj), ("o_proj", layer.self_attn.o_proj),
            ("gate_proj", layer.mlp.gate_proj), ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
        )
    }
    assert rotated == {name: name in ("o_proj", "down_proj") for name in rotated}
    with pytest.raises(ValueError, match="o_proj/down_proj"):
        ExperimentConfig(
            rotation="hadamard", online_hadamard=False, quant_target="all", quant_format="hif4",
            qwen_hisq_input_rotation=True, qwen_hisq_rotation_projections=("q_proj",),
        )


@pytest.mark.parametrize("weight_quant", ["gptq", "rtn"])
@pytest.mark.parametrize("sharing", ["norm", "global"])
def test_hisq_sharing_gives_the_readers_one_rotation(sharing: str, weight_quant: str) -> None:
    from faquant.hisq_rotation import derive_hisq_rotation

    torch.manual_seed(11)
    model = _tiny_qwen3_two_layer_hif4()
    calibration = torch.randint(0, 64, (2, 8))
    config = ExperimentConfig(
        quant_target="all",
        quant_format="hif4",
        weight_quant=weight_quant,
        qwen_hisq_input_rotation=True,
        qwen_hisq_rotation_block_size=16,
        qwen_hisq_rotation_sharing=sharing,
    )
    with torch.inference_mode():
        prepare_model(model, config, calibration_input_ids=calibration)
        assert torch.isfinite(model(calibration[:1], use_cache=False).logits).all()

    first, second = model.model.layers
    for layer in (first, second):
        attention, mlp = layer.self_attn, layer.mlp
        assert _rotation_key(attention.q_proj) == _rotation_key(attention.k_proj)
        assert _rotation_key(attention.q_proj) == _rotation_key(attention.v_proj)
        assert _rotation_key(mlp.gate_proj) == _rotation_key(mlp.up_proj)
        assert _rotation_key(attention.o_proj) != _rotation_key(attention.q_proj)
        assert _rotation_key(mlp.down_proj) != _rotation_key(mlp.gate_proj)
    shared = sharing == "global"
    assert (_rotation_key(first.self_attn.q_proj) == _rotation_key(second.self_attn.q_proj)) is shared
    assert (_rotation_key(first.self_attn.q_proj) == _rotation_key(first.mlp.gate_proj)) is shared
    if shared:
        residual = derive_hisq_rotation("global_residual", 64, seed=config.seed, block_size=16)
        assert _rotation_key(first.mlp.up_proj) == (
            tuple(residual.signs.tolist()), tuple(residual.permutation.tolist())
        )


def test_reader_norm_fusion_preserves_logits_and_moves_gamma_into_the_weights() -> None:
    torch.manual_seed(12)
    baseline = _tiny_qwen3()
    for layer in baseline.model.layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 2.0)
        layer.post_attention_layernorm.weight.data.uniform_(0.5, 2.0)
    fused = copy.deepcopy(baseline)
    inputs = torch.randint(0, 64, (2, 11))
    with torch.inference_mode():
        expected = baseline(inputs, use_cache=False).logits
        prepare_model(fused, ExperimentConfig(qwen_fuse_reader_norms=True))
        actual = fused(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for source, layer in zip(baseline.model.layers, fused.model.layers):
        assert torch.equal(layer.input_layernorm.weight, torch.ones_like(layer.input_layernorm.weight))
        torch.testing.assert_close(
            layer.mlp.up_proj.weight,
            source.mlp.up_proj.weight * source.post_attention_layernorm.weight,
        )
    assert fused.faquant_reader_norm_fusion_stats["layers"] == 2


@pytest.mark.parametrize(
    ("rotation", "projections", "sharing", "fused"),
    [
        ("hisq_od", ("o_proj", "down_proj"), "linear", False),
        ("hisq_norm", (), "norm", False),
        ("hisq_global", (), "global", False),
        ("hisq_global_fused", (), "global", True),
        ("fused_od", ("o_proj", "down_proj"), "linear", True),
    ],
)
def test_reader_ablation_modes_change_one_factor_each(
    rotation: str, projections: tuple[str, ...], sharing: str, fused: bool
) -> None:
    from dataclasses import replace

    from faquant.qad import qwen3_hif4_qad_config

    config = qwen3_hif4_qad_config(rotation=rotation)
    assert config.rotation == "none" and config.qwen_hisq_input_rotation
    assert config.qwen_hisq_rotation_projections == projections
    assert config.qwen_hisq_rotation_sharing == sharing
    assert config.qwen_fuse_reader_norms is fused

    torch.manual_seed(13)
    model = _tiny_qwen3_two_layer_hif4()
    for decoder_layer in model.model.layers:
        decoder_layer.post_attention_layernorm.weight.data.uniform_(0.5, 2.0)
    calibration = torch.randint(0, 64, (2, 8))
    with torch.inference_mode():
        prepare_model(
            model, replace(config, qwen_hisq_rotation_block_size=16), calibration_input_ids=calibration
        )
        assert torch.isfinite(model(calibration[:1], use_cache=False).logits).all()
    layer = model.model.layers[0]
    assert (layer.self_attn.q_proj.input_rotation_signs is not None) is (not projections)
    assert layer.self_attn.o_proj.input_rotation_signs is not None
    assert bool((layer.post_attention_layernorm.weight == 1).all()) is fused


@pytest.mark.parametrize("weight_quant", ["gptq", "rtn"])
@pytest.mark.parametrize(
    ("rotation", "reader_signs", "reader_block", "od_plain", "od_block"),
    [
        ("sbh_readers", "global", 16, False, 16),
        ("bh_readers", "ones", 16, False, 16),
        ("bh64_readers", "ones", 64, False, 16),
        ("bh32_readers", "ones", 32, False, 16),
        ("bh_all", "ones", 16, True, 16),
        ("bh_all_r64", "ones", 64, True, 16),
        ("bh64_all", "ones", 64, True, 64),
    ],
)
def test_partial_hisq_constructions(
    rotation: str,
    reader_signs: str,
    reader_block: int,
    od_plain: bool,
    od_block: int,
    weight_quant: str,
) -> None:
    from dataclasses import replace

    from faquant.hisq_rotation import derive_hisq_rotation
    from faquant.qad import qwen3_hif4_qad_config

    torch.manual_seed(14)
    model = _tiny_qwen3_two_layer_hif4()
    calibration = torch.randint(0, 64, (2, 8))
    config = replace(
        qwen3_hif4_qad_config(rotation=rotation, hisq_block_size=16), weight_quant=weight_quant
    )
    with torch.inference_mode():
        prepare_model(model, config, calibration_input_ids=calibration)
        assert torch.isfinite(model(calibration[:1], use_cache=False).logits).all()

    identity = torch.arange(64)
    residual = derive_hisq_rotation("global_residual", 64, seed=config.seed, block_size=16)
    for layer in model.model.layers:
        for module in (layer.self_attn.q_proj, layer.self_attn.v_proj, layer.mlp.up_proj):
            assert torch.equal(module.input_rotation_permutation, identity)
            expected = residual.signs if reader_signs == "global" else torch.ones(64, dtype=torch.int8)
            assert torch.equal(module.input_rotation_signs, expected)
            assert module.input_rotation_block_size == reader_block
        for module in (layer.self_attn.o_proj, layer.mlp.down_proj):
            assert torch.equal(module.input_rotation_permutation, identity) is od_plain
            assert module.input_rotation_block_size == od_block


@pytest.mark.parametrize("weight_quant", ["gptq", "rtn"])
def test_rbh_keeps_only_the_randomization_that_folds_offline(weight_quant: str) -> None:
    from dataclasses import replace

    from faquant.hisq_rotation import derive_hisq_rotation
    from faquant.qad import qwen3_hif4_qad_config

    torch.manual_seed(15)
    model = _tiny_qwen3_two_layer_hif4()
    calibration = torch.randint(0, 64, (2, 8))
    config = replace(
        qwen3_hif4_qad_config(rotation="rbh_r64", hisq_block_size=16), weight_quant=weight_quant
    )
    with torch.inference_mode():
        prepare_model(model, config, calibration_input_ids=calibration)
        assert torch.isfinite(model(calibration[:1], use_cache=False).logits).all()

    residual = derive_hisq_rotation("global_residual", 64, seed=config.seed, block_size=64)
    for index, layer in enumerate(model.model.layers):
        for module in (layer.self_attn.q_proj, layer.mlp.gate_proj):
            assert torch.equal(module.input_rotation_signs, residual.signs)
            assert torch.equal(module.input_rotation_permutation, residual.permutation)
            assert module.input_rotation_block_size == 64
        own = {
            name: derive_hisq_rotation(
                f"model.layers.{index}.{name}", 64, seed=config.qwen_hisq_rotation_seed, block_size=16
            )
            for name in ("self_attn.o_proj", "mlp.down_proj")
        }
        o_proj, down_proj = layer.self_attn.o_proj, layer.mlp.down_proj
        assert torch.equal(o_proj.input_rotation_signs, own["self_attn.o_proj"].signs)
        assert torch.equal(o_proj.input_rotation_permutation, torch.arange(64))
        assert torch.equal(down_proj.input_rotation_signs, own["mlp.down_proj"].signs)
        assert torch.equal(down_proj.input_rotation_permutation, own["mlp.down_proj"].permutation)
        assert o_proj.input_rotation_block_size == down_proj.input_rotation_block_size == 16


@pytest.mark.parametrize("weight_quant", ["gptq", "rtn"])
@pytest.mark.parametrize(("rotation", "global_seed"), [("rbh_r64", 18), ("rbhn_r64", None)])
def test_rbh_reader_randomization_seeds_and_sharing(
    rotation: str, global_seed: int | None, weight_quant: str
) -> None:
    from dataclasses import replace

    from faquant.hisq_rotation import derive_hisq_rotation
    from faquant.qad import qwen3_hif4_qad_config

    torch.manual_seed(16)
    model = _tiny_qwen3_two_layer_hif4()
    calibration = torch.randint(0, 64, (2, 8))
    config = replace(
        qwen3_hif4_qad_config(rotation=rotation, hisq_block_size=16, hisq_global_seed=global_seed),
        weight_quant=weight_quant,
    )
    with torch.inference_mode():
        prepare_model(model, config, calibration_input_ids=calibration)
        assert torch.isfinite(model(calibration[:1], use_cache=False).logits).all()

    def expected(source: str, seed: int):
        rotation_ = derive_hisq_rotation(source, 64, seed=seed, block_size=64)
        return rotation_.signs, rotation_.permutation

    for index, layer in enumerate(model.model.layers):
        for norm, modules in (
            ("input_layernorm", (layer.self_attn.q_proj, layer.self_attn.k_proj)),
            ("post_attention_layernorm", (layer.mlp.gate_proj, layer.mlp.up_proj)),
        ):
            if global_seed is not None:
                signs, permutation = expected("global_residual", global_seed)
            else:
                signs, permutation = expected(f"model.layers.{index}.{norm}", config.qwen_hisq_rotation_seed)
            for module in modules:
                assert torch.equal(module.input_rotation_signs, signs)
                assert torch.equal(module.input_rotation_permutation, permutation)
                assert module.input_rotation_block_size == 64
        assert torch.equal(layer.self_attn.o_proj.input_rotation_permutation, torch.arange(64))


def test_reader_ablation_options_reject_conflicts() -> None:
    with pytest.raises(ValueError, match="already folds the RMSNorm"):
        ExperimentConfig(rotation="hadamard", qwen_fuse_reader_norms=True)
    with pytest.raises(ValueError, match="needs qwen_hisq_input_rotation"):
        ExperimentConfig(qwen_hisq_rotation_sharing="norm")
    with pytest.raises(ValueError, match="linear, norm or global"):
        ExperimentConfig(
            quant_target="all", quant_format="hif4", qwen_hisq_input_rotation=True,
            qwen_hisq_rotation_sharing="layer",
        )
    with pytest.raises(ValueError, match="full, signs or plain"):
        ExperimentConfig(qwen_hisq_reader_construction="hadamard")
    with pytest.raises(ValueError, match="positive power of two"):
        ExperimentConfig(qwen_hisq_reader_block_size=96)

