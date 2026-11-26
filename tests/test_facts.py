import pytest

from prune_moe_kit.facts import inspect
from prune_moe_kit.weights import WeightsError, tensors

from conftest import write_st


def test_per_expert_layout(qwen_like):
    f = inspect(qwen_like)
    assert (f.model_type, f.layout, f.is_moe) == ("qwen3_5_moe", "per_expert", True)
    assert f.experts == {0: 4, 1: 4} and f.per_token == 2
    assert f.expert_bytes == {0: 440, 1: 440}            # 4 experts x (100 + 10); the shared expert is not routed
    assert f.routed_bytes == 880
    assert f.total_bytes == 80 + 48 + 384 + 2 * (32 + 16 + 440)
    assert f.router_bias is False and f.loop is None
    assert f.quant == "compressed-tensors"
    assert [f.freed(k) for k in (4, 3, 2, 9)] == [0, 220, 440, 0]


def test_draft_in_the_same_file_keeps_its_own_expert_count(qwen_like):
    d = inspect(qwen_like).draft
    assert (d.bytes, d.experts, d.own_file) == (48 + 384, 6, False)


def test_stacked_layout(inkling_like):
    f = inspect(inkling_like)
    assert (f.model_type, f.layout) == ("inkling2_loop", "stacked")
    assert f.experts == {0: 8, 1: 8, 2: 8} and f.per_token == 2
    assert f.expert_bytes == {0: 648, 1: 648, 2: 648}    # a tensor with another first dimension is not sliced
    assert f.router_bias is True
    assert f.loop == (1, 2, 3)
    assert f.quant == "NVFP4"
    assert f.freed(6) == 3 * 2 * 81


def test_dense_draft_in_its_own_file(inkling_like):
    d = inspect(inkling_like).draft
    assert (d.bytes, d.experts, d.own_file) == (32, None, True)


def test_freed_bytes_with_a_different_count_in_each_layer(tmp_path):
    t = {f"layers.{layer}.ffn.experts.{e}.w1.rank0.trellis": ("U8", (10,), 10)
         for layer, n in ((0, 4), (1, 2)) for e in range(n)}
    write_st(tmp_path / "exl3-layer.safetensors", t)
    f = inspect(tmp_path)
    assert f.experts == {0: 4, 1: 2} and f.max_experts == 4 and f.model_type == "unknown"
    assert f.freed(3) == 10 and f.freed(2) == 20 and f.freed(1) == 40


def test_dense_model_has_no_routed_experts(tmp_path):
    write_st(tmp_path / "model.safetensors", {"model.layers.0.mlp.gate_proj.weight": ("BF16", (4, 4), 32)})
    f = inspect(tmp_path)
    assert (f.is_moe, f.layout, f.routed_bytes, f.max_experts, f.draft) == (False, "none", 0, 0, None)


def test_folder_with_no_weight_file(tmp_path):
    with pytest.raises(WeightsError, match="no safetensors file"):
        inspect(tmp_path)


def test_file_that_is_not_safetensors(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\xff" * 64)
    with pytest.raises(WeightsError, match="not a safetensors file"):
        tensors(tmp_path)
