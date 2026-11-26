"""Weight files with no read permission, the EXL3 facts, GGUF folders, and dense models."""

import json
import os

import pytest

from prune_moe_kit import cli, preflight
from prune_moe_kit.facts import inspect
from prune_moe_kit.weights import FormatError, WeightsError, tensors

from conftest import write_st

needs_user = pytest.mark.skipif(os.geteuid() == 0, reason="root reads each file")


@pytest.fixture
def locked(tmp_path):
    """One layer file of 4 experts with 3 tensors each and no read permission, an open file, and the index."""
    layer = {f"layers.0.ffn.experts.{e}.w{w}.rank0.trellis": ("U8", (10,), 10) for e in range(4) for w in (1, 2, 3)}
    carried = {"embed.weight": ("BF16", (4, 4), 32), "layers.0.ffn.gate.bias": ("F32", (4,), 16)}
    write_st(tmp_path / "exl3-layer-000.safetensors", layer)
    write_st(tmp_path / "carried-001.safetensors", carried)
    index = {"weight_map": {**dict.fromkeys(layer, "exl3-layer-000.safetensors"),
                            **dict.fromkeys(carried, "carried-001.safetensors")}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "num_hash_layers": 3}))
    (tmp_path / "quantization_config.json").write_text(json.dumps({"quant_method": "exl3"}))
    (tmp_path / "exl3-layer-000.safetensors").chmod(0)
    yield tmp_path
    (tmp_path / "exl3-layer-000.safetensors").chmod(0o600)


@needs_user
def test_a_locked_file_gives_the_names_of_the_index_and_shares_of_its_size(locked):
    size = (locked / "exl3-layer-000.safetensors").stat().st_size
    f = inspect(locked)
    assert (f.layout, f.experts, f.unread_files) == ("per_expert", {0: 4}, 1)
    assert f.expert_bytes == {0: size // 12 * 12} and f.total_bytes == size // 12 * 12 + 48
    assert f.freed(3) == round(size // 12 * 12 / 4)
    assert (f.quant, f.hash_layers, f.router_bias) == ("exl3", 3, True)
    notes = {c.name: c.note for c in preflight.checks(f, None)}
    assert "1 weight files have no read permission" in notes["sizes"]
    assert "layers 0 to 2 route by a token table" in notes["hash routing"]
    assert "by hand: tools/prune_experts.py" in notes["prune tool"]


@needs_user
def test_a_locked_file_that_the_index_does_not_list(locked):
    (locked / "model.safetensors.index.json").unlink()
    with pytest.raises(WeightsError, match="the index does not list it"):
        tensors(locked)


@needs_user
def test_a_locked_file_with_a_stacked_layout(tmp_path):
    write_st(tmp_path / "model.safetensors", {"layers.0.mlp.experts.w2_weight": ("U8", (4, 8), 32)})
    index = {"weight_map": {"layers.0.mlp.experts.w2_weight": "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model.safetensors").chmod(0)
    with pytest.raises(WeightsError, match="a stacked layout needs the shapes"):
        inspect(tmp_path)
    (tmp_path / "model.safetensors").chmod(0o600)


def test_gguf_folder_is_no_go(tmp_path, capsys):
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    with pytest.raises(FormatError):
        tensors(tmp_path)
    assert cli.main(["preflight", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "GGUF files" in out and "verdict   NO-GO" in out


def test_dense_model_lists_the_other_levers(tmp_path):
    write_st(tmp_path / "model.safetensors", {"model.layers.0.mlp.gate_proj.weight": ("BF16", (4, 4), 32)})
    out = preflight.text(preflight.report(inspect(tmp_path)))
    assert "other levers" in out and all(name in out for name, _ in preflight.LEVERS)


def test_moe_model_does_not_list_the_other_levers(qwen_like):
    assert "other levers" not in preflight.text(preflight.report(inspect(qwen_like)))
