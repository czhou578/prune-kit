import json
import os
import struct

import pytest

from prune_kit import cli, st_moe


def store(path, tensors):
    """Write a safetensors file. ``tensors`` is ``{name: (shape, bytes)}``."""
    head, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (shape, data) in tensors.items():
        head[name] = {"dtype": "U8", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    raw = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(data for _, data in tensors.values()))


def read(path):
    """``{name: (shape, bytes)}`` of a safetensors file."""
    head, start = st_moe.header(path)
    raw = path.read_bytes()
    return {name: (h["shape"], raw[start + h["data_offsets"][0]:start + h["data_offsets"][1]])
            for name, h in head.items() if name != "__metadata__"}


def rows(*ids, width=3):
    """One row of ``width`` bytes for each id: the id, repeated."""
    return b"".join(bytes([i]) * width for i in ids)


@pytest.fixture
def source(tmp_path):
    """2 layers with 4 experts each in the per-expert layout, and a draft layer with 4 stacked experts."""
    folder = tmp_path / "full"
    folder.mkdir()
    t = {"model.embed.weight": ((2, 4), os.urandom(8))}
    for layer in range(2):
        p = f"model.layers.{layer}.mlp."
        t[p + "gate.weight"] = ((4, 3), rows(0, 1, 2, 3))
        t[p + "shared_expert.down_proj.weight"] = ((4, 2), os.urandom(8))   # 4 rows, and not a tensor of the router
        for e in range(4):
            t[f"{p}experts.{e}.down_proj.weight"] = ((5,), bytes([10 * layer + e]) * 5)
    store(folder / "model-00001-of-00002.safetensors", t)
    store(folder / "model-00002-of-00002.safetensors", {
        "mtp.layers.0.mlp.gate.weight": ((4, 3), rows(0, 1, 2, 3)),
        "mtp.layers.0.mlp.experts.down_proj": ((4, 2, 3), rows(0, 1, 2, 3, width=6)),
        "mtp.norm.weight": ((4,), os.urandom(4))})
    (folder / "config.json").write_text(json.dumps({"model_type": "m", "text_config": {"num_experts": 4, "n": 4}}))
    (folder / "tokenizer.json").write_text("{}")
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": {}}))
    return folder


RANK = {"layers": {"0": [{"id": 3}, {"id": 1}, {"id": 0}, {"id": 2}], "1": [{"id": 2}, {"id": 0}, {"id": 1}, {"id": 3}],
                   "2": [{"id": 1}, {"id": 3}, {"id": 0}, {"id": 2}]}}


def test_prune_keeps_the_ranked_experts_with_new_ids(source, tmp_path):
    before = {f.name: f.read_bytes() for f in source.iterdir()}
    out = tmp_path / "k2"
    assert st_moe.prune(source, out, RANK, 2) == {"kept": 12, "dropped": 4, "bytes": 8 + 2 * (6 + 8 + 10) + 6 + 12 + 4}
    got = read(out / "model-00001-of-00002.safetensors")
    assert got["model.layers.0.mlp.experts.0.down_proj.weight"][1] == bytes([1]) * 5    # the old expert 1
    assert got["model.layers.0.mlp.experts.1.down_proj.weight"][1] == bytes([3]) * 5    # the old expert 3
    assert got["model.layers.1.mlp.experts.1.down_proj.weight"][1] == bytes([12]) * 5
    assert "model.layers.0.mlp.experts.2.down_proj.weight" not in got
    assert got["model.layers.0.mlp.gate.weight"] == ([2, 3], rows(1, 3))
    assert got["model.layers.1.mlp.gate.weight"][1] == rows(0, 2)
    assert got["model.layers.0.mlp.shared_expert.down_proj.weight"][0] == [4, 2]
    draft = read(out / "model-00002-of-00002.safetensors")
    assert draft["mtp.layers.0.mlp.experts.down_proj"] == ([2, 2, 3], rows(1, 3, width=6))
    assert draft["mtp.layers.0.mlp.gate.weight"][1] == rows(1, 3) and draft["mtp.norm.weight"][0] == [4]
    assert json.loads((out / "config.json").read_text())["text_config"] == {"num_experts": 2, "n": 4}
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["metadata"] == {"total_size": 1} and len(index["weight_map"]) == 12
    assert index["weight_map"]["mtp.norm.weight"] == "model-00002-of-00002.safetensors"
    assert (out / "tokenizer.json").exists()
    assert json.loads((out / "PRUNE.json").read_text())["kept_ids"]["mtp.0"] == [1, 3]
    assert st_moe.check(source, out) == 12
    assert before == {f.name: f.read_bytes() for f in source.iterdir()}   # the source did not change


@pytest.mark.parametrize("mode, shape, data", [("first", [2, 2, 3], rows(0, 1, width=6)),
                                               ("keep", [4, 2, 3], rows(0, 1, 2, 3, width=6))])
def test_the_draft_modes_need_no_rank_of_the_draft(source, tmp_path, mode, shape, data):
    rank = {"layers": {k: v for k, v in RANK["layers"].items() if k != "2"}}
    with pytest.raises(st_moe.PruneError, match="no layer 2 for the draft model"):
        st_moe.prune(source, tmp_path / "x", rank, 2)
    st_moe.prune(source, tmp_path / "out", rank, 2, draft=mode)
    got = read(tmp_path / "out/model-00002-of-00002.safetensors")
    assert got["mtp.layers.0.mlp.experts.down_proj"] == (shape, data)
    assert st_moe.check(source, tmp_path / "out") == 12


def test_the_stacked_layout_keeps_rows_and_a_tensor_with_one_value(tmp_path):
    folder = tmp_path / "full"
    folder.mkdir()
    store(folder / "model.safetensors", {
        "model.llm.layers.0.mlp.experts.w13_weight": ((4, 2), rows(0, 1, 2, 3, width=2)),
        "model.llm.layers.0.mlp.experts.w13_weight_amax": ((4,), bytes([0, 1, 2, 3])),
        "model.llm.layers.0.mlp.experts.w13_scale": ((1,), b"\x07"),
        "model.llm.layers.0.mlp.gate.weight": ((4, 2), rows(0, 1, 2, 3, width=2)),
        "model.llm.layers.0.mlp.gate.bias": ((4,), bytes([0, 1, 2, 3]))})
    (folder / "config.json").write_text(json.dumps({"text_config": {"n_routed_experts": 4}}))
    (folder / "tiktoken").mkdir()
    (folder / "tiktoken/tokenizer.model").write_text("the tokenizer of some models is in a folder")
    st_moe.prune(folder, tmp_path / "out", {"layers": {"0": [{"id": 3}, {"id": 0}, {"id": 2}, {"id": 1}]}}, 3)
    assert (tmp_path / "out/tiktoken/tokenizer.model").read_text().startswith("the tokenizer")
    got = read(tmp_path / "out/model.safetensors")
    assert got["model.llm.layers.0.mlp.experts.w13_weight"] == ([3, 2], rows(0, 2, 3, width=2))
    assert got["model.llm.layers.0.mlp.experts.w13_weight_amax"] == ([3], bytes([0, 2, 3]))
    assert got["model.llm.layers.0.mlp.experts.w13_scale"] == ([1], b"\x07")
    assert got["model.llm.layers.0.mlp.gate.bias"] == ([3], bytes([0, 2, 3]))
    assert json.loads((tmp_path / "out/config.json").read_text())["text_config"]["n_routed_experts"] == 3
    assert st_moe.check(folder, tmp_path / "out") == 5


def test_check_finds_a_changed_byte_a_lost_tensor_and_a_wrong_config(source, tmp_path):
    out = tmp_path / "k2"
    st_moe.prune(source, out, RANK, 2)
    path = out / "model-00001-of-00002.safetensors"
    good = path.read_bytes()
    path.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(st_moe.PruneError, match="differs from its source"):
        st_moe.check(source, out)
    store(path, {name: value for name, value in list(read(source / path.name).items())[:3]})
    with pytest.raises(st_moe.PruneError, match="the tensor count differs"):
        st_moe.check(source, out)
    path.write_bytes(good)
    (out / "config.json").write_text(json.dumps({"text_config": {"num_experts": 4}}))
    with pytest.raises(st_moe.PruneError, match="does not have the expert count 2"):
        st_moe.check(source, out)


def test_prune_refuses_a_wrong_k_a_short_rank_and_a_folder_that_exists(source, tmp_path):
    for k in (0, 4):
        with pytest.raises(st_moe.PruneError, match="K must be between 1 and 3"):
            st_moe.prune(source, tmp_path / "x", RANK, k)
    short = {"layers": {**RANK["layers"], "1": [{"id": 2}, {"id": 2}, {"id": 9}]}}
    with pytest.raises(st_moe.PruneError, match="layer 1: the rank file does not give 2 experts"):
        st_moe.prune(source, tmp_path / "x", short, 2)
    (source / "config.json").write_text(json.dumps({"num_experts": 8}))
    with pytest.raises(st_moe.PruneError, match="config.json has no expert count of 4"):
        st_moe.prune(source, tmp_path / "x", RANK, 2)
    assert not (tmp_path / "x").exists()
    (source / "config.json").write_text(json.dumps({"num_experts": 4}))
    st_moe.prune(source, tmp_path / "x", RANK, 2)
    with pytest.raises(FileExistsError):
        st_moe.prune(source, tmp_path / "x", RANK, 2)


def test_build_command_prunes_and_checks(source, tmp_path, capsys):
    rank, out = tmp_path / "rank.json", tmp_path / "k2"
    rank.write_text(json.dumps(RANK))
    assert cli.main(["build", str(source), "--rank", str(rank), "--k", "2", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "kept 12 tensors, dropped 4" in text and "check: 12 tensors are equal to their source" in text
    assert cli.main(["build", str(source), "--out", str(out), "--check"]) == 0
    assert cli.main(["build", str(source), "--out", str(tmp_path / "x")]) == 2
    assert "a build needs --rank and --k" in capsys.readouterr().err
    assert cli.main(["build", str(source), "--rank", str(rank), "--k", "9", "--out", str(tmp_path / "x")]) == 2
    assert cli.main(["build", str(source), "--rank", str(rank), "--k", "2", "--out", str(out)]) == 2   # it exists
