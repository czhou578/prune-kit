import json

from prune_moe_kit import cli, preflight
from prune_moe_kit.facts import inspect

from conftest import write_st


def states(folder, available=None):
    return {c.name: c.state for c in preflight.checks(inspect(folder), available)}


def test_default_cuts():
    assert preflight.default_ks(256) == [240, 224, 208, 192, 176, 160]
    assert preflight.default_ks(216) == [208, 192, 176, 160, 144]
    assert preflight.default_ks(128) == [120, 112, 104, 96, 88, 80]
    assert preflight.default_ks(8) == [7, 6, 5]
    assert preflight.default_ks(0) == []


def test_known_model_type_needs_only_the_prune_tool(qwen_like):
    assert states(qwen_like) == {"routed experts": "yes", "prune tool": "yes", "capture": "yes", "mask": "info",
                                 "draft model": "yes"}
    r = preflight.report(inspect(qwen_like))
    assert r["verdict"] == "GO" and "prune-moe-kit build: it gives the folder of the prune by hand" in preflight.text(r)


def test_a_model_type_that_ran_through_prune_moe_kit_is_proven(inkling_like):
    r = preflight.report(inspect(inkling_like), available=10)
    out = preflight.text(r)
    assert r["verdict"] == "GO" and "a server loaded K 120" in out and "one row for each pass" in out


def test_new_model_type_lists_each_missing_part(inkling_like):
    config = json.loads((inkling_like / "config.json").read_text())
    (inkling_like / "config.json").write_text(json.dumps(config | {"model_type": "new_loop"}))
    s = states(inkling_like, available=10)
    assert s == {"routed experts": "yes", "prune tool": "unknown", "capture": "unknown", "mask": "info",
                 "draft model": "info", "loop": "info", "memory": "info"}
    r = preflight.report(inspect(inkling_like), available=10)
    assert r["verdict"] == "GO WITH WORK: prune tool, capture"
    out = preflight.text(r)
    assert "layers 1 to 1 run 3 times" in out and "needs a window" in out and "the router has a bias" in out


def test_moe_draft_of_a_new_model_type_is_work(qwen_like):
    config = json.loads((qwen_like / "config.json").read_text())
    (qwen_like / "config.json").write_text(json.dumps(config | {"model_type": "new_moe"}))
    assert states(qwen_like)["draft model"] == "unknown"
    assert preflight.report(inspect(qwen_like))["verdict"] == "GO WITH WORK: prune tool, capture, draft model"


def test_memory_that_fits(qwen_like):
    note = next(c.note for c in preflight.checks(inspect(qwen_like), 1 << 40) if c.name == "memory")
    assert "fits now" in note


def test_cuts_of_the_report(qwen_like):
    r = preflight.report(inspect(qwen_like), ks=[3, 2])
    assert [(c["k"], c["freed_bytes"]) for c in r["cuts"]] == [(3, 220), (2, 440)]
    assert r["cuts"][0]["share"] == 220 / r["total_bytes"]
    assert (r["layers"], r["experts_min"], r["experts_max"], r["experts_total"], r["per_token"]) == (2, 4, 4, 8, 2)
    assert "K 3" in preflight.text(r) and "110.00" not in preflight.text(r)


def test_dense_model_is_no_go(tmp_path, capsys):
    write_st(tmp_path / "model.safetensors", {"model.layers.0.mlp.gate_proj.weight": ("BF16", (4, 4), 32)})
    assert cli.main(["preflight", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "verdict   NO-GO" in out and "dense" in out and "cut " not in out


def test_command_prints_text_and_json(qwen_like, capsys):
    assert cli.main(["preflight", str(qwen_like), "--k", "3"]) == 0
    assert "K 3" in capsys.readouterr().out
    assert cli.main(["preflight", str(qwen_like), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["moe"] is True and r["verdict"] == "GO" and r["checks"][0]["state"] == "yes"


def test_command_with_a_model_that_cannot_be_read(tmp_path, capsys):
    assert cli.main(["preflight", str(tmp_path / "missing")]) == 2
    assert cli.main(["preflight", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "no folder with this path" in err and "no safetensors file" in err
