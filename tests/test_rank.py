import json
from pathlib import Path

import pytest

from prune_kit import capture, cli, rank, sessions

from conftest import write_st
from test_capture import Server, data

RECORD = Path.home() / "Projects/dsv4-specialize/results"


def row(session, segment, counts, **more):
    return {"session": session, "segment": segment, "prompt_tokens": 10, "unfilled_rows": 1, "counts": counts, **more}


def test_rank_orders_each_layer_by_the_count():
    rows = [row("a", 0, [[5, 0, 1, 1], [0, 2, 2, 9]]),
            row("a", 1, [[1, 0, 3, 1], [1, 0, 0, 0]], gen_counts=[[0, 9, 0, 0], [0, 0, 0, 0]]),
            {"session": "b", "segment": 0, "error": "OSError"}]
    r = rank.rank(rows)
    assert (r["segments"], r["prompt_tokens"], r["unfilled_rows"], r["order"]) == (2, 20, 2, "count")
    assert [e["id"] for e in r["layers"]["0"]] == [1, 0, 2, 3]            # counts 9, 6, 4, 2
    assert [e["id"] for e in r["layers"]["1"]] == [3, 1, 2, 0]            # equal counts keep the order of the ids
    assert r["layers"]["0"][0] == {"id": 1, "count": 9, "share": round(9 / 21, 6)}
    assert rank.removed(r, 2) == pytest.approx((6 / 21 + 3 / 14) / 2) and rank.removed(r, 4) == 0


def test_the_rank_of_a_model_with_a_loop_folds_the_passes_of_a_layer():
    # 4 layers, and the layers 1 and 2 run 2 times: the server gives the rows 0, 1, 2, 1, 2, 3
    counts = [[1, 0], [2, 0], [0, 3], [0, 4], [5, 0], [0, 6]]
    r = rank.rank([row("a", 0, counts, loop=[1, 3, 2])])
    by_id = lambda layer: [e["count"] for e in sorted(r["layers"][layer], key=lambda e: e["id"])]
    assert [by_id(layer) for layer in r["layers"]] == [[1, 0], [2, 4], [5, 3], [0, 6]]
    with pytest.raises(rank.RankError, match="fewer than the 2 passes"):
        rank.rank([row("a", 0, counts[:4], loop=[1, 3, 2])])


def test_rank_refuses_a_capture_with_no_counts_or_with_mixed_shapes():
    with pytest.raises(rank.RankError, match="no segment with counts"):
        rank.rank([{"session": "b", "segment": 0, "error": "x"}])
    with pytest.raises(rank.RankError, match="segment b:1 has another count"):
        rank.rank([row("a", 0, [[1, 2]]), row("b", 1, [[1, 2, 3]])])


def test_overlap_is_one_for_equal_halves_and_low_for_unlike_halves():
    same = [row("s", n, [[9, 8, 1, 0]]) for n in range(6)]
    assert rank.overlap(same, [2, 3]) == {2: 1.0, 3: 1.0}
    a, b = rank.halves(same)
    assert len(a) == len(b) == 3 and {r["segment"] for r in a + b} == set(range(6))
    unlike = [{**r, "counts": [[9, 8, 1, 0]] if r in a else [[0, 1, 8, 9]]} for r in same]
    assert rank.overlap(unlike, [2]) == {2: 0.0}


def test_capture_and_rank_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "models--org--name/snapshots/abc"
    model.mkdir(parents=True)
    write_st(model / "model.safetensors", {f"model.layers.0.mlp.experts.{e}.w": ("U8", (4,), 4) for e in range(6)})
    (model / "config.json").write_text(json.dumps({"model_type": "m", "num_experts_per_tok": 2}))
    sessions.save(tmp_path / "d", data()[:1])
    (tmp_path / "d/tools.json").write_text(json.dumps({"tools": [{"type": "function"}]}))
    (tmp_path / "flow.yml").write_text(f"name: w\ndata: {{sources: [], folder: {tmp_path / 'd'}}}\n")
    server = Server()
    monkeypatch.setattr(capture, "post", server)
    args = ["capture", str(model), "--workflow", "flow.yml", "--base-url", "http://x", "--served", "m"]
    assert cli.main([*args, "--body", '{"reasoning_effort": "low"}']) == 0
    out = tmp_path / "runs/org--name/w/capture.jsonl"
    assert f"capture   {out.relative_to(tmp_path)}" in capsys.readouterr().out
    assert len(out.read_text().splitlines()) == 2
    assert server.bodies[0][1]["tools"] == [{"type": "function"}] and server.bodies[0][1]["reasoning_effort"] == "low"
    assert cli.main(["capture", "none", "--workflow", "flow.yml", "--base-url", "http://x"]) == 2
    assert cli.main(["rank", str(out), "--k", "4", "2"]) == 0
    text = capsys.readouterr().out
    assert "2 segments, 14 prompt tokens" in text and "6 per layer, in 1 layers" in text and "K 4 " in text
    assert [e["id"] for e in json.loads((out.parent / "rank.json").read_text())["layers"]["0"]][0] == 5
    assert cli.main(["rank", str(tmp_path / "none.jsonl")]) == 2


@pytest.mark.skipif(not (RECORD / "qwen-capture.jsonl").exists(), reason="the capture of the Qwen prune is not here")
def test_the_rank_of_the_qwen_capture_is_the_rank_of_the_record():
    with (RECORD / "qwen-capture.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    new, old = rank.rank(rows), json.loads((RECORD / "qwen-rank.json").read_text())
    assert (new["segments"], new["prompt_tokens"]) == (old["segments"], old["prompt_tokens"])
    for layer, experts in old["layers"].items():
        assert [(e["id"], e["count"]) for e in new["layers"][layer]] == [(e["id"], e["count"]) for e in experts]
    assert rank.removed(new, 208) == pytest.approx(0.0374, abs=0.0002)
