import json
from pathlib import Path

import pytest
import yaml

from prune_kit import capture, cli, probe, window, workflow
from prune_kit.facts import inspect

from test_capture import Server, data
from test_window import MACHINE, Fake, at


def flow(tmp_path, points="auto"):
    path = tmp_path / "flow.yml"
    path.write_text(f"name: w\ndata: {{sources: []}}\npoints: {points}\n")
    return workflow.load(path)


def test_the_plan_has_the_full_model_with_its_capture_flag_and_one_point_for_each_cut(qwen_like, tmp_path):
    facts, profile = inspect(qwen_like), probe.load_profile("qwen3_5_moe")
    plan = probe.plan(facts, flow(tmp_path, "[3, 2]"), profile, {"path": ["/opt/vllm/bin"]}, port=8999, tokens=5000)
    full, k3, k2 = plan["points"]
    run = f"runs/{qwen_like.name}/w/probe"
    assert [p["name"] for p in plan["points"]] == ["full", "k3", "k2"]
    assert full["health"] == "http://127.0.0.1:8999/health"
    assert full["boot"].startswith("env HF_HUB_OFFLINE=1") and "PATH=/opt/vllm/bin:$PATH vllm serve" in full["boot"]
    assert f"serve {qwen_like} --host 127.0.0.1 --port 8999 --served-model-name probe" in full["boot"]
    assert "--enable-return-routed-experts" in full["boot"] and "--enable-return-routed-experts" not in k3["boot"]
    assert full["boot"].endswith(f"> {run}/server-full.log 2>&1") and f"serve {run}/k3 " in k3["boot"]
    kinds = [c.split(" -m prune_kit.cli ")[1].split()[0:2] for c in full["clients"]]
    assert [k[0] for k in kinds] == ["capture", "gate", "gate", "rank", "build", "build"]
    assert "--tokens 5000" in full["clients"][0] and "--record" in full["clients"][1] and "--n 20" in full["clients"][1]
    assert f"--k 3 --draft keep --out {run}/k3" in full["clients"][4] and "--k 3 2" in full["clients"][3]
    assert k2["clients"] == [full["clients"][2].replace("score-full", "score-k2")]
    assert full["need_gib"] == k2["need_gib"] == 9.0     # the weights of the test are bytes; the overhead is 9 GiB


def test_the_cuts_come_from_the_workflow_or_from_the_default(qwen_like, tmp_path):
    facts = inspect(qwen_like)   # 4 experts in each layer, 2 per token
    assert probe.cuts(flow(tmp_path), facts) == [3] and probe.cuts(flow(tmp_path, "[2, 3]"), facts) == [3, 2]
    for points in ("[4]", "[1]"):
        with pytest.raises(probe.ProbeError, match="each cut must be below the 4 experts"):
            probe.cuts(flow(tmp_path, points), facts)
    with pytest.raises(probe.ProbeError, match="no profile for the model type new_moe.*qwen3_5_moe"):
        probe.load_profile("new_moe")


def test_the_probe_command_writes_a_window_file_that_the_dry_run_accepts(qwen_like, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flow.yml").write_text("name: w\ndata: {sources: []}\n")
    machine = Path(__file__).parent.parent / "examples/machine-spark.yml"
    args = ["probe", str(qwen_like), "--workflow", "flow.yml", "--machine", str(machine)]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    path = Path(f"runs/{qwen_like.name}/w/probe/window.yml")
    assert f"window    {path}" in out and "points    full (9 GiB), k3 (9 GiB)" in out and "--dry-run" in out
    assert [p["name"] for p in window.load_window(path)] == ["full", "k3"]
    assert cli.main(["window", str(path), "--machine", str(machine), "--stop", "qwen", "--dry-run", "--results", "r.json",
                     "--start-after", "2099-01-03 22:00", "--boot-by", "2099-01-04 01:00",
                     "--restore-at", "2099-01-04 03:00"]) == 0
    assert "prune_kit.cli build" in capsys.readouterr().out and not Path("r.json").exists()
    (path.parent / "capture.jsonl").write_text("")
    assert cli.main(args) == 2 and "holds the files of an earlier probe (capture.jsonl)" in capsys.readouterr().err


def test_a_window_writes_the_result_and_the_memory_of_each_point(tmp_path):
    fake = Fake(client_s=600)
    fake.mem = lambda clock: (40.0, 8.0) if clock < at(3) else (12.0, 8.0)
    points = [{"name": "full", "boot": "serve-full", "health": "test/health", "clients": ["score"]}]
    w = window.Window(MACHINE, ["qwen"], at(-1), at(60), at(120), fake, lambda line: None, tmp_path / "r.json")
    assert w.run(points) == 0
    assert json.loads((tmp_path / "r.json").read_text()) == {"restored": True, "points": [
        {"name": "full", "booted": True, "clients": [0], "available_before_gib": 40.0, "available_lowest_gib": 12.0}]}


def test_a_sample_capture_stops_at_its_tokens_and_mixes_the_sessions(tmp_path):
    sessions_ = [{**data()[0], "id": f"s{n}"} for n in range(6)]   # 2 segments each: 6 and 8 prompt tokens
    log = []
    capture.run("http://x", "m", sessions_, None, tmp_path / "c.jsonl", 6, post=Server(), log=log.append, max_tokens=20)
    rows = [json.loads(line) for line in (tmp_path / "c.jsonl").open()]
    assert 20 <= sum(r["prompt_tokens"] for r in rows) < 28 and len({r["session"] for r in rows}) > 1
    capture.run("http://x", "m", sessions_, None, tmp_path / "c.jsonl", 6, post=Server(), log=log.append, max_tokens=20)
    assert len((tmp_path / "c.jsonl").read_text().splitlines()) == len(rows)      # a second start adds nothing
    assert yaml.safe_load((probe.PROFILES / "qwen3_5_moe.yml").read_text())["capture_flag"].startswith("--enable")
