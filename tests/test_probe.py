import json
from pathlib import Path

import pytest
import yaml

from prune_kit import capture, cli, probe, rank, window, workflow
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


def test_each_test_server_gets_the_flags_and_the_variables_of_the_command(qwen_like, tmp_path):
    facts, profile = inspect(qwen_like), probe.load_profile("qwen3_5_moe")
    plan = probe.plan(facts, flow(tmp_path, "[3]"), profile, {}, more_flags="--speculative-config '{\"a\": 1}'",
                      more_env={"PYTHONPATH": "/a patch", "HF_HUB_OFFLINE": 0})
    for point in plan["points"]:
        assert "--speculative-config '{\"a\": 1}' > runs/" in point["boot"]
        assert point["boot"].startswith("env HF_HUB_OFFLINE=0 ")
        assert " PYTHONPATH='/a patch' vllm serve" in point["boot"]
    assert "--enable-return-routed-experts --speculative-config" in plan["points"][0]["boot"]


def test_a_profile_with_a_container_gets_absolute_paths_its_mounts_and_a_down_command(qwen_like, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    facts, profile = inspect(qwen_like), probe.load_profile("inkling2_loop")
    full, k3 = probe.plan(facts, flow(tmp_path, "[3]"), profile, {})["points"]
    model, run = qwen_like.resolve(), (tmp_path / "runs" / qwen_like.name / "w/probe").resolve()
    for point, served in ((full, model), (k3, run / "k3")):
        assert point["down"] == "docker rm -f prune-kit-probe"
        assert f" -v {model}:{model}:ro -v {run}:{run}:ro " in point["boot"]
        assert f" {served} --host 127.0.0.1 --port 8890 --served-model-name probe " in point["boot"]
        assert """--limit-mm-per-prompt '{"image": 0}'""" in point["boot"]
    assert all(f"--served probe --profile {facts.model_type} " in c for c in full["clients"][1:3] + k3["clients"])


def test_the_root_of_a_folder_with_links_holds_the_files_that_the_links_name(tmp_path):
    (tmp_path / "repo/blobs").mkdir(parents=True)
    (tmp_path / "repo/snapshots/a").mkdir(parents=True)
    (tmp_path / "repo/blobs/x").write_text("x")
    (tmp_path / "repo/snapshots/a/config.json").symlink_to("../../blobs/x")
    assert probe.root(tmp_path / "repo/snapshots/a") == (tmp_path / "repo").resolve()
    assert probe.root(tmp_path / "repo/blobs") == (tmp_path / "repo/blobs").resolve()


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
    times = ["--start-after", "2099-01-03 22:00", "--boot-by", "2099-01-04 01:00", "--restore-at", "2099-01-04 03:00"]
    assert cli.main(["window", str(path), "--machine", str(machine), "--stop", "qwen", "--dry-run",
                     "--results", "r.json", *times]) == 0
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


RECORD = Path.home() / "Projects/dsv4-specialize/results"


def scores(path, nll, turns=6):
    """A score file: each turn has 100 tokens, and the NLL grows a little from turn to turn."""
    rows = [{"session": "s", "segment": 0, "index": n, "tokens": 100, "nll": nll * (1 + n / 100), "top1": 90,
             "decision_ok": True, "tool_calls": 2, "tool_names_ok": 2} for n in range(turns)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def ran(qwen_like, tmp_path, costs):
    """The files of a probe that ran on the test model (4 experts in each layer). ``costs``: ``{k: NLL}``."""
    run = tmp_path / "probe"
    run.mkdir()
    counts = [[70, 20, 8, 2], [2, 8, 20, 70]]
    rows = [{"session": "s", "segment": n, "prompt_tokens": 50, "unfilled_rows": 0, "counts": counts} for n in range(4)]
    (run / "capture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "rank.json").write_text(json.dumps(rank.rank(rows)))
    scores(run / "score-full.jsonl", 30.0)
    for k, nll in costs.items():
        scores(run / f"score-k{k}.jsonl", nll)
    return run


def test_the_report_gives_the_cost_of_each_cut_and_the_estimate_of_the_line(qwen_like, tmp_path):
    facts, w = inspect(qwen_like), flow(tmp_path)
    run = ran(qwen_like, tmp_path, {3: 30.3, 2: 31.5})     # +1% and +5%, for 2% and 10% of removed routing
    (run / "window.json").write_text(json.dumps({"points": [
        {"name": "full", "available_before_gib": 34.0, "available_lowest_gib": 5.5}]}))
    r = probe.report(facts, w, run)
    k3, k2 = r["points"]
    assert (k3["k"], k3["freed_bytes"], k2["freed_bytes"]) == (3, 220, 440) and k3["overlap"] == 1.0
    assert k3["removed_pct"] == pytest.approx(2.0) and k3["nll_change_pct"] == pytest.approx(1.0)
    assert k3["band"] == pytest.approx([1.0, 1.0]) and (k3["nll_gate"], k2["nll_gate"]) == ("pass", "fail")
    assert r["slope"] == pytest.approx(52 / 104) and r["estimate_k"] == 3 and r["full_peak_gib"] == 28.5
    out = probe.text(r)
    assert "4 segments, 200 prompt tokens" in out and "the nll gate holds down to about K 3" in out
    assert "+0.50% NLL for each 1% of removed routing" in out and "the full model used 28.5 GiB at the peak" in out


def test_a_band_that_holds_the_limit_is_near_the_limit(qwen_like, tmp_path):
    run = ran(qwen_like, tmp_path, {3: 30.9})
    rows = [json.loads(line) for line in (run / "score-k3.jsonl").open()]
    for n, r in enumerate(rows):
        r["nll"] *= 1.3 if n % 2 else 0.75             # the turns do not agree on the cost
    (run / "score-k3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    point = probe.report(inspect(qwen_like), flow(tmp_path), run)["points"][0]
    assert point["band"][0] < 3 < point["band"][1] and point["nll_gate"] == "near the limit"


def test_a_probe_with_no_reference_or_with_no_cut(qwen_like, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flow.yml").write_text("name: w\ndata: {sources: []}\n")
    args = ["probe", str(qwen_like), "--workflow", "flow.yml"]
    assert cli.main([*args, "--report"]) == 2 and "the probe did not run to its reference" in capsys.readouterr().err
    assert cli.main(args) == 2 and "needs --machine" in capsys.readouterr().err
    ran(qwen_like, tmp_path, {}).rename(tmp_path / "moved")
    Path(f"runs/{qwen_like.name}/w").mkdir(parents=True)
    (tmp_path / "moved").rename(f"runs/{qwen_like.name}/w/probe")
    assert cli.main([*args, "--report"]) == 0 and "no cut has a score" in capsys.readouterr().out
    assert cli.main([*args, "--report", "--json"]) == 0 and json.loads(capsys.readouterr().out)["points"] == []


@pytest.mark.skipif(not (RECORD / "qwen-fit-g208-score.jsonl").exists(), reason="the record is not on this machine")
def test_the_report_of_the_recorded_gate_files_gives_the_numbers_of_the_record(tmp_path):
    from prune_kit.resolve import ModelNotFound, resolve
    try:
        facts = inspect(resolve("unsloth/Qwen3.6-35B-A3B-NVFP4-Fast", []))
    except ModelNotFound:
        pytest.skip("the full Qwen is not on this machine")
    files = {"capture.jsonl": "qwen-capture.jsonl", "rank.json": "qwen-rank.json",
             "score-full.jsonl": "qwen-fit-g256-score.jsonl",
             "score-k208.jsonl": "qwen-fit-g208-score.jsonl", "score-k192.jsonl": "qwen-fit-g192-score.jsonl"}
    for name, source in files.items():
        (tmp_path / name).symlink_to(RECORD / source)
    r = probe.report(facts, workflow.load(Path(__file__).parent.parent / "examples/qwen-prune-record.yml"), tmp_path)
    assert [round(p["nll_change_pct"], 2) for p in r["points"]] == [4.15, 8.04]
    assert [p["nll_gate"] for p in r["points"]] == ["near the limit", "fail"] and 216 <= r["estimate_k"] <= 228
