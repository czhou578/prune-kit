import json

import pytest

from prune_moe_kit import cli, gates, probe
from prune_moe_kit.facts import inspect

from test_probe import flow, ran


def events(tokens, pieces):
    """A stream of ``pieces`` deltas and one event with the usage."""
    out = [{"choices": [{"delta": {"role": "assistant"}}]}]
    out += [{"choices": [{"delta": {"reasoning_content" if n == 0 else "content": "x"}}]} for n in range(pieces)]
    return [*out, {"choices": [], "usage": {"completion_tokens": tokens}}]


def test_the_speed_is_the_median_of_the_prompts(tmp_path):
    ticks, bodies = iter(range(100)), []

    def stream(base, body):
        bodies.append(body)
        return events({0: 41, 1: 21}.get(len(bodies) - 1, 0), 5)    # each later prompt gives no tokens

    r = gates.measure_speed("http://x", "m", tmp_path / "speed.json", stream=stream, clock=lambda: next(ticks))
    assert [run["speed"] for run in r["runs"]] == [10.0, 5.0] and r["tokens_per_second"] == 7.5   # 4 s for each prompt
    assert json.loads((tmp_path / "speed.json").read_text())["tokens_per_second"] == 7.5
    assert bodies[0]["stream"] and bodies[0]["stream_options"] == {"include_usage": True} and len(bodies) == 8
    assert [run["prompt"] for run in r["runs"]] == [0, 1]
    slow = {"runs": [{"prompt": 0, "speed": 9.0}, {"prompt": 1, "speed": 5.5}, {"prompt": 5, "speed": 1.0}]}
    assert gates.speed_change(r, slow) == 0.0 and gates.speed_change(None, slow) is None   # -10% and +10%


def test_the_tasks_gate_and_the_memory_gate():
    suite = lambda n: {"total_tasks": 85, "total_correct": n}
    assert gates.tasks(suite(50), suite(47), 3) == {"reference": 50, "new": 47, "of": 85, "ok": True}
    assert gates.tasks(suite(50), suite(46), 3)["ok"] is False and gates.tasks(None, suite(46), 3) is None
    window = {"points": [{"name": "k3", "booted": True, "available_before_gib": 34.0, "available_lowest_gib": 6.0},
                         {"name": "k2", "booted": False, "available_before_gib": 34.0, "available_lowest_gib": 34.0}]}
    assert gates.memory(window, "k3", 26.5) == {"peak_gib": 28.0, "goal_gib": 26.5, "ok": False}
    assert gates.memory(window, "k3", None)["ok"] is None and gates.memory(window, "k2", 30) is None


@pytest.mark.parametrize("nll, tools, tasks, judge, memory, want", [
    ("pass", True, True, True, True, "pass"), ("pass", True, None, None, None, "pass"),
    ("fail", True, True, True, None, "your decision"), ("near the limit", True, True, None, None, "open"),
    ("fail", True, False, True, None, "fail"), ("pass", False, True, True, True, "fail"),
    ("pass", True, True, True, False, "fail"), ("fail", True, True, False, None, "fail")])
def test_the_verdict_of_a_point(nll, tools, tasks, judge, memory, want):
    results = {"tasks": tasks, "judge": judge, "memory": memory}
    more = {name: None if ok is None else {"ok": ok} for name, ok in results.items()}
    assert gates.verdict(nll, tools, more) == want


def test_the_report_shows_each_gate_that_has_a_file(qwen_like, tmp_path):
    run = ran(qwen_like, tmp_path, {3: 31.5})                       # +5%: the nll gate fails
    suite = lambda n: {"total_tasks": 85, "total_correct": n}
    memory = {"name": "k3", "booted": True, "available_before_gib": 30, "available_lowest_gib": 8}
    files = {"tools-full.json": suite(50), "tools-k3.json": suite(49), "window.json": {"points": [memory]},
             "speed-full.json": {"tokens_per_second": 30.0, "runs": [{"prompt": 0, "speed": 30.0}]},
             "speed-k3.json": {"tokens_per_second": 31.5, "runs": [{"prompt": 0, "speed": 27.0}]},
             "judge-k3.json": {"ok": True, "summary": "12 better, 11 worse, 7 equal"}}
    for name, content in files.items():
        (run / name).write_text(json.dumps(content))
    point = probe.report(inspect(qwen_like), flow(tmp_path), run)["points"][0]
    assert point["verdict"] == "your decision" and point["tasks"]["ok"] and point["memory"]["peak_gib"] == 22
    out = probe.text(probe.report(inspect(qwen_like), flow(tmp_path), run))
    assert ("K 3     verdict: your decision; tasks 49 of 85 (the full model: 50); peak 22.0 GiB; "
            "speed 31.5 tok/s (the full model: 30.0, -10% for the same prompts); "
            "judge no loss (12 better, 11 worse, 7 equal)") in out
    (run / "tools-k3.json").write_text(json.dumps({"total_tasks": 85, "total_correct": 40}))
    assert probe.report(inspect(qwen_like), flow(tmp_path), run)["points"][0]["verdict"] == "fail"


def test_a_plan_with_a_rank_file_and_more_gates(qwen_like, tmp_path):
    facts, profile, w = inspect(qwen_like), probe.load_profile("qwen3_5_moe"), flow(tmp_path)
    machine = {"tools": {"model_benchmarks": {"dir": "/opt/mb", "python": "/opt/py"}}}
    plan = probe.plan(facts, w, profile, machine, turns=60, ranked="full-rank.json", more=("tasks", "speed"))
    full, k3 = plan["points"]
    kinds = [c.split()[3] if "prune_moe_kit.cli" in c else "suite" for c in full["clients"]]
    assert kinds == ["gate", "gate", "build", "suite", "gate"] and "--enable-return-routed-experts" not in full["boot"]
    assert "--rank full-rank.json --k 3" in full["clients"][2] and "--n 60" in full["clients"][0]
    suite = "cd /opt/mb && /opt/py -m benchmarks.tool_calling --base-url http://127.0.0.1:8890/v1"
    assert full["clients"][3].startswith(suite)
    assert full["clients"][3].endswith("/probe/tools-full.json") and "--speed" in k3["clients"][-1]
    with pytest.raises(probe.ProbeError, match="needs tools.model_benchmarks"):
        probe.plan(facts, w, profile, {}, more=("tasks",))


def test_the_speed_command_needs_its_options(capsys):
    assert cli.main(["gate", "--speed"]) == 2 and "--speed needs --out and --base-url" in capsys.readouterr().err
