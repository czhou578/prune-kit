from pathlib import Path

import pytest

from prune_moe_kit import cli, preflight, workflow
from prune_moe_kit.facts import inspect

EXAMPLE = Path(__file__).parent.parent / "examples/coding-fast-route.yml"


def write(tmp_path, text):
    path = tmp_path / "flow.yml"
    path.write_text(text)
    return path


def test_the_example_loads_with_the_defaults():
    w = workflow.load(EXAMPLE)
    assert w.name == "coding-fast-route" and w.points == "auto"
    assert [s["kind"] for s in w.data["sources"]] == ["gateway_records", "claude_code", "repo"]
    assert w.data["min_held_out_turns"] == 60 and w.gates["tasks"]["max_drop"] == 3
    assert w.memory == {"peak_gib": 26.5, "context": 128000, "margin_gib": 4, "overhead_gib": None}
    assert w.folder == Path("runs/data/coding-fast-route")


def test_a_small_file_gets_each_default(tmp_path):
    w = workflow.load(write(tmp_path, "name: a\ndata: {sources: [{kind: repo}], folder: ~/x}\n"))
    assert w.gates["nll"] == {"limit_pct": 3.0, "turns": 60} and w.data["held_out"] == {"share": 0.15, "projects": []}
    assert w.folder == Path.home() / "x" and w.need_bytes(10 << 30) is None


def test_each_fault_is_in_one_error(tmp_path):
    text = ("name: a b\ndata: {sources: [{kind: web}, {kind: repo, route: x}], held_out: {share: 1.5}}\n"
            "gates: {nll: {limit: 3, turns: many}}\nmemory: {peak_gib: -1}\npoints: [208, 0]\n")
    with pytest.raises(workflow.WorkflowError) as e:
        workflow.load(write(tmp_path, text))
    for fault in ("has 8 faults", "name: use letters", "data.sources[0].kind: must be one of",
                  "data.sources[1].route: not a key of the kind repo", "data.held_out.share: must be at least 0",
                  "gates.nll.limit: not a known key", "gates.nll.turns: must be of type int",
                  "memory.peak_gib: must not be negative", 'points: must be "auto" or a list'):
        assert fault in str(e.value)


@pytest.mark.parametrize("text, fault", [("- a\n", "the file: must be a mapping"), ("a: [", "is not YAML"),
                                         ("data: {sources: []}\n", "name: is missing")])
def test_a_file_with_no_mapping_or_no_name(tmp_path, text, fault):
    with pytest.raises(workflow.WorkflowError, match=fault):
        workflow.load(write(tmp_path, text))


def test_a_missing_file(tmp_path):
    with pytest.raises(workflow.WorkflowError, match="cannot read the workflow file"):
        workflow.load(tmp_path / "none.yml")


def flow(tmp_path, memory):
    return workflow.load(write(tmp_path, f"name: w\ndata: {{sources: []}}\nmemory: {memory}\n"))


def test_the_need_is_the_largest_k_that_reaches_the_goal(qwen_like, tmp_path):
    f = inspect(qwen_like)   # 2 layers of 4 experts with 110 bytes each
    over = (f.total_bytes - 300) / (1 << 30)   # a goal that is 300 bytes below the weights
    n = preflight.need(f, flow(tmp_path, f"{{peak_gib: {over!r}}}"))
    assert (n["bytes"], n["k"], n["share"]) == (300, 2, 0.5)   # K 3 frees 220 bytes, K 2 frees 440
    out = preflight.text(preflight.report(f, workflow=flow(tmp_path, f"{{peak_gib: {over!r}, context: 128000}}")))
    assert "at a context of 128,000 tokens, for the workflow w" in out and "K 2 or lower (50% of the experts)" in out
    assert "a lower limit: memory.overhead_gib is not set" in out


def test_a_goal_that_no_cut_reaches_and_a_goal_that_needs_no_cut(qwen_like, tmp_path):
    f = inspect(qwen_like)
    r = preflight.report(f, workflow=flow(tmp_path, "{peak_gib: 0, overhead_gib: 1}"))
    assert r["need"]["k"] is None and "no cut of the experts reaches it" in preflight.text(r)
    out = preflight.text(preflight.report(f, workflow=flow(tmp_path, "{peak_gib: 1, overhead_gib: 0}")))
    assert "no cut: the model fits the goal" in out and "a lower limit" not in out


def test_preflight_command_with_a_workflow(qwen_like, tmp_path, capsys):
    assert cli.main(["preflight", str(qwen_like), "--workflow", str(EXAMPLE)]) == 0
    assert "goal      a peak of 26.5 GiB" in capsys.readouterr().out
    assert cli.main(["preflight", str(qwen_like), "--workflow", str(tmp_path / "none.yml")]) == 2
    assert "cannot read the workflow file" in capsys.readouterr().err
