import json

import pytest

from prune_kit import capture, cli, judge, probe, sessions
from prune_kit.facts import inspect

from test_probe import flow
from test_replay import held


def recorded(data, text):
    """One recorded turn for target 1 of each session. ``text(n)`` is the content of turn n."""
    return [{"session": s["id"], "segment": 0, "index": 1, "skill": s["skill"],
             "turn": {"role": "assistant", "content": text(n), "reasoning_content": "hidden thoughts"}}
            for n, s in enumerate(data)]


def test_the_pairs_are_blind_and_fixed():
    data = [held(f"s{n}", "app", 1) for n in range(20)]
    ref, new = recorded(data, lambda n: f"full {n}"), recorded(data, lambda n: f"point {n}")
    new[3]["turn"] = {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a"}'}}]}
    blind = judge.pairs(data, ref, [*new[:19], {**new[19], "error": "x"}], turns=30)
    assert blind == judge.pairs(data, ref, new[:19], turns=30) and len(blind) == 19
    firsts = [p["new"] for p in blind]
    assert 4 < firsts.count("A") < 15 and all(p[p["new"]].startswith(("point", "[tool call]")) for p in blind)
    assert blind[3][blind[3]["new"]] == '[tool call] read_file({"path": "a"})' and "[user] q0" in blind[0]["request"]
    assert "hidden thoughts" not in json.dumps(blind) and "SYSTEM" not in json.dumps(blind)
    assert len(judge.pairs(data, ref, new, turns=5)) == 5


def test_the_sign_test():
    assert judge.sign_test(0, 0) == 1.0 and judge.sign_test(5, 5) == 1.0
    assert judge.sign_test(1, 9) == pytest.approx(22 / 1024) and judge.sign_test(9, 1) == pytest.approx(22 / 1024)
    assert judge.sign_test(3, 7) > 0.05


def judge_that_likes(word):
    """A judge that prefers the answer with ``word`` and says EQUAL when both have it or none has it."""
    def post(base, path, body):
        prompt = body["messages"][0]["content"]
        a, b = (word in prompt.split(f"## Answer {x}\n")[1].split("## Answer")[0] for x in "AB")
        return {"choices": [{"message": {"content": f"I think... {'EQUAL' if a == b else 'A' if a else 'B'}"}}]}
    return post


def test_the_judge_counts_the_votes_and_finds_a_loss_only_with_the_test(tmp_path):
    data = [held(f"s{n}", "app", 1) for n in range(12)]
    ref, new = recorded(data, lambda n: f"full {n}"), recorded(data, lambda n: f"point {n}")
    r = judge.judge("http://x", "j", judge.pairs(data, ref, new, 30), post=judge_that_likes("full"))
    assert (r["better"], r["worse"], r["equal"], r["ok"]) == (0, 12, 0, False) and r["p"] < 0.001
    r = judge.judge("http://x", "j", judge.pairs(data, ref, new, 30), post=judge_that_likes("point"))
    assert (r["better"], r["ok"]) == (12, True)                      # a win is "no loss", and not a gain
    mixed = recorded(data, lambda n: f"point {n}" if n < 4 else f"full {n}" if n < 8 else "none")
    r = judge.judge("http://x", "j", judge.pairs(data, ref, mixed, 30), post=judge_that_likes("full"))
    assert (r["better"], r["worse"], r["equal"], r["ok"]) == (0, 8, 4, False)
    r = judge.judge("http://x", "j", judge.pairs(data, ref, mixed[:6], 30), post=judge_that_likes("full"))
    assert (r["worse"], r["equal"], r["ok"]) == (4, 2, True) and "worse in 4" in r["summary"]   # 4 to 0: p = 0.125


def test_the_judge_command_writes_the_result_and_the_blind_page(tmp_path, monkeypatch, capsys):
    data = [held(f"s{n}", "app", 1) for n in range(6)]
    sessions.save(tmp_path / "d", data)
    for name, text in (("ref", "full <b>"), ("new", "point")):
        rows = recorded(data, lambda n: f"{text} {n}")
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "flow.yml").write_text(f"name: w\ndata: {{sources: [], folder: {tmp_path / 'd'}}}\n"
                                       "gates: {judge: {turns: 4, model: judge-model}}\n")
    bodies = []
    monkeypatch.setattr(capture, "post", lambda base, path, body: bodies.append(body) or judge_that_likes("point")(
        base, path, body))
    args = ["judge", str(tmp_path / "ref.jsonl"), str(tmp_path / "new.jsonl"), "--workflow", str(tmp_path / "flow.yml"),
            "--base-url", "http://x", "--out", str(tmp_path / "judge-k3.json"), "--page", str(tmp_path / "pairs.html")]
    assert cli.main([*args, "--body", '{"chat_template_kwargs": {"thinking": false}}']) == 0
    assert "better in 4, worse in 0" in capsys.readouterr().out and len(bodies) == 4
    assert bodies[0]["model"] == "judge-model" and bodies[0]["chat_template_kwargs"] == {"thinking": False}
    assert json.loads((tmp_path / "judge-k3.json").read_text())["ok"] is True
    page = (tmp_path / "pairs.html").read_text()
    assert page.count("<h2>Pair") == 4 and "full &lt;b&gt;" in page and "<details><summary>The key" in page
    assert cli.main([*args[:3], "--workflow", "none.yml", "--base-url", "http://x", "--out", "o"]) == 2


def test_a_plan_with_the_judge_records_the_turns_of_each_cut(qwen_like, tmp_path):
    plan = probe.plan(inspect(qwen_like), flow(tmp_path), probe.load_profile("qwen3_5_moe"), {}, more=("judge",))
    full, k3 = plan["points"]
    assert k3["clients"][-1].endswith("--n 30 --out runs/" + f"{qwen_like.name}/w/probe/recorded-k3.jsonl")
    assert "--record" in k3["clients"][-1] and not any("recorded-full" in c for c in full["clients"])
