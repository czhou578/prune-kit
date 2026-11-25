import json

import pytest

from prune_kit import capture, cli, replay, sessions


def tok(text, rank=1, logprob=-0.5, tid=1):
    return tid, {"decoded_token": text, "rank": rank, "logprob": logprob}


def test_score_tokens_finds_the_decision_and_the_tool_names():
    toks = [tok("I think"), tok("</think>", tid=7), tok("<tool_call>", rank=2), tok('{"name": "read'), tok('_file"'),
            tok(', "name": "ter', rank=1), tok('minal"}', rank=3)]
    s = replay.score_tokens(toks, end_think=7)
    assert (s["tokens"], s["nll"], s["top1"], s["reasoning_tokens"]) == (7, 3.5, 5, 2)
    assert (s["decision_ok"], s["tool_calls"], s["tool_names_ok"]) == (False, 2, 1)
    plain = replay.score_tokens([tok("Yes"), tok(".")], end_think=7)   # no reasoning: the first token decides
    assert plain["decision_ok"] and plain["reasoning_tokens"] == 0 and plain["tool_calls"] == 0


def held(sid, project, turns, source="claude_code", pad=10):
    seg = [m for n in range(turns) for m in ({"role": "user", "content": f"q{n}" + "x" * pad},
                                             {"role": "assistant", "content": f"a{n}", "target": True})]
    return sessions.session(sid, project, source, [seg], True, "SYSTEM")


def test_pick_takes_from_each_project_in_turn_and_is_fixed():
    data = [held("a", "big", 9), held("b", "big", 9), held("c", "small", 1), held("far", "long", 2, pad=400),
            {**held("cal", "big", 3), "split": "calibration"}]
    got = replay.pick(data, 4, max_chars=300)
    assert got == replay.pick(data, 4, max_chars=300) and len(set(got)) == 4
    # Each segment gives one target before a segment gives a second target, and the projects take turns.
    assert {sid for sid, _, _ in got[:3]} == {"a", "b", "c"} and got[1][0] == "c" and got[3][0] in ("a", "b")
    assert all(i > 0 for _, _, i in got)
    every = replay.pick(data, 99, max_chars=300)   # the context of "far" is above max_chars; "cal" is not held out
    assert len(every) == 2 + 2 + 1 and {sid for sid, _, _ in every} == {"a", "b", "c"}


class Server:
    """A context message is 3 tokens. A turn is its words, and each word scores -1.0 with rank 1."""

    def __init__(self):
        self.bodies = []

    def __call__(self, base, path, body):
        self.bodies.append((path, body))
        if path == "/tokenize":
            return {"tokens": [7] if "prompt" in body else [0] * 3 * len(body["messages"])}
        if any(m["content"] == "q0fail" for m in body["messages"]):
            raise OSError("the server sent 500")
        if "prompt_logprobs" not in body:
            message = {"content": "yes it is", "reasoning": "think first", "tool_calls": [{"id": "c"}]}
            return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}
        turn, before = body["messages"][-1], 3 * (len(body["messages"]) - 1)
        words = (turn.get("reasoning", "") + " </think> " + turn["content"]).split()
        logprobs = [{"7" if w == "</think>" else "1": {"decoded_token": w, "rank": 1, "logprob": -1.0}} for w in words]
        return {"prompt_logprobs": [None] * before + logprobs}


def test_record_then_score_and_a_second_start_does_the_failed_turn_only(tmp_path, monkeypatch):
    server = Server()
    monkeypatch.setattr(capture, "post", server)
    data = [held("a", "app", 1), held("b", "app", 1, pad=4)]
    data[1]["segments"][0][0]["content"] = "q0fail"
    rec, out = tmp_path / "r/recorded.jsonl", tmp_path / "score.jsonl"
    assert replay.record("http://x", "m", data, [{"type": "function"}], 2, rec) == 1
    rows = sessions.load(rec)
    good = next(r for r in rows if "error" not in r)
    assert good["turn"] == {"role": "assistant", "content": "yes it is", "reasoning_content": "think first",
                            "tool_calls": [{"id": "c"}]} and good["completion_tokens"] == 5
    chat = server.bodies[0][1]
    assert chat["messages"][0] == {"role": "system", "content": "SYSTEM"} and chat["temperature"] == 0.6
    assert "target" not in chat["messages"][1] and isinstance(chat["seed"], int)
    assert "OSError" in next(r for r in rows if r is not good)["error"]
    data[1]["segments"][0][0]["content"] = "q0"
    assert replay.record("http://x", "m", data, None, 2, rec) == 0 and len(sessions.load(rec)) == 3
    recorded = [r for r in sessions.load(rec) if "error" not in r]
    assert replay.score("http://x", "m", data, None, recorded, out) == 0
    scored = sessions.load(out)
    assert [(r["tokens"], r["nll"], r["reasoning_tokens"], r["decision_ok"]) for r in scored] == [(6, 6.0, 3, True)] * 2
    assert replay.score("http://x", "m", data, None, recorded, out) == 0 and len(sessions.load(out)) == 2
    own = tmp_path / "own.jsonl"
    assert replay.record(None, None, data, None, 2, own, from_data=True) == 0
    assert [(r["turn"]["content"], r["finish_reason"]) for r in sessions.load(own)] == [("a0", "data")] * 2


def scores(path, nll, names=4, decisions=True, turns=3):
    rows = [{"session": "s", "segment": 0, "index": n, "tokens": 100, "nll": nll, "top1": 90, "decision_ok": decisions,
             "tool_calls": 4, "tool_names_ok": names} for n in range(turns)]
    rows.append({"session": "s", "segment": 9, "index": 0, "error": "x"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_compare_gives_each_gate_and_the_command_gives_the_exit_code(tmp_path, capsys):
    ref = scores(tmp_path / "ref.jsonl", 30.0)
    result = replay.compare(sessions.load(ref), sessions.load(scores(tmp_path / "new.jsonl", 30.6, names=3, turns=2)))
    assert result["reference"]["turns"] == 2 and result["nll_change_pct"] == pytest.approx(2.0)
    assert list(result["gates"].values()) == [True, False, True]            # 75% of the tool names
    assert cli.main(["gate", "--compare", ref, scores(tmp_path / "ok.jsonl", 30.8)]) == 0
    assert "nll change  +2.67%" in capsys.readouterr().out
    assert cli.main(["gate", "--compare", ref, scores(tmp_path / "bad.jsonl", 31.0, decisions=False)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  nll: at most +3%" in out and "FAIL  tools: the decisions down by at most 3" not in out
    flow = tmp_path / "flow.yml"
    flow.write_text("name: w\ndata: {sources: []}\ngates: {nll: {limit_pct: 5}, tools: {decisions_max_drop: 2}}\n")
    assert cli.main(["gate", "--compare", ref, str(tmp_path / "bad.jsonl"), "--workflow", str(flow)]) == 1
    out = capsys.readouterr().out
    assert "PASS  nll: at most +5%" in out and "FAIL  tools: the decisions down by at most 2" in out
    assert cli.main(["gate", "--score", "--out", "x"]) == 2
    assert "--score also needs --recorded" in capsys.readouterr().err
