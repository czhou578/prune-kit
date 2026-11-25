import json
from pathlib import Path

import pytest

from prune_kit import preflight, sessions, workflow
from prune_kit.facts import inspect

RECORD = Path.home() / "Projects/dsv4-specialize/data/qwen-data/sessions.jsonl"


def turn(text, **more):
    return {"role": "assistant", "content": text, "target": True, **more}


def user(text):
    return {"role": "user", "content": text}


def make(sid, source="claude_code", project="p", held=False, chars=4000, turns=2):
    seg = [m for _ in range(turns) for m in (user("u" * (chars // turns // 2)), turn("a" * (chars // turns // 2)))]
    return sessions.session(sid, project, source, [seg], held)


def test_scrub_replaces_keys_at_each_depth():
    key, token = "sk-" + "a1" * 12, "ghp_" + "B2" * 16
    call = {"function": {"name": "terminal", "arguments": json.dumps({"command": f"export KEY={key}"})}}
    out = sessions.scrub({"segments": [[{"content": f"Authorization: Bearer {'x' * 24}", "tool_calls": [call]}]],
                          "system_prompt": f"token {token}", "n": 3})
    text = json.dumps(out)
    assert key not in text and token not in text and "x" * 24 not in text
    assert text.count("[REDACTED]") == 3 and out["n"] == 3
    assert sessions.scrub("task-force sk-short") == "task-force sk-short"


def test_cut_keeps_the_start_and_the_end():
    assert sessions.cut("abc") == "abc"
    out = sessions.cut("a" * 9000 + "b" * 9000, limit=12_000)
    assert out.startswith("a" * 8000 + "\n[... cut ...]\n") and out.endswith("b" * 3000) and len(out) < 12_000


def test_size_counts_the_content_and_the_arguments():
    call = {"function": {"name": "read_file", "arguments": "x" * 10}}
    assert sessions.size({"content": "abc", "tool_calls": [call]}) == 13
    assert sessions.size({"content": None, "tool_calls": None}) == 0


def test_segments_cut_before_a_user_message():
    msgs = [m for _ in range(4) for m in (user("u" * 600), turn("a" * 600))]
    segs = sessions.split_segments(msgs, seg_chars=2000, hard=10_000, least=100)
    assert [len(s) for s in segs] == [4, 4] and all(s[0]["role"] == "user" for s in segs)


def test_a_long_run_with_no_user_message_gets_a_hard_cut():
    msgs = [user("u")] + [m for _ in range(6) for m in (turn("a" * 500), {"role": "tool", "content": "t" * 500})]
    segs = sessions.split_segments(msgs, seg_chars=1000, hard=2000, least=100)
    assert len(segs) > 1 and all(s[0]["role"] == "user" for s in segs)
    assert segs[1][0] == sessions.CONTINUE and segs[1][1]["role"] == "assistant"


def test_small_segments_and_segments_with_no_target_are_dropped():
    assert sessions.split_segments([user("u" * 50), turn("a" * 50)], least=2000) == []
    assert sessions.split_segments([user("u" * 3000), {"role": "assistant", "content": "a" * 3000}]) == []


def test_the_held_out_choice_is_fixed_and_near_the_share():
    held = [sessions.is_held_out(f"session-{n}", 0.15) for n in range(2000)]
    assert held == [sessions.is_held_out(f"session-{n}", 0.15) for n in range(2000)]
    assert 0.12 < sum(held) / 2000 < 0.18
    assert not any(sessions.is_held_out(f"s{n}", 0) for n in range(200))


def test_the_budget_takes_from_each_project_in_turn():
    data = [make(f"a{n}", project="big") for n in range(10)] + [make("b0", project="small"),
                                                                 make("h", held=True), make("o", source="repo")]
    out = sessions.limit_calibration(data, {"claude_code": 4000})
    kept = [s["id"] for s in out if s["source"] == "claude_code" and s["split"] == "calibration"]
    assert len(kept) == 4 and "b0" in kept                       # 4 segments of 1,000 tokens; the small project is in
    assert {"h", "o"} <= {s["id"] for s in out}                  # other splits and sources stay


def test_save_scrubs_and_load_reads_the_same_sessions(tmp_path):
    s = make("s1")
    s["segments"][0][0]["content"] = "key hf_" + "a" * 32 + " and a line separator   in the text"
    got = sessions.load(sessions.save(tmp_path / "out", [s, make("s2")]))
    assert [g["id"] for g in got] == ["s1", "s2"] and "[REDACTED]" in got[0]["segments"][0][0]["content"]
    assert " " in got[0]["segments"][0][0]["content"]


def test_check_counts_the_two_splits():
    data = [make("c1", chars=8000), make("c2", source="repo", chars=4000), make("h1", held=True, turns=3)]
    data[2]["segments"][0].insert(0, turn("a target with no context does not count"))
    r = sessions.check(data, min_calibration_tokens=3000, min_held_out_turns=3)
    assert (r["calibration_tokens"], r["held_out_turns"], r["ok"]) == (3000, 3, True)
    assert [(x["source"], x["split"]) for x in r["rows"]] == [
        ("claude_code", "calibration"), ("claude_code", "held_out"), ("repo", "calibration")]
    assert not sessions.check(data, 3001, 3)["ok"] and not sessions.check(data, 3000, 4)["ok"]
    assert "2,000 tokens" in sessions.table(r)[0]


def test_the_preflight_reports_the_data_of_the_workflow(qwen_like, tmp_path):
    flow = tmp_path / "flow.yml"
    flow.write_text(f"name: w\ndata: {{sources: [], folder: {tmp_path / 'd'}, min_calibration_tokens: 1500,"
                    " min_held_out_turns: 2}\n")
    w = workflow.load(flow)
    r = preflight.report(inspect(qwen_like), workflow=w)
    assert r["verdict"] == "GO WITH WORK: data" and "sessions.jsonl does not exist" in preflight.text(r)
    sessions.save(w.folder, [make("c", chars=8000), make("h", held=True)])
    r = preflight.report(inspect(qwen_like), workflow=w)
    out = preflight.text(r)
    assert r["verdict"] == "GO" and r["data"]["ok"]
    assert "2,000 calibration tokens (minimum 1,500), 2 held-out turns (minimum 2)" in out
    assert "claude_code     calibration" in out


@pytest.mark.skipif(not RECORD.exists(), reason="the data of the Qwen prune is not on this machine")
def test_the_data_of_the_qwen_prune_loads_unchanged():
    r = sessions.check(sessions.load(RECORD), 1_000_000, 60)
    assert r["ok"] and {x["source"] for x in r["rows"]} >= {"claude", "discord", "repo"}
