import json
import subprocess

from prune_kit import cli, sessions, sources, workflow

LONG = "x" * 3000   # a segment needs 2,000 characters


def use(tid, name, **args):
    return {"type": "tool_use", "id": tid, "name": name, "input": args}


def result(tid, text):
    return {"type": "tool_result", "tool_use_id": tid, "content": [{"type": "text", "text": text}]}


def test_to_chat_merges_a_turn_and_keeps_the_results_of_its_calls():
    items = [("user", "<system-reminder>noise</system-reminder>Fix the bug", None),
             ("assistant", [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "I read it."}], "m1"),
             ("assistant", [use("t1", "Read", file_path="a.py", limit=5)], "m1"),
             ("user", [result("t1", "print(1)"), result("t9", "of an unknown call")], None),
             ("assistant", "Done.", "m2")]
    chat = sources.to_chat(items)
    assert [m["role"] for m in chat] == ["user", "assistant", "tool", "assistant"]
    assert chat[0]["content"] == "Fix the bug" and chat[1]["content"] == "I read it." and chat[1]["target"]
    call = chat[1]["tool_calls"][0]["function"]
    assert call["name"] == "Read" and json.loads(call["arguments"]) == {"file_path": "a.py", "limit": 5}
    assert chat[2] == {"role": "tool", "tool_call_id": "t1", "content": "print(1)"}


def test_the_hermes_map_renames_a_call_and_drops_a_call_with_no_match():
    items = [("user", "go", None), ("assistant", [use("t1", "Read", file_path="a.py"), use("t2", "TodoWrite")], "m"),
             ("user", [result("t1", "a"), result("t2", "b")], None)]
    chat = sources.to_chat(items, "hermes")
    assert [c["function"]["name"] for c in chat[1]["tool_calls"]] == ["read_file"]
    assert json.loads(chat[1]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"} and len(chat) == 3


def transcript(folder, name, turns):
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "summary"}, {"type": "user", "isMeta": True, "message": {"content": "meta"}}]
    for n in range(turns):
        rows += [{"type": "user", "message": {"content": f"question {n} {LONG}"}},
                 {"type": "assistant", "message": {"id": f"m{n}", "content": f"answer {n}"}}]
    (folder / f"{name}.jsonl").write_text("not json\n" + "".join(json.dumps(r) + "\n" for r in rows))


def test_claude_code_gives_one_session_for_each_transcript_of_a_wanted_project(tmp_path):
    transcript(tmp_path / "-home-me-Projects-llm-reader", "aaaaaaaa-1", 2)
    transcript(tmp_path / "-home-me-Projects-other", "bbbbbbbb-1", 2)
    transcript(tmp_path / "-home-me-Projects-held", "cccccccc-1", 1)
    held = lambda key, project: project == "held"
    got = list(sources.claude_code({"root": str(tmp_path), "projects": ["llm-*", "held"]}, held))
    assert [(s["id"], s["skill"], s["split"]) for s in got] == [
        ("cc_held_cccccccc", "held", "held_out"), ("cc_llm-reader_aaaaaaaa", "llm-reader", "calibration")]
    assert [m["role"] for m in got[1]["segments"][0]] == ["user", "assistant", "user", "assistant"]


def record(rid, chain, messages, reply, base=None, offset=0, dest="second-model", status=200, **request):
    return {"id": rid, "chain": chain, "base": base, "offset": offset, "dest": dest, "status": status,
            "request": {"messages": messages, **request}, "response": {"content": [{"type": "text", "text": reply}]}}


def test_gateway_records_rebuild_each_chain(tmp_path):
    rows = [record("1", "chainA", [{"role": "user", "content": "one " + LONG}], "first", system=[{"text": "SYS"}]),
            record("2", "chainA", [{"role": "assistant", "content": "first"}, {"role": "user", "content": "two"}],
                   "second", base="1", offset=1),
            record("3", "chainA", [{"role": "user", "content": "lost"}], "x", base="0", offset=7),
            record("4", "chainB", [{"role": "user", "content": LONG}], "other route", dest="deepseek"),
            record("5", "chainC", [{"role": "user", "content": LONG}], "failed", status=500)]
    (tmp_path / "2026-11-25.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = list(sources.gateway_records({"root": str(tmp_path), "route": "second-model"}, lambda key, project: False))
    assert len(got) == 1 and got[0]["system_prompt"] == "SYS" and got[0]["skill"] == "second-model"
    assert [m["content"][:6] for m in got[0]["segments"][0]] == ["one xx", "first", "two", "second"]
    assert len(list(sources.gateway_records({"root": str(tmp_path)}, lambda key, project: False))) == 2


def test_build_writes_the_data_folder_of_a_workflow(tmp_path, capsys):
    transcript(tmp_path / "cc/-home-me-Projects-app", "aaaaaaaa-1", 2)
    code = tmp_path / "code"
    code.mkdir()
    (code / "a.py").write_text(("print('key sk-" + "a1" * 12 + "')\n") * 30)
    (code / "notes.bin").write_text("x" * 900)
    subprocess.run("git init -q && git add -A", shell=True, cwd=code, check=True)
    seg = [[{"role": "user", "content": "q"}]]
    sessions.save(tmp_path / "old", [sessions.session("d1", "daily", "discord", seg, True),
                                     sessions.session("s1", "x", "subagent", seg, False)])
    (tmp_path / "tools.json").write_text("[]")
    (tmp_path / "system.txt").write_text("SYSTEM")
    flow = tmp_path / "flow.yml"
    flow.write_text(f"""name: w
data:
  folder: {tmp_path / 'out'}
  tools: {tmp_path / 'tools.json'}
  system_prompt: {tmp_path / 'system.txt'}
  sources:
    - {{kind: sessions, path: {tmp_path / 'old/sessions.jsonl'}, sources: [discord]}}
    - {{kind: claude_code, root: {tmp_path / 'cc'}, budget: 100000}}
    - {{kind: repo, paths: [{code}]}}
  held_out: {{share: 0, projects: [code]}}
  min_calibration_tokens: 1000
  min_held_out_turns: 0
""")
    got = sources.build(workflow.load(flow))
    assert {(s["source"], s["split"]) for s in got} == {("discord", "held_out"), ("claude_code", "calibration"),
                                                        ("repo", "held_out")}
    saved = (tmp_path / "out/sessions.jsonl").read_text()
    assert "[REDACTED]" in saved and "sk-a1" not in saved and (tmp_path / "out/tools.json").read_text() == "[]"
    by_source = {s["source"]: s["system_prompt"] for s in got}
    assert by_source["claude_code"] == "SYSTEM" and by_source["discord"] is None
    assert cli.main(["data", str(flow), "--stats"]) == 0 and "the data meets the minimums" in capsys.readouterr().out
    flow.write_text(flow.read_text().replace("min_held_out_turns: 0", "min_held_out_turns: 5"))
    assert cli.main(["data", str(flow)]) == 1 and "the data is below a minimum" in capsys.readouterr().out
    assert cli.main(["data", str(tmp_path / "none.yml")]) == 2
