"""The data sources of a workflow (plans/PLAN_PRUNE_MOE_KIT.md §5.2). Each source gives sessions in the format of
``sessions.py``. ``build`` reads each source of a workflow file and writes ``sessions.jsonl``.

- ``claude_code``: the Claude Code transcripts of ``~/.claude/projects``, one session per transcript.
- ``gateway_records``: the records of ``model-gateway``, one session per conversation chain.
- ``repo``: code files of a repository, each as a short session: a request, a file read, and the file.
- ``sessions``: a ``sessions.jsonl`` that exists, with its own split. The export of the Hermes sessions of the stock
  bot is such a file.

A transcript and a record hold messages in the Anthropic format. ``to_chat`` makes chat messages in the OpenAI format
of them. Thinking blocks and images are left out, and a long tool result is cut.
"""

import fnmatch
import json
import re
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

from .sessions import cut, is_held_out, limit_calibration, load, save, session, split_segments

NOISE = re.compile(r"<system-reminder>.*?</system-reminder>|<local-command-\w+>.*?</local-command-\w+>"
                   r"|<command-(name|message|args)>.*?</command-\1>", re.S)
CODE = {".py", ".ts", ".tsx", ".js", ".go", ".sql", ".sh", ".cu", ".c", ".cpp", ".h", ".rs", ".md", ".yml", ".yaml",
        ".toml"}
HERMES = {"Bash": ("terminal", {"command": "command"}), "WebSearch": ("web_search", {"query": "query"}),
          "Read": ("read_file", {"file_path": "path", "offset": "offset", "limit": "limit"}),
          "Write": ("write_file", {"file_path": "path", "content": "content"}),
          "Edit": ("patch", {"file_path": "path", "old_string": "old_string", "new_string": "new_string"}),
          "Grep": ("search_files", {"pattern": "pattern", "path": "path", "glob": "file_glob"}),
          "Glob": ("search_files", {"pattern": "pattern", "path": "path"})}


def clean(text) -> str:
    return NOISE.sub("", text or "").strip()


def rename(name, args, tool_map):
    """The name and the arguments of a tool call. With the map ``hermes``, a Claude Code tool becomes the Hermes tool
    of the same use, and a call with no such tool gives None."""
    if tool_map != "hermes":
        return name, args or {}
    if name not in HERMES:
        return None
    new, keys = HERMES[name]
    return new, {keys[k]: v for k, v in (args or {}).items() if k in keys}


def to_chat(items, tool_map=None) -> list:
    """Chat messages for ``items``: one ``(role, content, key)`` per Anthropic message. Assistant messages that
    follow each other with the same ``key`` are one turn. Each assistant turn is a target."""
    out, kept, last = [], set(), None
    for role, content, key in items:
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        blocks = [b for b in blocks if isinstance(b, dict)]
        if role == "assistant":
            if key is None or key != last or not out or out[-1]["role"] != "assistant":
                out.append({"role": "assistant", "content": "", "target": True})
                last = key
            for b in blocks:
                if b.get("type") == "text":
                    out[-1]["content"] = (out[-1]["content"] + "\n" + clean(b.get("text"))).strip()
                elif b.get("type") == "tool_use" and (call := rename(b.get("name"), b.get("input"), tool_map)):
                    kept.add(b["id"])
                    out[-1].setdefault("tool_calls", []).append({"id": b["id"], "type": "function", "function": {
                        "name": call[0], "arguments": json.dumps(call[1])}})
            continue
        for b in blocks:
            if b.get("type") == "tool_result" and b.get("tool_use_id") in kept:
                text = b["content"] if isinstance(b.get("content"), str) else "\n".join(
                    x.get("text") or "" for x in b.get("content") or [] if isinstance(x, dict))
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": cut(clean(text))})
            elif b.get("type") == "text" and (text := clean(b.get("text"))):
                out.append({"role": "user", "content": cut(text, 40_000)})
    return [m for m in out if m["role"] != "assistant" or m["content"] or m.get("tool_calls")]


def project_name(folder: str) -> str:
    """``llm-reader`` for the transcript folder ``-home-<user>-Projects-llm-reader``."""
    return re.sub(r"^-home-[^-]+", "", folder).lstrip("-").removeprefix("Projects-")


def claude_code(opts, held):
    """``held(key, project)`` says if a session is held out."""
    for f in sorted(Path(opts.get("root", "~/.claude/projects")).expanduser().glob("*/*.jsonl")):
        project = project_name(f.parent.name)
        if not any(fnmatch.fnmatch(project, p) for p in opts.get("projects", ["*"])) or "tmp-" in f.parent.name:
            continue
        rows = []
        with f.open(errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                skip = ("isSidechain", "isMeta", "isCompactSummary", "isVisibleInTranscriptOnly")
                if r.get("type") in ("user", "assistant") and not any(r.get(k) for k in skip):
                    m = r.get("message") or {}
                    rows.append((r["type"], m.get("content"), m.get("id")))
        if segs := split_segments(to_chat(rows, opts.get("map"))):
            yield session(f"cc_{project}_{f.stem[:8]}", project, "claude_code", segs, held(f.stem, project))


def gateway_records(opts, held):
    """One session for each chain: the longest exchange of the chain, with the reply as its last turn."""
    full, chains = {}, {}
    for path in sorted(Path(opts.get("root", "~/.local/share/model-gateway/records")).expanduser().glob("*.jsonl")):
        for e in load(path):
            request = e["request"]
            if e.get("base"):
                prev = full.get(e["base"])
                if prev is None or len(prev.get("messages") or []) != e.get("offset"):
                    continue   # the base line is missing: this exchange cannot be rebuilt
                request = {**prev, **request, "messages": prev["messages"] + request["messages"]}
            full[e["id"]] = request
            reply = (e.get("response") or {}).get("content")
            if e.get("status") == 200 and reply and opts.get("route") in (None, e.get("dest")):
                chains[e["chain"]] = (e, request, reply)
    for chain, (e, request, reply) in chains.items():
        items = [(m["role"], m["content"], None) for m in request["messages"]] + [("assistant", reply, None)]
        system = request.get("system")
        system = system if isinstance(system, str) or not system else "\n\n".join(b.get("text", "") for b in system)
        if segs := split_segments(to_chat(items, opts.get("map"))):
            yield session(f"gw_{chain[:12]}", e["dest"], "gateway_records", segs, held(chain, e["dest"]), system)


def repo(opts, held):
    """``files`` code files of each repository, in a fixed hash order."""
    for folder in (Path(p).expanduser() for p in opts.get("paths", [])):
        listed = subprocess.run(["git", "-C", str(folder), "ls-files"], capture_output=True, text=True)
        names = listed.stdout.split("\n")
        names = [n for n in names if Path(n).suffix in CODE and (folder / n).is_file()
                 and 500 <= (folder / n).stat().st_size <= 40_000]
        for n in sorted(names, key=lambda n: sha256(n.encode()).hexdigest())[:opts.get("files", 6)]:
            call = {"id": "call_" + sha256(n.encode()).hexdigest()[:12], "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": n})}}
            seg = [{"role": "user", "content": f"In the repository {folder.name}, read `{n}`. Explain what it does, "
                                               "then propose one concrete improvement as a patch."},
                   {"role": "assistant", "content": "", "tool_calls": [call], "target": True},
                   {"role": "tool", "tool_call_id": call["id"],
                    "content": cut((folder / n).read_text(errors="replace"), 40_000)}]
            yield session(f"repo_{folder.name}_{call['id'][5:]}", folder.name, "repo", [seg], held(None, folder.name))


def build(workflow) -> list:
    """Read each source, keep the budget of each source, and write the data folder of the workflow."""
    d = workflow.data
    held = lambda key, project: project in d["held_out"]["projects"] or (
        key is not None and is_held_out(key, d["held_out"]["share"]))
    system = Path(d["system_prompt"]).expanduser().read_text() if d["system_prompt"] else None
    out, budgets = [], {}
    for opts in d["sources"]:
        if opts["kind"] == "sessions":
            old = load(Path(opts["path"]).expanduser())
            out += [s for s in old if s["source"] in opts.get("sources", [s["source"]])]
            continue
        new = {"claude_code": claude_code, "gateway_records": gateway_records, "repo": repo}[opts["kind"]](opts, held)
        out += [{**s, "system_prompt": system or s["system_prompt"]} for s in new]
        if "budget" in opts:
            budgets[opts["kind"]] = opts["budget"]
    out = limit_calibration(out, budgets)
    save(workflow.folder, out)
    if d["tools"]:
        shutil.copyfile(Path(d["tools"]).expanduser(), workflow.folder / "tools.json")
    return out
