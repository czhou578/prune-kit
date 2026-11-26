"""The data format of a workflow (plans/PLAN_PRUNE_MOE_KIT.md §5.2): ``sessions.jsonl``, the split and the secret pass.

The format is that of the record (spark-moe-specialize), so the data of the two prunes loads unchanged. One line is
one session::

    {"id": ..., "skill": <project>, "source": ..., "split": "calibration" | "held_out", "system_prompt": ...,
     "segments": [[<chat message in the OpenAI format>, ...], ...]}

An assistant message with ``"target": true`` is a turn that the model wrote: the capture counts it, and the replay
scores it. A token count in this module is an estimate: the characters divided by 4.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SECRET = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
    r"|xox[bap]-[A-Za-z0-9-]{20,}|Bearer [A-Za-z0-9._-]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)
CONTINUE = {"role": "user", "content": "Continue the task."}


def scrub(value):
    """``value`` with ``[REDACTED]`` in place of each string part that looks like a key. It goes into lists and
    mappings."""
    if isinstance(value, str):
        return SECRET.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def cut(text: str, limit: int = 12_000) -> str:
    """A long text keeps its first two thirds and its last quarter of ``limit``."""
    if len(text) <= limit:
        return text
    return text[:limit * 2 // 3] + "\n[... cut ...]\n" + text[-(limit // 4):]


def size(message: dict) -> int:
    """The characters of a message: its content and the arguments of its tool calls."""
    calls = message.get("tool_calls") or []
    return len(message.get("content") or "") + sum(len(c["function"]["arguments"]) for c in calls)


def tokens(session: dict) -> int:
    return sum(size(m) for seg in session["segments"] for m in seg) // 4


def split_segments(messages, seg_chars=160_000, hard=360_000, least=2_000):
    """Segments of about ``seg_chars`` characters. A cut comes before a user message. A segment past ``hard`` is cut
    before the next assistant turn, and the new segment starts with a short user message. A segment with fewer than
    ``least`` characters or with no target is dropped."""
    segs, cur, n = [], [], 0
    for m in messages:
        soft = m["role"] == "user" and n >= seg_chars
        if cur and (soft or (m["role"] == "assistant" and n >= hard)):
            segs.append(cur)
            cur, n = ([] if soft else [dict(CONTINUE)]), 0
        cur.append(m)
        n += size(m)
    segs = [s if s[0]["role"] == "user" else [dict(CONTINUE), *s] for s in [*segs, cur] if s]
    return [s for s in segs if sum(map(size, s)) >= least and any(m.get("target") for m in s)]


def is_held_out(key: str, share: float) -> bool:
    """A fixed choice by the hash of ``key``: true for about ``share`` of the keys."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100 < round(share * 100)


def session(sid, project, source, segments, held, system=None) -> dict:
    return {"id": sid, "skill": project, "source": source, "system_prompt": system,
            "split": "held_out" if held else "calibration", "segments": segments}


def limit_calibration(sessions: list, budgets: dict) -> list:
    """Keep at most ``budgets[source]`` tokens of calibration segments for each source: one segment at a time from
    each project in turn, in a fixed hash order, so that no project fills the budget alone."""
    out = [s for s in sessions if s["split"] != "calibration" or s["source"] not in budgets]
    by_id = {s["id"]: s for s in sessions}
    for source, budget in budgets.items():
        queues = defaultdict(list)
        for s in sessions:
            if s["split"] == "calibration" and s["source"] == source:
                queues[s["skill"]] += [(hashlib.sha256(f"{s['id']}:{g}".encode()).hexdigest(), s["id"], g)
                                       for g in range(len(s["segments"]))]
        queues, keep, total = {k: sorted(v) for k, v in sorted(queues.items())}, defaultdict(list), 0
        while total < budget and any(queues.values()):
            for q in queues.values():
                if q and total < budget:
                    _, sid, g = q.pop(0)
                    keep[sid].append(g)
                    total += sum(map(size, by_id[sid]["segments"][g])) // 4
        out += [{**by_id[sid], "segments": [by_id[sid]["segments"][g] for g in sorted(gs)]} for sid, gs in keep.items()]
    return out


def load(path) -> list:
    with Path(path).open() as f:   # not splitlines(): a JSON string can hold a Unicode line separator
        return [json.loads(line) for line in f]


def save(folder, sessions) -> Path:
    """Write ``sessions.jsonl`` into ``folder`` after the secret pass. The file can still hold private text."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "sessions.jsonl"
    path.write_text("".join(json.dumps(scrub(s)) + "\n" for s in sessions))
    return path


def stats(sessions) -> dict:
    """``{(source, split): counts}`` with the sessions, the segments, the tokens and the target turns. A target at
    the start of a segment has no context and does not count."""
    rows = defaultdict(Counter)
    for s in sessions:
        turns = sum(bool(m.get("target")) for seg in s["segments"] for m in seg[1:])
        rows[(s["source"], s["split"])].update(sessions=1, segments=len(s["segments"]), tokens=tokens(s), turns=turns)
    return dict(sorted(rows.items()))


def check(sessions, min_calibration_tokens: int, min_held_out_turns: int) -> dict:
    """The totals of the two splits against the minimums of the workflow."""
    rows = stats(sessions)
    cal = sum(c["tokens"] for (_, split), c in rows.items() if split == "calibration")
    held = sum(c["turns"] for (_, split), c in rows.items() if split == "held_out")
    return {"rows": [{"source": src, "split": split, **c} for (src, split), c in rows.items()],
            "calibration_tokens": cal, "held_out_turns": held,
            "ok": cal >= min_calibration_tokens and held >= min_held_out_turns}


def table(report: dict) -> list[str]:
    return [f"{r['source']:<16}{r['split']:<12}{r['sessions']:>6} sessions{r['segments']:>7} segments"
            f"{r['tokens']:>12,} tokens{r['turns']:>8} turns" for r in report["rows"]]
