"""The gate ``judge`` (plans/PLAN_PRUNE_KIT.md §5.6): a blind judge on pairs of answers, and a blind page for you.

The full model and a pruned point each write one turn for the same held-out targets (``prune-kit gate --record``,
with the same seeds). A judge model reads the request and the two answers under the names A and B, in an order that
a hash of the turn fixes, and says which answer is better. A sign test compares the wins of the two models. The gate
finds a loss only when the point loses more often than it wins and the test shows it (p below 0.05): a judge result
is "no loss", and never a gain (rule R5 of BLUEPRINT.md).

The page shows the same pairs with no model names, for your own review. The key is at its end.
"""

import hashlib
import html
import json
import re
from math import comb
from pathlib import Path

from . import capture, replay

ASK = ("You compare two answers of an AI assistant to the same request. Judge only which answer helps the user "
       "more: it is correct, it does what the request asks, and its tool calls fit the task. Reply with one word: "
       "A, B, or EQUAL.")
WORD = re.compile(r"\b(A|B|EQUAL)\b")


def answer(turn: dict) -> str:
    """The text of a recorded turn, with its tool calls and with no reasoning."""
    calls = [f"[tool call] {c['function']['name']}({c['function']['arguments']})" for c in turn.get("tool_calls") or []]
    return "\n".join([turn.get("content") or "", *calls]).strip() or "(no text)"


def request(session: dict, r: dict, limit: int = 6000) -> str:
    """The end of the context of a turn: its last three messages, each cut to ``limit`` characters."""
    before = replay.context(session, session["segments"][r["segment"]], r["index"])[-3:]
    return "\n\n".join(f"[{m['role']}] {(m.get('content') or '')[-limit:]}" for m in before if m["role"] != "system")


def pairs(sessions, ref_rows, new_rows, turns: int) -> list:
    """The first ``turns`` common turns as blind pairs. ``new`` is A for one half of the hashes."""
    by_id = {s["id"]: s for s in sessions}
    new = {replay.key(r): r for r in new_rows if "error" not in r}
    out = []
    for r in [r for r in ref_rows if "error" not in r and replay.key(r) in new][:turns]:
        new_first = int(hashlib.sha256(repr(replay.key(r)).encode()).hexdigest(), 16) % 2 == 0
        texts = answer(new[replay.key(r)]["turn"]), answer(r["turn"])
        out.append({"key": list(replay.key(r)), "request": request(by_id[r["session"]], r),
                    "new": "A" if new_first else "B",
                    "A": texts[0] if new_first else texts[1], "B": texts[1] if new_first else texts[0]})
    return out


def sign_test(wins: int, losses: int) -> float:
    """The two-sided p value of the sign test: the chance of a split this unequal when both models are equal."""
    n, low = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(low + 1)) / 2 ** n)


def judge(base, model, blind: list, extra=None, post=None) -> dict:
    """Ask the judge for each pair. A reply with no clear word counts as equal."""
    post, votes = post or capture.post, []
    for p in blind:
        prompt = f"{ASK}\n\n## The request\n{p['request']}\n\n## Answer A\n{p['A']}\n\n## Answer B\n{p['B']}"
        reply = post(base, "/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 600,
                                                   **(extra or {}), "messages": [{"role": "user", "content": prompt}]})
        words = WORD.findall((reply["choices"][0]["message"].get("content") or "").upper())
        votes.append("equal" if not words or words[-1] == "EQUAL" else "better" if words[-1] == p["new"] else "worse")
    better, worse, equal = (votes.count(v) for v in ("better", "worse", "equal"))
    p_value = sign_test(better, worse)
    summary = f"the point is better in {better}, worse in {worse}, equal in {equal}; sign test p = {p_value:.2f}"
    return {"turns": len(blind), "better": better, "worse": worse, "equal": equal, "p": round(p_value, 4),
            "ok": not (worse > better and p_value < 0.05), "votes": votes, "summary": summary}


def page(blind: list, title: str) -> str:
    """One HTML page with the blind pairs. The key is in a closed block at the end."""
    cell = lambda text: f"<pre>{html.escape(text)}</pre>"
    rows = "".join(f"<h2>Pair {n}</h2><h3>The request</h3>{cell(p['request'][-3000:])}<div class=pair><div><h3>Answer A"
                   f"</h3>{cell(p['A'])}</div><div><h3>Answer B</h3>{cell(p['B'])}</div></div>"
                   for n, p in enumerate(blind, 1))
    key = ", ".join(f"pair {n}: the pruned point is {p['new']}" for n, p in enumerate(blind, 1))
    return (f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>body{{font:15px sans-serif;"
            "max-width:1400px;margin:2em auto;padding:0 1em}pre{white-space:pre-wrap;background:#8881;padding:.8em}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1em}</style>"
            f"<h1>{html.escape(title)}</h1><p>Each pair has the answer of the full model and the answer of the pruned "
            "point, in a mixed order. Write down the better answer of each pair. Then open the key.</p>"
            f"{rows}<details><summary>The key</summary><p>{html.escape(key)}</p></details>")


def run(base, model, sessions, ref_rows, new_rows, turns, out, page_path=None, extra=None, post=None) -> dict:
    blind = pairs(sessions, ref_rows, new_rows, turns)
    if not blind:
        raise ValueError("the two files have no common turn")
    result = judge(base, model, blind, extra, post)
    Path(out).write_text(json.dumps(result, indent=1))
    if page_path:
        Path(page_path).write_text(page(blind, f"Blind pairs: {Path(out).stem}"))
    return result
