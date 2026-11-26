"""The replay gate (plans/PLAN_PRUNE_KIT.md §5.6, the gates ``nll`` and ``tools``): record, score, compare.

- ``record``: the full model writes one turn for each of N held-out targets. The context is the segment up to the
  target. The seed of each turn is fixed, so a second run records the same sample.
- ``score``: a model scores the recorded turns. One request sends the context and the recorded turn as the last
  message, with ``prompt_logprobs``. The tokens after the context are the turn. A row holds the NLL of the turn, its
  token count, the tokens that rank first, the decision (the first token after the reasoning ranks first) and the
  tool names (each token of each ``"name": "..."`` ranks first).
- ``compare``: the totals of two score files on their common turns, against the limits of the gates.

The reference of a window is the full model, scored in that window (rule R3 of BLUEPRINT.md).
"""

import hashlib
import json
import re
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from . import capture
from .sessions import load

NAME = re.compile(r'"name": ?"([^"]+)"')
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}   # the settings of the Qwen model card with reasoning
LIMITS = {"nll": {"limit_pct": 3.0}, "tools": {"names_min_pct": 98, "decisions_max_drop": 3}}
MEASURES = ("tokens", "nll", "top1", "decision_ok", "tool_calls", "tool_names_ok")
key = lambda r: (r["session"], r["segment"], r["index"])


def context(session, seg, i) -> list:
    """The system prompt (if any) and the segment up to message ``i``, without the data fields."""
    system = [{"role": "system", "content": session["system_prompt"]}] if session.get("system_prompt") else []
    return system + [{k: v for k, v in m.items() if k not in capture.STRIP} for m in seg[:i]]


def pick(sessions, n, max_chars=240_000, per_segment=2) -> list:
    """``n`` held-out targets as ``(session id, segment, index)``: one source and project after the other in a fixed
    hash order, with a context of at most ``max_chars`` characters. Each segment gives one target before a segment
    gives a second target."""
    groups = defaultdict(list)
    for s in sessions:
        for g, seg in enumerate(s["segments"] if s["split"] == "held_out" else []):
            sizes, total = [], 0
            for m in seg:
                sizes.append(total)
                total += len(json.dumps(m))
            order = lambda i: hashlib.sha256(f"{s['id']}:{g}:{i}".encode()).hexdigest()
            targets = [i for i, m in enumerate(seg) if m.get("target") and 0 < i and sizes[i] <= max_chars]
            groups[(s["source"], s["skill"])] += [(j, order(i), s["id"], g, i)
                                                  for j, i in enumerate(sorted(targets, key=order)[:per_segment])]
    rounds = [[r for row in zip_longest(*(sorted(x for x in v if x[0] == j) for _, v in sorted(groups.items())))
               for r in row if r] for j in range(per_segment)]
    return [(sid, g, i) for _, _, sid, g, i in [r for round_ in rounds for r in round_][:n]]


def _rows(path, todo, work) -> int:
    """Append ``work(item)`` to the file for each item of ``todo`` that the file does not hold with no error. A
    failure becomes a row with the error. Return the number of failures."""
    path, failed = Path(path), 0
    done = {key(r) for r in load(path) if "error" not in r} if path.exists() else set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for item in todo:
            if key(item) in done:
                continue
            row, began = {k: item[k] for k in ("session", "segment", "index", "skill")}, time.monotonic()
            try:
                row |= work(item)
            except Exception as e:   # the row records the failure, and the run goes on
                row["error"], failed = f"{type(e).__name__}: {e}"[:300], failed + 1
            f.write(json.dumps(row | {"seconds": round(time.monotonic() - began, 1)}) + "\n")
            f.flush()
    return failed


def record(base, model, sessions, tools, n, out, max_tokens=800, from_data=False, sampling=SAMPLING) -> int:
    """Write the recorded turns. ``from_data``: the own turn of each target in the data, with no server."""
    by_id, tool_part = {s["id"]: s for s in sessions}, ({"tools": tools} if tools else {})

    def work(item):
        s, g, i = by_id[item["session"]], item["segment"], item["index"]
        if from_data:
            return {"turn": {k: v for k, v in s["segments"][g][i].items() if k != "target"}, "finish_reason": "data"}
        seed = int(hashlib.sha256(f"{s['id']}:{g}:{i}".encode()).hexdigest()[:8], 16)
        reply = capture.post(base, "/v1/chat/completions", {"model": model, **tool_part, "max_tokens": max_tokens,
                                                            "seed": seed, **sampling,
                                                            "messages": context(s, s["segments"][g], i)})
        m = reply["choices"][0]["message"]
        reasoning = m.get("reasoning_content") or m.get("reasoning")
        return {"turn": {"role": "assistant", "content": m.get("content") or "",
                         **({"reasoning_content": reasoning} if reasoning else {}),
                         **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {})},
                "completion_tokens": reply["usage"]["completion_tokens"],
                "finish_reason": reply["choices"][0]["finish_reason"]}

    todo = [{"session": sid, "segment": g, "index": i, "skill": by_id[sid]["skill"]} for sid, g, i in pick(sessions, n)]
    return _rows(out, todo, work)


def score_tokens(toks, end_think, start_think=None, decision_tokens=1) -> dict:
    """The measures of the turn tokens ``[(token id, info)]``. The reasoning ends after the first ``end_think``.
    With ``start_think``, a turn has reasoning only when that token is one of its first two tokens. The decision is
    right when each of the first ``decision_tokens`` tokens after the reasoning ranks first."""
    cut = next((k + 1 for k, (t, _) in enumerate(toks) if t == end_think), 0)
    if start_think is not None and start_think not in [t for t, _ in toks[:2]]:
        cut = 0
    text, offsets = "", []
    for _, info in toks:
        offsets.append(len(text))
        text += info.get("decoded_token") or ""
    names = []
    for m in NAME.finditer(text):
        a, b = m.span(1)
        inside = [info for (_, info), o in zip(toks, offsets) if o < b and o + len(info.get("decoded_token") or "") > a]
        names.append(all(info["rank"] == 1 for info in inside))
    return {"tokens": len(toks), "nll": -sum(info["logprob"] for _, info in toks),
            "top1": sum(info["rank"] == 1 for _, info in toks), "reasoning_tokens": cut,
            "decision_ok": cut + decision_tokens <= len(toks)
                           and all(info["rank"] == 1 for _, info in toks[cut:cut + decision_tokens]),
            "tool_calls": len(names), "tool_names_ok": sum(names)}


def score(base, model, sessions, tools, recorded, out, end_think="</think>", start_think=None,
          decision_tokens=1) -> int:
    """Score each recorded turn on the server. ``end_think`` and ``start_think`` are the tokens that end and start
    the reasoning (``score_tokens``): the text of a token, or its id for a server that does not read a special
    token from a text."""
    by_id, tool_part = {s["id"]: s for s in sessions}, ({"tools": tools} if tools else {})
    def token(text):
        if isinstance(text, int):
            return text
        ids = capture.post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
        if len(ids) != 1:
            raise ValueError(f"{text} is {len(ids)} tokens on this server: the reasoning needs one token")
        return ids[0]

    end, start_id = token(end_think), token(start_think) if start_think else None

    def work(r):
        before = context(by_id[r["session"]], by_id[r["session"]]["segments"][r["segment"]], r["index"])
        start = len(capture.post(base, "/tokenize", {"model": model, "messages": before, **tool_part,
                                                     "add_generation_prompt": True})["tokens"])
        turn = r["turn"] | ({"reasoning": r["turn"]["reasoning_content"]} if "reasoning_content" in r["turn"] else {})
        reply = capture.post(base, "/v1/chat/completions", {
            "model": model, **tool_part, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
            "add_generation_prompt": False, "messages": [*before, turn]})
        own = reply["prompt_logprobs"][start:]   # one mapping for each token of the turn
        return score_tokens([(int(t), info) for p in own if p for t, info in p.items()], end, start_id,
                            decision_tokens)

    return _rows(out, recorded, work)


def totals(rows) -> dict:
    t = {m: sum(r[m] for r in rows) for m in MEASURES}
    return {"turns": len(rows), "nll": t["nll"] / max(t["tokens"], 1), "top1": t["top1"] / max(t["tokens"], 1),
            "decisions": int(t["decision_ok"]),
            "tool_names": t["tool_names_ok"] / t["tool_calls"] if t["tool_calls"] else 1.0}


def compare(ref_rows, new_rows, limits=LIMITS) -> dict:
    """The totals of the reference and of the new model on the turns that both scored, and the result of each gate."""
    ref, new = ({key(r): r for r in rows if "error" not in r and r["tokens"]} for rows in (ref_rows, new_rows))
    common = sorted(ref.keys() & new.keys())
    a, b = totals([ref[k] for k in common]), totals([new[k] for k in common])
    nll, tools = limits["nll"], limits["tools"]
    return {"reference": a, "new": b, "nll_change_pct": 100 * (b["nll"] / a["nll"] - 1) if a["nll"] else 0.0, "gates": {
        f"nll: at most +{nll['limit_pct']:g}%": b["nll"] <= a["nll"] * (1 + nll["limit_pct"] / 100),
        f"tools: the names at least {tools['names_min_pct']:g}% of the reference":
            b["tool_names"] >= a["tool_names"] * tools["names_min_pct"] / 100,
        f"tools: the decisions down by at most {tools['decisions_max_drop']}":
            b["decisions"] >= a["decisions"] - tools["decisions_max_drop"]}}


def text(result: dict) -> str:
    a, b = result["reference"], result["new"]
    lines = [f"{'':<12}{'reference':>12}{'new':>12}"] + [f"{m:<12}{a[m]:>12.4f}{b[m]:>12.4f}" for m in a]
    lines.append(f"nll change  {result['nll_change_pct']:+.2f}%")
    return "\n".join(lines + [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in result["gates"].items()])
