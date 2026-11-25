"""The capture client (plans/PLAN_PRUNE_KIT.md §4, stage 2): count the routed experts of the calibration sessions.

The server must return the routed experts. On vLLM 0.33 that is the flag ``--enable-return-routed-experts``. For
each segment of the calibration split, one ``/v1/chat/completions`` request sends the context and the last target
turn of the segment, with ``max_tokens: 1``. The reply holds ``choices[0].routed_experts``: a base64 ``.npy`` array
of the shape (prompt tokens, layers, experts per token).

- The rows of the shared prefix (the system prompt and the tools) count once, the first time the prefix appears.
- A row of equal ids is a token that the capture did not fill. It does not count.
- ``generate``: the model also writes some tokens after the segment, and their routing counts.

Each segment writes one row of counts to the output file, so a second start continues after the last row.
"""

import ast
import base64
import hashlib
import json
import struct
import time
import urllib.request
from array import array
from collections import Counter
from pathlib import Path

STRIP = ("target", "reasoning_content")
CODES = {"<i2": "h", "<u2": "H", "<i4": "i", "<u4": "I", "<i8": "q", "<u8": "Q", "|u1": "B", "|i1": "b"}


class CaptureError(Exception):
    pass


def post(base: str, path: str, body: dict, timeout: int = 3600) -> dict:
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def decode(b64: str) -> tuple[array, tuple]:
    """The values and the shape of a base64 ``.npy`` array of integers."""
    raw = base64.b64decode(b64)
    if raw[:6] != b"\x93NUMPY":
        raise CaptureError("routed_experts is not a .npy array")
    wide = raw[6] > 1
    start = (12 if wide else 10) + struct.unpack("<I" if wide else "<H", raw[8:12 if wide else 10])[0]
    head = ast.literal_eval(raw[12 if wide else 10:start].decode("latin1"))
    if head["descr"] not in CODES or head["fortran_order"]:
        raise CaptureError(f"routed_experts has the type {head['descr']}, which the capture does not read")
    values = array(CODES[head["descr"]])
    values.frombytes(raw[start:])
    return values, tuple(head["shape"])


def count(values, shape, start: int = 0, experts: int | None = None) -> tuple[list, int]:
    """The counts of each expert in each layer for the rows from ``start`` on, and the number of unfilled rows."""
    _, layers, per_token = shape
    step, counts, unfilled = layers * per_token, [], 0
    for layer in range(layers):
        cols = [values[start * step + layer * per_token + j::step] for j in range(per_token)]
        c = Counter()
        for col in cols:
            c.update(col)
        empty = [row[0] for row in zip(*cols) if row.count(row[0]) == per_token] if per_token > 1 else []
        c.subtract({e: per_token * n for e, n in Counter(empty).items()})
        unfilled += len(empty)
        counts.append([c.get(e, 0) for e in range(experts or max(c, default=-1) + 1)])
    return counts, unfilled


def chat_body(model, session, seg, i, tools, extra) -> dict:
    """The request whose prompt ends with the target ``seg[i]``, with its reasoning."""
    context = [{k: v for k, v in m.items() if k not in STRIP} for m in seg[:i]]
    target = {k: v for k, v in seg[i].items() if k not in STRIP} | {"reasoning": seg[i].get("reasoning_content") or ""}
    system = [{"role": "system", "content": session["system_prompt"]}] if session.get("system_prompt") else []
    return {"model": model, **({"tools": tools} if tools else {}), "max_tokens": 1, "temperature": 0,
            "add_generation_prompt": False, "return_token_ids": True, "messages": [*system, *context, target], **extra}


def routed(reply: dict) -> tuple[array, tuple, list]:
    """The routing rows of the prompt of a reply, and the prompt tokens."""
    choice, prompt = reply["choices"][0], reply["prompt_token_ids"]
    if not choice.get("routed_experts"):
        raise CaptureError("the reply has no routed_experts: boot the server with its capture flag")
    values, shape = decode(choice["routed_experts"])
    if len(shape) != 3 or shape[0] < len(prompt):
        raise CaptureError(f"routing of the shape {shape} for {len(prompt)} prompt tokens")
    return values, shape, prompt   # the server also returns rows of the new tokens: ``count`` gets the shape


def common_prefix(a, b) -> int:
    return next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def capture_one(base, model, s, g, i, tools, seen, experts, extra=None, generate=0, post=post) -> dict:
    """The row of one segment. ``seen`` holds the keys of the prefixes that an earlier row counted."""
    extra, seg = extra or {}, s["segments"][g]
    body = chat_body(model, s, seg, i, tools, extra)
    values, shape, prompt = routed(post(base, "/v1/chat/completions", body))
    values = values[:len(prompt) * shape[1] * shape[2]]   # without the row of the one new token
    # The prefix is the system prompt and the tools. A user message closes it, because the template of Qwen refuses
    # a list with no user message. The common prefix stops where the two token lists differ.
    system = [m for m in body["messages"] if m["role"] == "system"] or [{"role": "system", "content": ""}]
    kwargs = extra.get("chat_template_kwargs", {}) | {k: extra[k] for k in ("reasoning_effort",) if k in extra}
    shared = post(base, "/tokenize", {"model": model, "messages": [*system, {"role": "user", "content": "."}],
                                      **({"tools": tools} if tools else {}), "add_generation_prompt": False,
                                      **({"chat_template_kwargs": kwargs} if kwargs else {})})["tokens"]
    prefix = common_prefix(shared, prompt)
    key = hashlib.sha256(json.dumps(prompt[:prefix]).encode()).hexdigest()
    counts, unfilled = count(values, shape, 0 if key not in seen else prefix, experts)
    row = {"prompt_tokens": len(prompt), "prefix_tokens": prefix, "prefix": key, "prefix_counted": key not in seen,
           "layers": shape[1], "top_k": shape[2], "unfilled_rows": unfilled, "counts": counts}
    if generate:
        # The context is the segment up to its last target, or the whole segment when messages follow that target.
        context = seg[:i] if i == len(seg) - 1 else seg
        more = chat_body(model, s, [*context, {}], len(context), tools, extra)
        more["messages"].pop()
        more.update(max_tokens=generate, add_generation_prompt=True, temperature=0.6)
        reply = post(base, "/v1/chat/completions", more)
        values, shape, again = routed(reply)   # the rows that the first request did not cover count
        counts, unfilled = count(values, shape, common_prefix(prompt, again), experts)
        row |= {"gen_counts": counts, "gen_unfilled": unfilled, "gen_tokens": reply["usage"]["completion_tokens"]}
    return row


def targets(sessions, split: str = "calibration"):
    """``(session, segment index, index of the last target)`` for each segment of the split."""
    for s in sessions:
        for g, seg in enumerate(s["segments"] if s["split"] == split else []):
            last = max((i for i, m in enumerate(seg) if m.get("target")), default=None)
            if last is not None:
                yield s, g, last


def run(base, model, sessions, tools, out, experts, limit=None, extra=None, generate=0, post=post, log=print) -> int:
    """Capture each calibration segment that ``out`` does not hold yet. A failed segment gets a row with the
    error, and the run goes on. Return the number of failed segments."""
    out = Path(out)
    rows = [json.loads(line) for line in out.open()] if out.exists() else []
    done = {(r["session"], r["segment"]) for r in rows if "error" not in r}
    seen = {r["prefix"] for r in rows if r.get("prefix_counted")}
    todo, failed = list(targets(sessions))[:limit], 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for n, (s, g, i) in enumerate(todo, 1):
            if (s["id"], g) in done:
                continue
            row, began = {"session": s["id"], "segment": g, "index": i, "skill": s["skill"]}, time.monotonic()
            try:
                row |= capture_one(base, model, s, g, i, tools, seen, experts, extra, generate, post)
                seen.add(row["prefix"])
            except Exception as e:   # the row records the failure
                row["error"], failed = f"{type(e).__name__}: {e}"[:300], failed + 1
            row["seconds"] = round(time.monotonic() - began, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            log(f"[{n}/{len(todo)}] {s['skill']:<24} {row.get('prompt_tokens', '-'):>7} tokens {row['seconds']:>6} s "
                f"{row.get('error', '')}")
    return failed
