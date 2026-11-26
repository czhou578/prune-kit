"""The gates ``tasks``, ``speed`` and ``memory`` (plans/PLAN_PRUNE_KIT.md §5.6), and the result of one point.

- ``tasks``: the tool-call suite of ``model-benchmarks`` (85 tasks) writes one JSON file for each test server. The
  gate compares the passed tasks of a point with those of the full model.
- ``speed``: ``measure_speed`` asks a server for eight fixed answers and measures the decode speed. The report shows
  the speed and its change against the full model for the same prompts, and it has no limit. One prompt differs from
  the next by 10% and more, so the change is the median of the changes of the prompts.
- ``memory``: the results file of the window holds the available memory before the boot of a point and the lowest
  value while its clients ran. The difference is the peak of the point.

``verdict`` puts the gates of one point together, as §5.6 says.
"""

import json
import time
import urllib.request
from pathlib import Path
from statistics import median

PROMPTS = ("Write a Python function that merges two sorted lists into one sorted list. Then explain each step.",
           "Explain how a hash map handles a collision. Give one short example in C.",
           "Write a SQL query that returns the three customers with the highest total order value, and explain it.",
           "Write a shell script that finds the ten largest files below a folder. Then explain each command.",
           "Explain the difference between a process and a thread. Give one example of a race condition.",
           "Write a JavaScript function that removes the duplicates of an array and keeps the order. Explain it.",
           "Explain what a database index is, and when an index makes a query slower.",
           "Write a Python class for a queue with a maximum size, with tests for the empty and the full queue.")


def stream(base: str, body: dict):
    """The events of a streamed chat completion, as parsed JSON."""
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as reply:
        for line in reply:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                yield json.loads(line[6:])


def measure_speed(base: str, model: str, out, max_tokens: int = 400, stream=stream, clock=time.monotonic) -> dict:
    """Write the decode speed of the server to ``out``: the tokens after the first one, divided by the time between
    the first and the last token. The result is the median of the prompts."""
    runs = []
    for n, prompt in enumerate(PROMPTS):
        first = last = tokens = None
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        for event in stream(base, body):
            delta = (event.get("choices") or [{}])[0].get("delta") or {}
            if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                last = clock()
                first = first if first is not None else last
            tokens = (event.get("usage") or {}).get("completion_tokens") or tokens
        if tokens and tokens > 1 and last > first:
            runs.append({"prompt": n, "tokens": tokens, "seconds": round(last - first, 3),
                         "speed": (tokens - 1) / (last - first)})
    result = {"runs": runs, "tokens_per_second": round(median(r["speed"] for r in runs), 2) if runs else None}
    Path(out).write_text(json.dumps(result, indent=1))
    return result


def speed_change(ref: dict | None, new: dict) -> float | None:
    """The median change of the speed in percent, over the prompts that both servers answered."""
    a, b = ({r["prompt"]: r["speed"] for r in (d or {}).get("runs", []) if "prompt" in r} for d in (ref, new))
    common = sorted(a.keys() & b.keys())
    return round(median(100 * (b[n] / a[n] - 1) for n in common), 1) if common else None


def read(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def tasks(ref: dict | None, new: dict | None, max_drop: int) -> dict | None:
    """The tool-call suite of a point against the full model: at most ``max_drop`` tasks fewer."""
    if not ref or not new:
        return None
    return {"reference": ref["total_correct"], "new": new["total_correct"], "of": new["total_tasks"],
            "ok": new["total_correct"] >= ref["total_correct"] - max_drop}


def memory(window: dict | None, name: str, goal_gib: float | None) -> dict | None:
    """The peak of a point in the window, against the peak goal of the workflow."""
    point = next((p for p in (window or {}).get("points", []) if p["name"] == name and p["booted"]), None)
    if point is None:
        return None
    peak = round(point["available_before_gib"] - point["available_lowest_gib"], 2)
    return {"peak_gib": peak, "goal_gib": goal_gib, "ok": None if goal_gib is None else peak <= goal_gib}


def point(run, name: str, workflow) -> dict:
    """The gates ``tasks``, ``speed`` and ``memory`` of the point ``name`` in a run folder. None: not measured."""
    run, limits = Path(run), workflow.gates
    speed, base = read(run / f"speed-{name}.json"), read(run / "speed-full.json")
    suite = tasks(read(run / "tools-full.json"), read(run / f"tools-{name}.json"), limits["tasks"]["max_drop"])
    return {"tasks": suite,
            "memory": memory(read(run / "window.json"), name, workflow.memory["peak_gib"]),
            "speed": speed and {"tokens_per_second": speed["tokens_per_second"],
                                "reference": base and base["tokens_per_second"],
                                "change_pct": speed_change(base, speed)},
            "judge": read(run / f"judge-{name}.json")}


def verdict(nll_gate: str, tools_ok: bool, more: dict) -> str:
    """``pass``: each gate passes. ``your decision``: only ``nll`` does not pass, and ``tasks`` and ``judge`` have a
    result. ``fail``: another gate fails. ``open``: ``nll`` does not pass, and ``tasks`` or ``judge`` has no result."""
    results = {name: more[name] and more[name]["ok"] for name in ("tasks", "memory", "judge")}
    if not tools_ok or any(ok is False for ok in results.values()):
        return "fail"
    if nll_gate == "pass":
        return "pass"
    return "your decision" if results["tasks"] and results["judge"] else "open"
