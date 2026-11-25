"""The preflight, level 0 (plans/PLAN_PRUNE_KIT.md §5.4): what each cut frees, and whether the pipeline can run.

It uses the facts of the files only. It loads no weights, and it gives no estimate of the depth that a model
tolerates: only a measurement gives that.
"""

import math
from dataclasses import asdict, dataclass

from . import sessions
from .facts import Facts

GIB, MIB = 1 << 30, 1 << 20
# The model types of the two prunes by hand, with the tool of each stage in spark-moe-specialize.
BY_HAND = {
    "qwen3_5_moe": {"prune": "tools/qwen_prune.py", "draft": "qwen_mtp_patch/ keeps the count of the draft",
                    "capture": "the flag --enable-return-routed-experts of vLLM 0.33"},
    "deepseek_v4": {"prune": "tools/prune_experts.py", "draft": "the target skips the draft tensors",
                    "capture": "the server patch of image-patch/ with Model Runner V1"},
}
PROVEN = {"qwen3_5_moe"}   # st_moe gives the folder of the prune by hand for these model types (gate B)
WORK = ("no", "unknown")   # the states that need work before the pipeline can run
# What a dense model can do in place of a prune, with the result of each lever in spark-moe-specialize.
LEVERS = (
    ("quantization", "a smaller format of the same weights, for example NVFP4 in place of FP8"),
    ("draft model", "a 4-bit copy of the draft: 1.09 GiB on Qwen, with no change to the output of the target"),
    ("context", "a smaller KV cache pin, where the KV need grows with the context: 0.75 GiB on DeepSeek"),
    ("layers", "not a lever: on DeepSeek each removed layer cost +2.15% NLL or more"),
)
HELD = "the two prunes by hand: Qwen holds a cut of 19% of its experts with a cost, and DeepSeek holds 31%"


@dataclass
class Check:
    name: str
    state: str   # "yes", "no", "unknown", or "info" for a fact that blocks nothing
    note: str


def default_ks(experts: int) -> list[int]:
    """The cuts of the table: steps of about 1/16 of the experts, down to 5/8 of them."""
    step = max(1, 2 ** round(math.log2(experts / 16))) if experts else 1
    return [k for k in range((experts - 1) // step * step, 0, -step) if k >= experts * 5 / 8]


def available_bytes() -> int | None:
    """``MemAvailable`` of the machine, or None where ``/proc/meminfo`` does not exist."""
    try:
        with open("/proc/meminfo") as f:
            return next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return None


def prune_tool(f: Facts, hand: str) -> Check:
    """Can ``prune-kit build`` (st_moe) write the pruned folder of this model?"""
    if f.hash_layers or f.unread_files:
        why = "the token tables of the hash layers" if f.hash_layers else "weight files with no read permission"
        return Check("prune tool", "no", f"prune-kit build does not handle {why}{hand}")
    if f.model_type in PROVEN:
        return Check("prune tool", "yes", "prune-kit build: it gives the folder of the prune by hand, byte for byte")
    return Check("prune tool", "unknown", f"prune-kit build copies the {f.layout} layout, "
                                          "but no server has loaded such a copy of this model type")


def checks(f: Facts, available: int | None) -> list[Check]:
    """The answers to "can the pipeline run?" that the files give."""
    if not f.is_moe:
        return [Check("routed experts", "no", "the model is dense: the prune of the experts does not apply")]
    known = BY_HAND.get(f.model_type, {})
    hand = lambda stage: f"; by hand: {known[stage]}" if stage in known else ""
    low, high = min(f.experts.values()), f.max_experts
    count = f"{high}" if low == high else f"{low} to {high}"
    out = [
        Check("routed experts", "yes", f"{count} per layer, {f.per_token or '?'} per token"),
        prune_tool(f, hand("prune")),
        Check("capture", "yes" if "capture" in known else "unknown",
              known.get("capture", "not proven for this model type: test the capture flag of its server")),
        Check("mask", "info", "possible: the router has a bias" if f.router_bias
              else "the router has no bias: screen with small prunes"),
    ]
    if f.draft is None:
        out.append(Check("draft model", "info", "no draft in the folder"))
    elif f.draft.experts is None:
        out.append(Check("draft model", "info", "dense: no expert count to keep; test its acceptance"))
    else:
        out.append(Check("draft model", "yes" if "draft" in known else "unknown",
                         f"{f.draft.experts} experts of its own; "
                         + known.get("draft", "the server must keep that count beside a pruned target")))
    if f.hash_layers:
        out.append(Check("hash routing", "info", f"layers 0 to {f.hash_layers - 1} route by a token table: "
                                                 "a cut must move the entries of a dropped expert"))
    if f.unread_files:
        out.append(Check("sizes", "info", f"{f.unread_files} weight files have no read permission: "
                                          "the sizes come from the file sizes"))
    if f.loop:
        first, end, passes = f.loop
        out.append(Check("loop", "info", f"layers {first} to {end - 1} run {passes} times: "
                                         "the counts of such a layer are the sum of the passes"))
    if available is not None:
        need = "needs a window" if available < f.total_bytes else "fits now, before the KV cache and the overhead"
        out.append(Check("memory", "info", f"the full model loads {f.total_bytes / GIB:.2f} GiB, "
                                           f"and {available / GIB:.1f} GiB is available: {need}"))
    return out


def verdict(f: Facts, items: list[Check]) -> str:
    if not f.is_moe:
        return "NO-GO"
    work = [c.name for c in items if c.state in WORK]
    return "GO WITH WORK: " + ", ".join(work) if work else "GO"


def need(f: Facts, workflow) -> dict | None:
    """The cut that reaches the peak goal of the workflow: the largest K that frees the need. None with no goal."""
    goal = workflow.need_bytes(f.total_bytes)
    if goal is None:
        return None
    k = next((k for k in range(f.max_experts, (f.per_token or 1) - 1, -1) if f.freed(k) >= goal), None)
    return {**workflow.memory, "bytes": goal, "k": k, "share": (1 - k / f.max_experts) if k else None}


def data(workflow) -> tuple[dict | None, Check]:
    """The sessions of the workflow against its minimums: the report of ``sessions.check``, and the line of the list."""
    path = workflow.folder / "sessions.jsonl"
    if not path.exists():
        return None, Check("data", "no", f"{path} does not exist: build the sessions of the workflow")
    d = workflow.data
    r = sessions.check(sessions.load(path), d["min_calibration_tokens"], d["min_held_out_turns"])
    note = (f"{r['calibration_tokens']:,} calibration tokens (minimum {d['min_calibration_tokens']:,}), "
            f"{r['held_out_turns']} held-out turns (minimum {d['min_held_out_turns']})")
    return r, Check("data", "yes" if r["ok"] else "no", note)


def report(f: Facts, ks=None, available: int | None = None, workflow=None) -> dict:
    """The preflight of one model as plain data. ``text`` prints it."""
    items = checks(f, available)
    built = None
    if workflow:
        built, line = data(workflow)
        items.append(line)
    return {
        "data": built,
        "workflow": workflow.name if workflow else None, "need": need(f, workflow) if workflow else None,
        "folder": str(f.folder), "model_type": f.model_type, "moe": f.is_moe, "layout": f.layout, "quant": f.quant,
        "layers": len(f.experts), "experts_min": min(f.experts.values(), default=0), "experts_max": f.max_experts,
        "experts_total": sum(f.experts.values()), "per_token": f.per_token, "loop": f.loop,
        "total_bytes": f.total_bytes, "routed_bytes": f.routed_bytes, "draft_bytes": f.draft.bytes if f.draft else 0,
        "cuts": [{"k": k, "freed_bytes": f.freed(k), "share": f.freed(k) / f.total_bytes}
                 for k in (ks or default_ks(f.max_experts))],
        "checks": [asdict(c) for c in items], "verdict": verdict(f, items),
    }


def text(r: dict) -> str:
    kind = f"MoE, {r['layout']} layout" if r["moe"] else "dense"
    lines = [f"model     {r['folder']}",
             f"type      {r['model_type']}: {kind}" + (f", {r['quant']}" if r["quant"] else ""),
             f"weights   {r['total_bytes'] / GIB:.2f} GiB in the folder; draft {r['draft_bytes'] / GIB:.2f} GiB"]
    if r["moe"]:
        per_expert = r["routed_bytes"] / r["experts_total"] / MIB
        lines += [f"experts   {r['checks'][0]['note']}, in {r['layers']} layers; {per_expert:.2f} MiB per expert",
                  f"routed    {r['routed_bytes'] / GIB:.2f} GiB "
                  f"({r['routed_bytes'] / r['total_bytes']:.0%} of the weights)",
                  "", "cut       freed        of the weights"]
        lines += [f"K {c['k']:<7} {c['freed_bytes'] / GIB:5.2f} GiB    {c['share']:.0%}" for c in r["cuts"]]
    if n := r.get("need"):
        at = f" at a context of {n['context']:,} tokens" if n["context"] else ""
        lines += ["", f"goal      a peak of {n['peak_gib']:g} GiB{at}, for the workflow {r['workflow']}"]
        if n["bytes"] <= 0:
            lines.append("need      no cut: the model fits the goal")
        elif n["k"] is None:
            lines.append(f"need      free {n['bytes'] / GIB:.2f} GiB: no cut of the experts reaches it")
        else:
            lines += [f"need      free {n['bytes'] / GIB:.2f} GiB: K {n['k']} or lower "
                      f"({n['share']:.0%} of the experts)", f"          {HELD}"]
        if n["overhead_gib"] is None:
            lines.append("          a lower limit: memory.overhead_gib is not set, so the need has the weights only, "
                         "and not the KV pin or the overhead of the server")
    if r.get("data"):
        lines += ["", "data      the token counts are estimates: the characters divided by 4"]
        lines += ["  " + row for row in sessions.table(r["data"])]
    lines += ["", "can the pipeline run?"]
    lines += [f"  {c['name']:<15} {c['state']:<8} {c['note']}" for c in r["checks"]]
    if not r["moe"]:
        lines += ["", "other levers"] + [f"  {name:<15} {note}" for name, note in LEVERS]
    return "\n".join(lines + [f"verdict   {r['verdict']}"])
