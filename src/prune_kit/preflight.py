"""The preflight, level 0 (plans/PLAN_PRUNE_KIT.md §5.4): what each cut frees, and whether the pipeline can run.

It uses the facts of the files only. It loads no weights, and it gives no estimate of the depth that a model
tolerates: only a measurement gives that.
"""

import math
from dataclasses import asdict, dataclass

from .facts import Facts

GIB, MIB = 1 << 30, 1 << 20
# The model types of the two prunes by hand, with the tool of each stage in spark-moe-specialize.
BY_HAND = {
    "qwen3_5_moe": {"prune": "tools/qwen_prune.py", "draft": "qwen_mtp_patch/ keeps the count of the draft",
                    "capture": "the flag --enable-return-routed-experts of vLLM 0.33"},
    "deepseek_v4": {"prune": "tools/prune_experts.py", "draft": "the target skips the draft tensors",
                    "capture": "the server patch of image-patch/ with Model Runner V1"},
}
WORK = ("no", "unknown")   # the states that need work before the pipeline can run


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
        Check("prune tool", "no", f"prune-kit has no prune for the {f.layout} layout yet{hand('prune')}"),
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


def report(f: Facts, ks=None, available: int | None = None) -> dict:
    """The preflight of one model as plain data. ``text`` prints it."""
    items = checks(f, available)
    return {
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
    lines += ["", "can the pipeline run?"]
    lines += [f"  {c['name']:<15} {c['state']:<8} {c['note']}" for c in r["checks"]]
    return "\n".join(lines + [f"verdict   {r['verdict']}"])
