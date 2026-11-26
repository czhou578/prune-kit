"""The rank (plans/PLAN_PRUNE_MOE_KIT.md §4, stage 3): the experts of each layer, ordered by their use in a capture.

``rank`` sums the rows of ``capture.py`` into a rank file: for each layer, each expert with its count and its share
of the routing, the most used first. ``prune-moe-kit build`` keeps the first K experts of each layer of this file.
``overlap`` ranks two halves of the capture apart and compares their drop lists: a low overlap shows that the
capture is too small for a stable rank.
"""

import hashlib


class RankError(Exception):
    pass


def fold(total: list, loop) -> list:
    """The counts of each layer of the weights, for a model whose layers ``start`` to ``end - 1`` run ``passes``
    times. The server gives one row for each pass of a layer, in the order of the run: the layers before the loop,
    each pass of the loop, the layers after it. The counts of a layer are the sum of its passes."""
    start, end, passes = loop
    core = end - start
    if len(total) < start + passes * core:
        raise RankError(f"the capture has {len(total)} layer rows, fewer than the {passes} passes of the loop")
    out = [[0] * len(total[0]) for _ in range(len(total) - (passes - 1) * core)]
    for visit, row in enumerate(total):
        after = visit >= start + passes * core
        layer = visit if visit < start else visit - (passes - 1) * core if after else start + (visit - start) % core
        out[layer] = [a + b for a, b in zip(out[layer], row)]
    return out


def totals(rows) -> list:
    """The counts of each expert in each layer, over each row with no error. The counts of generated tokens add.
    The rows of a model with a loop hold the key ``loop``: see ``fold``."""
    rows = [r for r in rows if "error" not in r]
    if not rows:
        raise RankError("the capture has no segment with counts")
    total = [[0] * len(layer) for layer in rows[0]["counts"]]
    for r in rows:
        for counts in (r["counts"], r.get("gen_counts")):
            if counts and list(map(len, counts)) != list(map(len, total)):
                raise RankError(f"segment {r.get('session')}:{r.get('segment')} has another count of layers or experts")
            for layer, row in enumerate(counts or []):
                total[layer] = [a + b for a, b in zip(total[layer], row)]
    return fold(total, rows[0]["loop"]) if rows[0].get("loop") else total


def rank(rows) -> dict:
    """The rank file of a capture. Experts with equal counts keep the order of their ids."""
    good = [r for r in rows if "error" not in r]
    layers = {}
    for layer, counts in enumerate(totals(rows)):
        whole = sum(counts) or 1
        layers[str(layer)] = [{"id": e, "count": counts[e], "share": round(counts[e] / whole, 6)}
                              for e in sorted(range(len(counts)), key=lambda e: -counts[e])]
    return {"segments": len(good), "prompt_tokens": sum(r["prompt_tokens"] for r in good),
            "unfilled_rows": sum(r["unfilled_rows"] for r in good), "order": "count", "layers": layers}


def removed(ranked: dict, k: int) -> float:
    """The share of the routing that a cut to ``k`` experts removes: the mean of the layers."""
    count = lambda experts: sum(e["count"] for e in experts)
    layers = ranked["layers"].values()
    return sum(count(layer[k:]) / (count(layer) or 1) for layer in layers) / len(layers)


def halves(rows) -> tuple[list, list]:
    """Two halves of the segments, in a fixed order by a hash, so that each session gives to both halves."""
    good = sorted((r for r in rows if "error" not in r),
                  key=lambda r: hashlib.sha256(f"{r['session']}:{r['segment']}".encode()).hexdigest())
    return good[::2], good[1::2]


def overlap(rows, ks) -> dict:
    """``{k: share}``: the share of the dropped experts that the ranks of the two halves both drop, as the mean of
    the layers. 1.0 means equal drop lists."""
    a, b = (rank(half)["layers"] for half in halves(rows))
    out = {}
    for k in ks:
        shares = [len({e["id"] for e in a[layer][k:]} & {e["id"] for e in b[layer][k:]}) / max(1, len(a[layer]) - k)
                  for layer in a]
        out[k] = sum(shares) / len(shares)
    return out


def text(ranked: dict, ks, shares: dict | None) -> str:
    experts = len(ranked["layers"]["0"])
    lines = [f"capture   {ranked['segments']} segments, {ranked['prompt_tokens']:,} prompt tokens, "
             f"{ranked['unfilled_rows']} unfilled rows",
             f"experts   {experts} per layer, in {len(ranked['layers'])} layers", "",
             "cut       routing removed    overlap of two halves"]
    for k in ks:
        both = f"{shares[k]:.0%}" if shares else "one segment only"
        lines.append(f"K {k:<7} {removed(ranked, k):>8.2%}           {both}")
    return "\n".join(lines)
