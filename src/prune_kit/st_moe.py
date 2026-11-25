"""``st_moe``: the prune of a safetensors checkpoint with routed experts (plans/PLAN_PRUNE_KIT.md §5.3).

It covers the two layouts of ``facts.py``. In the per-expert layout, each expert of a layer has its own tensors
(``...layers.<L>...experts.<E>.<rest>``). In the stacked layout, one tensor holds each expert of a layer in its first
dimension (``...layers.<L>...experts.<name>``). The router is ``...layers.<L>...gate.weight``, with one row per
expert, and in some models ``gate.bias``.

``prune`` writes a copy of the checkpoint with K experts per layer. It keeps the first K experts of each layer of a
rank file, gives them the new ids 0 to K-1 in their old order, keeps the same rows of the router, and sets the expert
count of the config. It copies bytes and does no new quantization. It never changes its source: the output is a new
folder. ``check`` compares each tensor of a pruned folder with its source, byte for byte.
"""

import json
import re
import shutil
import struct
from pathlib import Path

EXPERT = re.compile(r"^((?:.*\.)?layers\.(\d+)\.(?:[A-Za-z_]\w*\.)*experts\.)(\d+)(\..+)$")
STACK = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(?:[A-Za-z_]\w*\.)*experts\.[A-Za-z_]\w*$")
ROUTER = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(?:[A-Za-z_]\w*\.)*gate\.(?:weight|bias)$")
COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
NOT_COPIED = ("config.json", "model.safetensors.index.json")
CHUNK = 64 << 20   # the largest piece of a tensor in memory


class PruneError(Exception):
    pass


def header(path) -> tuple[dict, int]:
    """The header of a safetensors file and the offset of its data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def key(layer: str, name: str, file: str):
    """The key of a layer: its number, or ``mtp.<number>`` for a layer of the draft model."""
    return f"mtp.{layer}" if name.startswith("mtp.") or file.startswith("mtp") else int(layer)


def counts(source) -> dict:
    """``{layer key: experts}`` for each layer with routed experts."""
    n = {}
    for f in sorted(Path(source).glob("*.safetensors")):
        for name, h in header(f)[0].items():
            if m := EXPERT.match(name):
                k, count = key(m[2], name, f.name), int(m[3]) + 1
            elif (m := STACK.match(name)) and h["shape"]:
                k, count = key(m[1], name, f.name), h["shape"][0]
            else:
                continue
            n[k] = max(n.get(k, 0), count)
    return n


def keep_sets(rank: dict, k: int, n: dict, draft: str = "rank") -> dict:
    """``{layer key: sorted kept ids}``: the first ``k`` experts of each layer of the rank file. The rank of draft
    layer j is the rank layer after the last layer of the model, plus j. ``draft="first"`` keeps the experts 0 to
    k-1 of the draft. ``draft="keep"`` keeps the draft whole (None): serve the draft from the source folder."""
    main = sorted(key for key in n if isinstance(key, int))
    keep = {}
    for layer in n:
        row = layer if isinstance(layer, int) else len(main) + int(layer[4:])
        if not isinstance(layer, int) and draft != "rank":
            keep[layer] = None if draft == "keep" else list(range(k))
        elif str(row) not in rank["layers"]:
            raise PruneError(f"the rank file has no layer {row}" + (
                "" if isinstance(layer, int) else " for the draft model: capture it, or use the draft mode first"))
        else:
            keep[layer] = sorted({e["id"] for e in rank["layers"][str(row)][:k]})
        if keep[layer] is not None and (len(keep[layer]) != k or keep[layer][-1] >= n[layer]):
            raise PruneError(f"layer {layer}: the rank file does not give {k} experts of the {n[layer]}")
    return keep


def plan(name: str, file: str, shape: list, keep: dict, n: dict):
    """What happens to a tensor: ``(new name, rows)``. The new name is None for a dropped expert. ``rows`` are the
    kept indexes of the first dimension, or None for a plain copy."""
    if m := EXPERT.match(name):
        ids = keep.get(key(m[2], name, file))
        if ids is None:
            return name, None
        return (f"{m[1]}{ids.index(int(m[3]))}{m[4]}" if int(m[3]) in ids else None), None
    if m := STACK.match(name) or ROUTER.match(name):
        layer = key(m[1], name, file)
        if keep.get(layer) is not None and shape and shape[0] == n[layer]:   # a tensor with one value stays whole
            return name, keep[layer]
    return name, None


def tensors_of(path, keep: dict, n: dict):
    """``(new name, dtype, new shape, ranges)`` for each kept tensor of a file, in file order. ``ranges`` are the
    ``(offset, size)`` parts of the source file. Kept rows that follow each other are one range."""
    head, start = header(path)
    head.pop("__metadata__", None)
    for name in sorted(head, key=lambda name: head[name]["data_offsets"][0]):
        h = head[name]
        new, rows = plan(name, Path(path).name, h["shape"], keep, n)
        if new is None:
            continue
        offset, size, shape = start + h["data_offsets"][0], h["data_offsets"][1] - h["data_offsets"][0], h["shape"]
        ranges = [(offset, size)]
        if rows is not None:
            row, ranges, shape = size // shape[0], [], [len(rows), *shape[1:]]
            for r in rows:
                if ranges and sum(ranges[-1]) == offset + r * row:
                    ranges[-1] = (ranges[-1][0], ranges[-1][1] + row)
                else:
                    ranges.append((offset + r * row, row))
        yield new, h["dtype"], shape, ranges


def pieces(f, ranges):
    """The bytes of ``ranges`` of an open file, in pieces of at most ``CHUNK`` bytes."""
    for offset, size in ranges:
        f.seek(offset)
        while size:
            piece = f.read(min(size, CHUNK))
            if not piece:
                raise PruneError(f"{f.name}: the file ends before offset {offset} + {size}")
            yield piece
            size -= len(piece)


def write(path, source, items, metadata=None) -> int:
    """Write the ``tensors_of`` items of the file ``source`` in this order, with no gap. Return the data bytes."""
    head, offset = ({"__metadata__": metadata} if metadata else {}), 0
    for name, dtype, shape, ranges in items:
        size = sum(r[1] for r in ranges)
        head[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(head, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as out, open(source, "rb") as f:
        out.write(struct.pack("<Q", len(raw)) + raw)
        for _, _, _, ranges in items:
            for piece in pieces(f, ranges):
                out.write(piece)
    return offset


def set_count(config, old: int, k: int) -> int:
    """Set each expert count of the config (at each depth) from ``old`` to ``k``. Return the number of changes."""
    changed = 0
    for name, value in config.items():
        if isinstance(value, dict):
            changed += set_count(value, old, k)
        elif name in COUNT_KEYS and value == old:
            config[name], changed = k, changed + 1
    return changed


def prune(source, out, rank: dict, k: int, draft: str = "rank") -> dict:
    """Write the pruned folder ``out``. Return the tensors kept, the tensors dropped and the bytes written."""
    source, out = Path(source), Path(out)
    n = counts(source)
    experts = {count for layer, count in n.items() if isinstance(layer, int)}
    if len(experts) != 1:
        raise PruneError(f"{source}: the layers must have one count of routed experts, and they have {sorted(experts)}")
    experts = experts.pop()
    if not 0 < k < experts:
        raise PruneError(f"K must be between 1 and {experts - 1}")
    keep = keep_sets(rank, k, n, draft)
    config = json.loads((source / "config.json").read_text())
    if not set_count(config, experts, k):
        raise PruneError(f"config.json has no expert count of {experts} under the keys {', '.join(COUNT_KEYS)}")
    files = {f: list(tensors_of(f, keep, n)) for f in sorted(source.glob("*.safetensors"))}
    need = sum(size for items in files.values() for *_, ranges in items for _, size in ranges)
    home = next(p for p in (out, *out.parents) if p.exists())
    if shutil.disk_usage(home).free < need * 1.02:
        raise PruneError(f"{home} has less free space than the {need / 2**30:.2f} GiB of the output")
    out.mkdir(parents=True, exist_ok=False)
    weight_map, dropped = {}, 0
    for f, items in files.items():
        head = header(f)[0]
        write(out / f.name, f, items, head.get("__metadata__"))
        weight_map.update({item[0]: f.name for item in items})
        dropped += len(head) - ("__metadata__" in head) - len(items)
    for f in source.iterdir():
        if f.suffix != ".safetensors" and f.is_file() and f.name not in NOT_COPIED:
            shutil.copy(f, out / f.name)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = source / NOT_COPIED[1]
    meta = json.loads(index.read_text()).get("metadata", {}) if index.exists() else {}
    (out / NOT_COPIED[1]).write_text(json.dumps({"metadata": meta, "weight_map": weight_map}, indent=1))
    (out / "PRUNE.json").write_text(json.dumps({"source": str(source), "keep": k, "draft": draft, "experts": experts,
                                                "kept_ids": {str(layer): ids for layer, ids in keep.items()}}))
    return {"kept": len(weight_map), "dropped": dropped, "bytes": need}


def check(source, out) -> int:
    """Compare each tensor of the pruned folder with its source. Return the number of tensors compared."""
    source, out = Path(source), Path(out)
    info = json.loads((out / "PRUNE.json").read_text())
    names = {"mtp": "mtp.0"}   # the name of the draft layer in a folder of the Qwen prune by hand
    keep = {(int(layer) if layer.isdigit() else names.get(layer, layer)): ids
            for layer, ids in info["kept_ids"].items()}
    n, total = counts(source), 0
    if set_count(json.loads((out / "config.json").read_text()), info["keep"], 0) == 0:
        raise PruneError(f"{out}: config.json does not have the expert count {info['keep']}")
    for f in sorted(source.glob("*.safetensors")):
        head, start = header(out / f.name)
        items = list(tensors_of(f, keep, n))
        if len(head) - ("__metadata__" in head) != len(items):
            raise PruneError(f"{f.name}: the tensor count differs")
        with open(out / f.name, "rb") as new, open(f, "rb") as old:
            for name, dtype, shape, ranges in items:
                h, size = head.get(name), sum(r[1] for r in ranges)
                span = h and h["data_offsets"][1] - h["data_offsets"][0]
                if h is None or (h["dtype"], h["shape"], span) != (dtype, shape, size):
                    raise PruneError(f"{f.name}: {name} is missing or has another shape")
                new.seek(start + h["data_offsets"][0])
                if any(new.read(len(piece)) != piece for piece in pieces(old, ranges)):
                    raise PruneError(f"{f.name}: {name} differs from its source")
        total += len(items)
    return total
