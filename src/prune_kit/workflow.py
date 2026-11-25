"""The workflow file (plans/PLAN_PRUNE_KIT.md §5.2): one YAML file that describes one workflow.

``load`` reads the file, adds the defaults, and checks each value. It reports each fault of the file in one error,
with the path of the key.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCES = {   # the kinds of data source, with the keys that each kind accepts
    "sessions": {"path"},                       # a sessions.jsonl of an earlier run: it keeps its own split
    "claude_code": {"root", "projects"},        # the Claude Code transcripts
    "gateway_records": {"root", "route"},       # the records of model-gateway
    "hermes": {"root"},                         # the sessions of Hermes
    "repo": {"paths", "files"},                 # code files of repositories
}
NUMBER, COUNT = (int, float), int
# Each key with its type and its default. None as a default: the key has no default and can be absent.
SCHEMA = {
    "name": (str, ...),
    "data": {
        "sources": (list, ...),
        "folder": (str, None),
        "system_prompt": (str, None),
        "tools": (str, None),
        "held_out": {"share": (NUMBER, 0.15), "projects": (list, [])},
        "min_calibration_tokens": (COUNT, 1_000_000),
        "min_held_out_turns": (COUNT, 60),
    },
    "memory": {"peak_gib": (NUMBER, None), "context": (COUNT, None), "margin_gib": (NUMBER, 4),
               "overhead_gib": (NUMBER, None)},
    "gates": {   # the limits of the two prunes (§5.6)
        "nll": {"limit_pct": (NUMBER, 3.0), "turns": (COUNT, 60)},
        "tools": {"names_min_pct": (NUMBER, 98), "decisions_max_drop": (COUNT, 3)},
        "tasks": {"sets": (list, []), "seeds": (COUNT, 3), "max_drop": (COUNT, 3)},
        "judge": {"turns": (COUNT, 30), "model": (str, None)},
    },
    "points": ((str, list), "auto"),
}


class WorkflowError(Exception):
    """The file does not exist, is not YAML, or has faults. The message lists each fault."""


@dataclass(frozen=True)
class Workflow:
    name: str
    path: Path
    data: dict
    memory: dict
    gates: dict
    points: str | list

    @property
    def folder(self) -> Path:
        """The folder of the sessions. The default is ``runs/data/<name>`` below the current folder."""
        return Path(self.data["folder"] or Path("runs/data") / self.name).expanduser()

    def need_bytes(self, total_bytes: int) -> int | None:
        """The bytes that a prune must free to reach the peak goal, or None with no goal. Without
        ``memory.overhead_gib`` (the KV pin and the overhead of the server), the result is a lower limit."""
        if self.memory["peak_gib"] is None:
            return None
        return round(total_bytes + ((self.memory["overhead_gib"] or 0) - self.memory["peak_gib"]) * (1 << 30))


def _fill(spec: dict, value, path: str, faults: list) -> dict:
    """``value`` with the defaults of ``spec``. Each fault goes to ``faults``."""
    if not isinstance(value, dict):
        faults.append(f"{path or 'the file'}: must be a mapping")
        value = {}
    faults += [f"{path}{key}: not a known key" for key in value if key not in spec]
    out = {}
    for key, rule in spec.items():
        if isinstance(rule, dict):
            out[key] = _fill(rule, value.get(key, {}), f"{path}{key}.", faults)
            continue
        kind, default = rule
        got = value.get(key, default)
        if got is ...:
            faults.append(f"{path}{key}: is missing")
        elif got is not None and (not isinstance(got, kind) or isinstance(got, bool)):
            names = " or ".join(t.__name__ for t in (kind if isinstance(kind, tuple) else (kind,)))
            faults.append(f"{path}{key}: must be of type {names}")
        elif isinstance(got, NUMBER) and got < 0:
            faults.append(f"{path}{key}: must not be negative")
        out[key] = got
    return out


def _check(w: dict, faults: list) -> None:
    """The rules that the types do not cover."""
    for n, source in enumerate(w["data"]["sources"] if isinstance(w["data"]["sources"], list) else []):
        kind = source.get("kind") if isinstance(source, dict) else None
        if kind not in SOURCES:
            faults.append(f"data.sources[{n}].kind: must be one of {', '.join(sorted(SOURCES))}")
            continue
        faults += [f"data.sources[{n}].{key}: not a key of the kind {kind}"
                   for key in source if key != "kind" and key not in SOURCES[kind]]
    share = w["data"]["held_out"]["share"]
    if isinstance(share, NUMBER) and not 0 <= share < 1:
        faults.append("data.held_out.share: must be at least 0 and below 1")
    points = w["points"]
    if points != "auto" and not (isinstance(points, list) and points and all(type(k) is int and k > 0 for k in points)):
        faults.append('points: must be "auto" or a list of K values')
    if isinstance(w["name"], str) and not w["name"].replace("-", "").replace("_", "").isalnum():
        faults.append("name: use letters, digits, - and _ only: the name is a folder name")


def load(path) -> Workflow:
    path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as e:
        raise WorkflowError(f"cannot read the workflow file {path}: {e.strerror}") from e
    except yaml.YAMLError as e:
        raise WorkflowError(f"{path} is not YAML: {e}") from e
    faults: list[str] = []
    w = _fill(SCHEMA, raw, "", faults)
    _check(w, faults)
    if faults:
        raise WorkflowError(f"{path} has {len(faults)} faults:\n" + "\n".join(f"  {fault}" for fault in faults))
    return Workflow(w["name"], path, w["data"], w["memory"], w["gates"], w["points"])
