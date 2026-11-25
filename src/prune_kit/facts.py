"""The facts of a checkpoint that the preflight needs (plans/PLAN_PRUNE_KIT.md §5.4), from its files only.

A routed expert is found by the name of its tensors, in one of two layouts:

- per expert: ``...layers.<L>.<...>.experts.<E>.<rest>`` (Qwen3.5/3.6 MoE, DeepSeek V4 Flash EXL3);
- stacked: ``...layers.<L>.<...>.experts.<name>``, where the first dimension is the expert (Inkling 2).

The tensors of the draft model (see ``Tensor.draft``) and of a shared expert are not routed experts.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .weights import Tensor, tensors

PER_EXPERT = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(?:[A-Za-z_]\w*\.)*experts\.(\d+)\.")
STACKED = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(?:[A-Za-z_]\w*\.)*experts\.[A-Za-z_]\w*$")
ROUTER_BIAS = re.compile(r"(?:^|\.)layers\.\d+\.(?:[A-Za-z_]\w*\.)*gate\.bias$")
DRAFT_EXPERT = re.compile(r"\.experts\.(\d+)\.")
PER_TOKEN = ("num_experts_per_tok", "num_experts_per_token")


@dataclass
class Draft:
    bytes: int
    experts: int | None     # None: the draft has no routed experts
    own_file: bool          # the draft is in its own weight file


@dataclass
class Facts:
    folder: Path
    model_type: str
    total_bytes: int                                        # each tensor of each weight file, the draft included
    layout: str                                             # "per_expert", "stacked" or "none"
    experts: dict[int, int] = field(default_factory=dict)       # layer -> routed experts
    expert_bytes: dict[int, int] = field(default_factory=dict)  # layer -> bytes of its routed experts
    per_token: int | None = None
    router_bias: bool = False
    draft: Draft | None = None
    loop: tuple[int, int, int] | None = None                # first core layer, last core layer + 1, passes
    quant: str | None = None

    @property
    def is_moe(self) -> bool:
        return bool(self.experts)

    @property
    def routed_bytes(self) -> int:
        return sum(self.expert_bytes.values())

    @property
    def max_experts(self) -> int:
        return max(self.experts.values(), default=0)

    def freed(self, k: int) -> int:
        """The bytes that a cut to at most ``k`` experts per layer removes."""
        return round(sum((n - min(n, k)) * self.expert_bytes[layer] / n for layer, n in self.experts.items()))


def _config(folder: Path) -> dict:
    path = folder / "config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _find(config: dict, keys) -> object:
    """The first of ``keys`` at the top level of the config or in one of its sub-configs."""
    for part in (config, *(v for v in config.values() if isinstance(v, dict))):
        for key in keys:
            if key in part:
                return part[key]
    return None


def _routed(items: list[Tensor]):
    """``(layout, experts per layer, bytes per layer)`` of the routed experts of the target model."""
    ids, size, stacked = defaultdict(set), Counter(), defaultdict(list)
    for t in items:
        if m := PER_EXPERT.match(t.name):
            ids[int(m[1])].add(int(m[2]))
            size[int(m[1])] += t.size
        elif (m := STACKED.match(t.name)) and t.shape:
            stacked[int(m[1])].append(t)
    if ids:
        return "per_expert", {layer: len(e) for layer, e in sorted(ids.items())}, dict(sorted(size.items()))
    experts, size = {}, {}
    for layer, group in sorted(stacked.items()):
        experts[layer] = max(group, key=lambda t: t.size).shape[0]   # the largest tensor holds the weights
        size[layer] = sum(t.size for t in group if t.shape[0] == experts[layer])
    return ("stacked" if experts else "none"), experts, size


def _draft(items: list[Tensor], target_files: set[str]) -> Draft | None:
    if not items:
        return None
    ids = {int(m[1]) for t in items if (m := DRAFT_EXPERT.search(t.name))}
    stacked = [t for t in items if ".experts." in t.name and not DRAFT_EXPERT.search(t.name) and t.shape]
    experts = len(ids) or (max(stacked, key=lambda t: t.size).shape[0] if stacked else None)
    return Draft(sum(t.size for t in items), experts, not {t.file for t in items} & target_files)


def inspect(folder: Path) -> Facts:
    """The facts of the checkpoint in ``folder``. It reads ``config.json`` and the headers of the weight files."""
    folder = Path(folder)
    config, items = _config(folder), tensors(folder)
    target = [t for t in items if not t.draft]
    layout, experts, expert_bytes = _routed(target)
    loop = _find(config, ("loop_config",)) or {}
    quant = config.get("quantization_config") or {}
    return Facts(
        folder=folder,
        model_type=config.get("model_type", "unknown"),
        total_bytes=sum(t.size for t in items),
        layout=layout,
        experts=experts,
        expert_bytes=expert_bytes,
        per_token=_find(config, PER_TOKEN),
        router_bias=any(ROUTER_BIAS.search(t.name) for t in target),
        draft=_draft([t for t in items if t.draft], {t.file for t in target}),
        loop=(loop["core_start"], loop["core_end"], loop["num_passes"]) if loop.get("num_passes", 1) > 1 else None,
        quant=quant.get("quant_algo") or quant.get("quant_method"),
    )
