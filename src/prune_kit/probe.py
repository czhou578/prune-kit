"""The probe, level 1 (plans/PLAN_PRUNE_KIT.md §5.5): two or three measured points in one window.

``plan`` writes the window file of a probe. ``prune-kit window`` runs that file, and it is the only command that
stops a worker. The window has one point for the full model and one point for each cut:

1. The full model boots with its capture flag. The clients capture a sample of the calibration split, record some
   held-out turns, score them (the reference), rank the sample, and build the pruned folder of each cut.
2. Each cut boots, and one client scores the same turns.
"""

import shlex
import sys
from pathlib import Path

import yaml

from .facts import Facts
from .preflight import default_ks
from .resolve import slug

GIB = 1 << 30
PROFILES = Path(__file__).parent / "profiles"
NAME = "probe"   # the model name of each test server


class ProbeError(Exception):
    pass


def load_profile(model_type: str) -> dict:
    """The adapter profile of a model type: the serve command of its test server, and its capture flag."""
    path = PROFILES / f"{model_type}.yml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in PROFILES.glob("*.yml")))
        raise ProbeError(f"no profile for the model type {model_type}: prune-kit has profiles for {known}")
    return yaml.safe_load(path.read_text())


def cuts(workflow, facts: Facts) -> list[int]:
    """The K values of the probe: ``points`` of the workflow file, or the first three default cuts."""
    ks = default_ks(facts.max_experts)[:3] if workflow.points == "auto" else sorted(workflow.points, reverse=True)
    if not ks or not all((facts.per_token or 1) <= k < facts.max_experts for k in ks):
        raise ProbeError(f"each cut must be below the {facts.max_experts} experts of a layer: {ks}")
    return ks


def folder(facts: Facts, workflow) -> Path:
    return Path("runs") / slug(facts.folder) / workflow.name / "probe"


def plan(facts: Facts, workflow, profile: dict, machine: dict, port: int = 8890, tokens: int = 400_000,
         turns: int = 20) -> dict:
    """The window file of the probe, as data."""
    run, url, flow = folder(facts, workflow), f"http://127.0.0.1:{port}", shlex.quote(str(workflow.path))
    kit = f"{shlex.quote(sys.executable)} -m prune_kit.cli"
    env = [f"{k}={shlex.quote(str(v))}" for k, v in profile.get("env", {}).items()]
    if machine.get("path"):
        env.append("PATH=" + ":".join([*machine["path"], "$PATH"]))

    def point(name: str, model, freed: int, flags: str, clients: list) -> dict:
        serve = profile["serve"].format(model=shlex.quote(str(model)), port=port, name=NAME)
        need = (facts.total_bytes - freed) / GIB + profile.get("overhead_gib", 0)
        return {"name": name, "need_gib": round(need, 1), "health": f"{url}/health", "clients": clients,
                "boot": " ".join(["env", *env, serve, flags]).strip() + f" > {run}/server-{name}.log 2>&1"}

    ks = cuts(workflow, facts)
    score = lambda name: (f"{kit} gate --score --workflow {flow} --base-url {url} --served {NAME} "
                          f"--recorded {run}/recorded.jsonl --out {run}/score-{name}.jsonl")
    full = [f"{kit} capture {shlex.quote(str(facts.folder))} --workflow {flow} --base-url {url} --served {NAME} "
            f"--tokens {tokens} --out {run}/capture.jsonl",
            f"{kit} gate --record --workflow {flow} --base-url {url} --served {NAME} --n {turns} "
            f"--out {run}/recorded.jsonl",
            score("full"),
            f"{kit} rank {run}/capture.jsonl --out {run}/rank.json --k {' '.join(map(str, ks))}"]
    full += [f"{kit} build {shlex.quote(str(facts.folder))} --rank {run}/rank.json --k {k} "
             f"--draft {profile.get('draft', 'rank')} --out {run}/k{k}" for k in ks]
    points = [point("full", facts.folder, 0, profile.get("capture_flag", ""), full)]
    points += [point(f"k{k}", run / f"k{k}", facts.freed(k), "", [score(f"k{k}")]) for k in ks]
    return {"points": points}


def write(facts: Facts, workflow, window: dict) -> Path:
    """Write the window file into the probe folder. Refuse a folder that holds a result of an earlier probe."""
    run = folder(facts, workflow)
    old = [p.name for p in run.glob("*") if p.name != "window.yml"] if run.exists() else []
    if old:
        raise ProbeError(f"{run} holds the files of an earlier probe ({', '.join(sorted(old)[:4])}): move the folder")
    run.mkdir(parents=True, exist_ok=True)
    path = run / "window.yml"
    path.write_text("# The window file of a probe, from `prune-kit probe`. Run it with `prune-kit window`.\n"
                    + yaml.safe_dump(window, sort_keys=False, width=10_000))
    return path
