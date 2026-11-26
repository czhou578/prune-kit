"""The probe, level 1 (plans/PLAN_PRUNE_KIT.md §5.5): two or three measured points in one window.

``report`` reads the files of a probe that ran: the cost of each cut, a band for that cost, and an estimate of the
deepest cut that holds the ``nll`` gate.

``plan`` writes the window file of a probe. ``prune-kit window`` runs that file, and it is the only command that
stops a worker. The window has one point for the full model and one point for each cut:

1. The full model boots with its capture flag. The clients capture a sample of the calibration split, record some
   held-out turns, score them (the reference), rank the sample, and build the pruned folder of each cut.
2. Each cut boots, and one client scores the same turns.
"""

import json
import os
import random
import shlex
import sys
from pathlib import Path

import yaml

from . import gates, rank, replay
from .facts import Facts
from .preflight import default_ks
from .resolve import slug
from .sessions import load

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


def folder(facts: Facts, workflow, name: str = "probe") -> Path:
    """The run folder: ``probe`` for a probe, ``full`` for the full gate of ``prune-kit probe --full``."""
    return Path("runs") / slug(facts.folder) / workflow.name / name


def root(folder: Path) -> Path:
    """The folder that holds a model folder and each file that its links name. A snapshot of the Hugging Face cache
    links into the blob folder of its model, so a container needs the folder of the model."""
    real = {str(p.resolve().parent) for p in folder.rglob("*") if p.is_symlink()}
    return Path(os.path.commonpath([str(folder.resolve()), *real]))


def plan(facts: Facts, workflow, profile: dict, machine: dict, port: int = 8890, tokens: int = 400_000,
         turns: int = 20, ranked=None, more=(), more_flags: str = "", more_env=None, name: str = "probe") -> dict:
    """The window file of the probe, as data. ``ranked``: a rank file of a full capture. The window then has no
    sample capture and builds each cut from that file. ``more``: the gates that each point also runs, of ``tasks``
    (the tool-call suite of model-benchmarks, from ``tools.model_benchmarks`` of the machine file), ``speed``, and
    ``judge`` (each cut also records its own turns, for ``prune-kit judge`` after the window). ``more_flags`` and
    ``more_env``: more flags and variables for each test server, for example those of a draft model.

    A profile whose ``serve`` has ``{mounts}`` starts its server in a container: the model paths are absolute, and
    the container gets the folder of the model and the run folder at the same paths. ``down`` of a profile is the
    command that removes the server after its stop. ``replay`` of a profile holds the settings of the replay gate:
    the gate commands then name the profile.

    ``tokens=None``: the capture is the whole calibration split. ``name``: the run folder (``folder``). Each point
    names the results file of its run folder, so two plans can be one window."""
    run, url, flow = folder(facts, workflow, name), f"http://127.0.0.1:{port}", shlex.quote(str(workflow.path))
    kit = f"{shlex.quote(sys.executable)} -m prune_moe_kit.cli"
    env = [f"{k}={shlex.quote(str(v))}" for k, v in {**profile.get("env", {}), **(more_env or {})}.items()]
    if machine.get("path"):
        env.append("PATH=" + ":".join([*machine["path"], "$PATH"]))
    inside = "{mounts}" in profile["serve"]
    mounts = " ".join(f"-v {shlex.quote(f'{d}:{d}:ro')}" for d in (root(facts.folder), run.resolve())) if inside else ""
    own = f" --profile {facts.model_type}" if profile.get("replay") else ""

    def extra(name: str) -> list:
        out, bench = [], machine.get("tools", {}).get("model_benchmarks")
        if "tasks" in more:
            if not bench:
                raise ProbeError("the gate tasks needs tools.model_benchmarks (dir, python) in the machine file")
            out.append(f"cd {shlex.quote(str(Path(bench['dir']).expanduser()))} && {bench.get('python', 'python3')} "
                       f"-m benchmarks.tool_calling --base-url {url}/v1 --model {NAME} "
                       f"--output {shlex.quote(str((run / f'tools-{name}.json').resolve()))}")
        if "judge" in more and name != "full":   # the answers of the point to the same targets, with the same seeds
            out.append(f"{kit} gate --record --workflow {flow} --base-url {url} --served {NAME}{own} "
                       f"--n {workflow.gates['judge']['turns']} --out {run}/recorded-{name}.jsonl")
        if "speed" in more:
            out.append(f"{kit} gate --speed --base-url {url} --served {NAME} --out {run}/speed-{name}.json")
        return out

    def point(name: str, model, freed: int, flags: str, clients: list) -> dict:
        model = Path(model).resolve() if inside else model
        serve = profile["serve"].format(model=shlex.quote(str(model)), port=port, name=NAME, mounts=mounts)
        down = {"down": profile["down"].format(name=NAME)} if profile.get("down") else {}
        need = (facts.total_bytes - freed) / GIB + profile.get("overhead_gib", 0)
        boot = " ".join(filter(None, ["env", *env, serve, flags, more_flags]))
        return {"name": name, "need_gib": round(need, 1), "health": f"{url}/health", "clients": clients + extra(name),
                "boot": f"{boot} > {run}/server-{name}.log 2>&1", **down, "results": f"{run}/window.json"}

    ks = cuts(workflow, facts)
    score = lambda name: (f"{kit} gate --score --workflow {flow} --base-url {url} --served {NAME}{own} "
                          f"--recorded {run}/recorded.jsonl --out {run}/score-{name}.jsonl")
    sample = [] if ranked else [f"{kit} capture {shlex.quote(str(facts.folder))} --workflow {flow} --base-url {url} "
                                f"--served {NAME}{f' --tokens {tokens}' if tokens else ''} --out {run}/capture.jsonl"]
    full = sample + [f"{kit} gate --record --workflow {flow} --base-url {url} --served {NAME}{own} --n {turns} "
                     f"--out {run}/recorded.jsonl", score("full")]
    if not ranked:
        full.append(f"{kit} rank {run}/capture.jsonl --out {run}/rank.json --k {' '.join(map(str, ks))}")
    source = shlex.quote(str(ranked)) if ranked else f"{run}/rank.json"
    full += [f"{kit} build {shlex.quote(str(facts.folder))} --rank {source} --k {k} "
             f"--draft {profile.get('draft', 'rank')} --out {run}/k{k}" for k in ks]
    points = [point("full", facts.folder, 0, "" if ranked else profile.get("capture_flag", ""), full)]
    points += [point(f"k{k}", run / f"k{k}", facts.freed(k), "", [score(f"k{k}")]) for k in ks]
    # The window reads the full model and writes one folder for each cut: the runner drops their cached pages.
    return {"release": [str(facts.folder), str(run)], "points": points}


def write(facts: Facts, workflow, window: dict, names=("probe",)) -> Path:
    """Write the window file into the first run folder. Refuse a run folder that holds a result of an earlier run."""
    for name in names:
        run = folder(facts, workflow, name)
        old = [p.name for p in run.glob("*") if p.name != "window.yml"] if run.exists() else []
        if old:
            raise ProbeError(f"{run} holds the files of an earlier run ({', '.join(sorted(old)[:4])}): move the folder")
        run.mkdir(parents=True, exist_ok=True)
    path = folder(facts, workflow, names[0]) / "window.yml"
    path.write_text("# The window file of a probe, from `prune-kit probe`. Run it with `prune-kit window`.\n"
                    + yaml.safe_dump(window, sort_keys=False, width=10_000))
    return path


def band(ref_rows, new_rows, draws: int = 1000) -> tuple[float, float]:
    """The 5% and the 95% point of the NLL change in percent, over new samples of the common turns. 20 turns give
    much noise: the band shows how far the cost of a cut can be from its measured value."""
    ref, new = ({replay.key(r): r for r in rows if "error" not in r and r["tokens"]} for rows in (ref_rows, new_rows))
    pairs = [(ref[k], new[k]) for k in sorted(ref.keys() & new.keys())]
    rng, changes = random.Random(0), []
    per_token = lambda rows: sum(r["nll"] for r in rows) / sum(r["tokens"] for r in rows)
    for _ in range(draws):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        changes.append(100 * (per_token([b for _, b in sample]) / per_token([a for a, _ in sample]) - 1))
    changes.sort()
    return changes[draws // 20], changes[-(draws // 20) - 1]


def report(facts: Facts, workflow, run: Path | None = None, ranked=None) -> dict:
    """The result of a probe as plain data. ``text`` prints it."""
    run = Path(run or folder(facts, workflow))
    try:
        ref = load(run / "score-full.jsonl")
        captured = load(run / "capture.jsonl") if (run / "capture.jsonl").exists() else []   # none: a given rank
        ranked = json.loads(Path(ranked or run / "rank.json").read_text())
    except OSError as e:
        raise ProbeError(f"the probe did not run to its reference: {e.filename} does not exist") from e
    limit = workflow.gates["nll"]["limit_pct"]
    ks = sorted((int(p.stem[7:]) for p in run.glob("score-k*.jsonl")), reverse=True)
    shares, points = rank.overlap(captured, ks) if len(captured) > 1 else {}, []
    for k in ks:
        new = load(run / f"score-k{k}.jsonl")
        result, (low, high) = replay.compare(ref, new, workflow.gates), band(ref, new)
        state = "pass" if high <= limit else "fail" if low > limit else "near the limit"
        points.append({"k": k, "freed_bytes": facts.freed(k), "removed_pct": 100 * rank.removed(ranked, k),
                       "overlap": shares.get(k), "turns": result["new"]["turns"], "band": [low, high],
                       "nll_change_pct": result["nll_change_pct"], "nll_gate": state,
                       "tools_gates": all(list(result["gates"].values())[1:]),
                       **gates.point(run, f"k{k}", workflow)})
        points[-1]["verdict"] = gates.verdict(state, points[-1]["tools_gates"], points[-1])
    # The line goes through the full model (no cut, no cost) and fits the measured points.
    square = sum(p["removed_pct"] ** 2 for p in points)
    slope = sum(p["removed_pct"] * p["nll_change_pct"] for p in points) / square if square else None
    estimate = None
    if slope and slope > 0:
        estimate = next((k for k in range(facts.per_token or 1, facts.max_experts)
                         if 100 * rank.removed(ranked, k) * slope <= limit), None)
    memory = json.loads(w.read_text())["points"] if (w := run / "window.json").exists() else []
    recorded = load(run / "recorded.jsonl") if (run / "recorded.jsonl").exists() else []
    peak = next((m["available_before_gib"] - m["available_lowest_gib"] for m in memory if m["name"] == "full"), None)
    return {"folder": str(run), "segments": ranked["segments"], "prompt_tokens": ranked["prompt_tokens"],
            "limit_pct": limit, "points": points, "slope": slope, "estimate_k": estimate, "recorded": len(recorded),
            "at_limit": sum(r.get("finish_reason") == "length" for r in recorded),
            "estimate_freed_bytes": facts.freed(estimate) if estimate else None, "full_peak_gib": peak,
            "weights_gib": facts.total_bytes / GIB}


def agree(probe_run: dict, full_run: dict) -> str:
    """The full gate against the band of the probe, for each cut of both reports (gate D of the plan)."""
    full = {p["k"]: p["nll_change_pct"] for p in full_run["points"]}
    lines = ["the probe against the full gate"]
    for p in probe_run["points"]:
        if p["k"] in full:
            (low, high), cost = p["band"], full[p["k"]]
            lines.append(f"K {p['k']:<6}probe {p['nll_change_pct']:+.2f}% (band {low:+.2f}% to {high:+.2f}%), "
                         f"full gate {cost:+.2f}%: {'inside the band' if low <= cost <= high else 'not in the band'}")
    return "\n".join(lines)


def text(r: dict) -> str:
    lines = [f"probe     {r['folder']}",
             f"sample    {r['segments']} segments, {r['prompt_tokens']:,} prompt tokens",
             *([f"turns     {r['recorded']} recorded, {r['at_limit']} stopped at the token limit: such a turn has no "
                "decision to score"] if r["at_limit"] else []), "",
             f"{'point':<8}{'freed':>10}{'routing removed':>18}{'overlap':>9}{'NLL change':>12}{'band (90%)':>20}  "
             f"gate nll (+{r['limit_pct']:g}%)"]
    for p in r["points"]:
        both = f"{p['overlap']:.0%}" if p["overlap"] is not None else "-"
        lines.append(f"K {p['k']:<6}{p['freed_bytes'] / GIB:>6.2f} GiB{p['removed_pct']:>17.2f}%{both:>9}"
                     f"{p['nll_change_pct']:>+11.2f}%{p['band'][0]:>+10.2f}% to {p['band'][1]:+.2f}%  {p['nll_gate']}"
                     + ("" if p["tools_gates"] else "; a tools gate fails"))
    lines.append("")
    for p in r["points"]:
        parts = [f"tasks {t['new']} of {t['of']} (the full model: {t['reference']})" if (t := p["tasks"]) else "",
                 f"peak {m['peak_gib']:.1f} GiB" + (f" (goal {m['goal_gib']:g})" if m["goal_gib"] else "")
                 if (m := p["memory"]) else "",
                 f"speed {v['tokens_per_second']} tok/s (the full model: {v['reference']}"
                 + (f", {c:+g}% for the same prompts)" if (c := v.get("change_pct")) is not None else ")")
                 if (v := p["speed"]) else "",
                 f"judge {'no loss' if j['ok'] else 'a loss'} ({j['summary']})" if (j := p["judge"]) else ""]
        lines.append(f"K {p['k']:<6}verdict: {p['verdict']}" + "".join(f"; {x}" for x in parts if x))
    if not r["points"]:
        lines.append("no cut has a score: the window ended before the first cut")
    elif r["estimate_k"]:
        lines += ["", f"estimate  the nll gate holds down to about K {r['estimate_k']} "
                      f"({r['estimate_freed_bytes'] / GIB:.2f} GiB freed)",
                  f"          a line through the points: +{r['slope']:.2f}% NLL for each 1% of removed routing",
                  "          the estimate is not a gate result: a cut near the limit needs the full gate with 60 turns"]
    else:
        lines += ["", "estimate  none: the cuts show no cost that grows with the removed routing"]
    if r["full_peak_gib"] is not None:
        lines.append(f"memory    the full model used {r['full_peak_gib']:.1f} GiB at the peak of the window "
                     f"({r['weights_gib']:.2f} GiB of weights in the folder)")
    return "\n".join(lines)
