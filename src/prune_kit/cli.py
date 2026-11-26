"""Command line: prune-kit stages | preflight | data | capture | rank | build | gate | window | probe | judge."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import (__version__, capture, gates, judge, preflight, probe, rank, replay, sessions, sources, st_moe,
               stages, window, workflow)
from .facts import inspect
from .resolve import ModelNotFound, resolve, slug
from .weights import FormatError, WeightsError


def cmd_stages(_args) -> int:
    print(stages.table())
    return 0


def cmd_preflight(a) -> int:
    """Exit code 0: the pipeline can run, possibly with work. 1: NO-GO. 2: the model cannot be read."""
    try:
        flow = workflow.load(a.workflow) if a.workflow else None
        facts = inspect(resolve(a.model, a.hub))
    except FormatError as e:
        print(f"{e}\nverdict   NO-GO")
        return 1
    except (ModelNotFound, WeightsError, workflow.WorkflowError, OSError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    r = preflight.report(facts, a.k, preflight.available_bytes(), flow)
    print(json.dumps(r, indent=2) if a.json else preflight.text(r))
    return 1 if r["verdict"] == "NO-GO" else 0


def cmd_data(a) -> int:
    """Exit code 0: the data meets the minimums of the workflow. 1: it does not. 2: the file has a fault."""
    try:
        flow = workflow.load(a.workflow)
        data = sessions.load(flow.folder / "sessions.jsonl") if a.stats else sources.build(flow)
    except (workflow.WorkflowError, OSError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    d = flow.data
    r = sessions.check(data, d["min_calibration_tokens"], d["min_held_out_turns"])
    print("\n".join([f"data      {flow.folder / 'sessions.jsonl'}", *sessions.table(r),
                     f"calibration  {r['calibration_tokens']:,} tokens (minimum {d['min_calibration_tokens']:,})",
                     f"held out     {r['held_out_turns']} target turns (minimum {d['min_held_out_turns']})",
                     "verdict   " + ("the data meets the minimums" if r["ok"] else "the data is below a minimum")]))
    return 0 if r["ok"] else 1


def _tools(folder: Path):
    """The tool definitions of a data folder: a list, or a mapping with the key ``tools``. None with no file."""
    tools = json.loads(t.read_text()) if (t := folder / "tools.json").exists() else None
    return tools.get("tools") if isinstance(tools, dict) else tools


def cmd_capture(a) -> int:
    """Exit code 0: each segment has its counts. 1: a segment failed. 2: a fault before the first request."""
    try:
        flow, folder = workflow.load(a.workflow), resolve(a.model, a.hub)
        facts = inspect(folder)
        data = sessions.load(flow.folder / "sessions.jsonl")
        tools = _tools(flow.folder)
        served = a.served or capture.post(a.base_url, "/v1/models", None)["data"][0]["id"]
    except (ModelNotFound, WeightsError, workflow.WorkflowError, OSError, ValueError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    out = Path(a.out or Path("runs") / slug(folder) / flow.name / "capture.jsonl")
    failed = capture.run(a.base_url, served, data, tools, out, facts.max_experts, a.limit, json.loads(a.body),
                         a.generate, max_tokens=a.tokens, loop=facts.loop)
    print(f"capture   {out}" + (f": {failed} segments failed; start the command again" if failed else ""))
    return 1 if failed else 0


def cmd_rank(a) -> int:
    """Exit code 0: the rank file is written. 2: the capture cannot give a rank."""
    try:
        with open(a.capture) as f:
            rows = [json.loads(line) for line in f]
        ranked = rank.rank(rows)
    except (rank.RankError, OSError, ValueError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    out = Path(a.out or Path(a.capture).with_name("rank.json"))
    out.write_text(json.dumps(ranked))
    ks = a.k or preflight.default_ks(len(ranked["layers"]["0"]))
    print(rank.text(ranked, ks, rank.overlap(rows, ks) if ranked["segments"] > 1 else None) + f"\nrank      {out}")
    return 0


def cmd_build(a) -> int:
    """Exit code 0: the pruned folder is equal to its source in each kept tensor. 2: a fault."""
    try:
        source = resolve(a.model, a.hub)
        if not a.check:
            if a.rank is None or a.k is None:
                raise st_moe.PruneError("a build needs --rank and --k")
            r = st_moe.prune(source, a.out, json.loads(Path(a.rank).read_text()), a.k, a.draft)
            print(f"kept {r['kept']} tensors, dropped {r['dropped']}, wrote {r['bytes'] / 2**30:.2f} GiB to {a.out}")
        print(f"check: {st_moe.check(source, a.out)} tensors are equal to their source")
    except (ModelNotFound, st_moe.PruneError, OSError, ValueError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_gate(a) -> int:
    """Exit code 0: done, and with --compare each gate passes. 1: a gate fails or a turn failed. 2: a fault."""
    try:
        flow = workflow.load(a.workflow) if a.workflow else None
        if a.compare:
            result = replay.compare(*map(sessions.load, a.compare), flow.gates if flow else replay.LIMITS)
            print(replay.text(result))
            return 0 if all(result["gates"].values()) else 1
        if a.speed:
            if not (a.out and a.base_url):
                raise ValueError("--speed needs --out and --base-url")
            served = a.served or capture.post(a.base_url, "/v1/models", None)["data"][0]["id"]
            print(f"speed     {gates.measure_speed(a.base_url, served, a.out)['tokens_per_second']} tokens per second")
            return 0
        if not (a.out and (a.data or flow) and (a.base_url or a.from_data) and (a.record or a.recorded)):
            raise ValueError("--record and --score need --out, --base-url, and --data or --workflow; "
                             "--score also needs --recorded")
        folder = Path(a.data) if a.data else flow.folder
        data, tools = sessions.load(folder / "sessions.jsonl"), _tools(folder)
        served = a.served or (None if a.from_data else capture.post(a.base_url, "/v1/models", None)["data"][0]["id"])
        own = probe.load_profile(a.profile).get("replay", {}) if a.profile else {}
        if a.record:
            turns = a.n or (flow.gates["nll"]["turns"] if flow else 60)
            more = {k: own[k] for k in ("sampling", "max_tokens") if k in own}
            failed = replay.record(a.base_url, served, data, tools, turns, a.out, from_data=a.from_data, **more)
        else:
            more = {k: own[k] for k in ("end_think", "start_think", "decision_tokens") if k in own}
            failed = replay.score(a.base_url, served, data, tools, sessions.load(a.recorded), a.out, **more)
    except (workflow.WorkflowError, probe.ProbeError, OSError, ValueError, KeyError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    done = "recorded" if a.record else "scored"
    print(f"{done}  {a.out}" + (f": {failed} turns failed; start the command again" if failed else ""))
    return 1 if failed else 0


def cmd_window(a) -> int:
    """Exit code 0: each point ran. 1: a point failed or did not run. 2: a fault of a file. 3: the restore failed."""
    try:
        machine, points = window.load_machine(a.machine), window.load_window(a.window)
        release = window.load(a.window, {}).get("release", [])
        times = [datetime.fromisoformat(t) for t in (a.start_after, a.boot_by, a.restore_at)]
    except (window.WindowError, ValueError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    log = lambda line: print(f"{datetime.now():%F %T} {line}", flush=True)
    system = window.DryRun(log, machine, datetime.now()) if a.dry_run else None
    return window.Window(machine, a.stop, *times, system, log, None if a.dry_run else a.results, release).run(points)


def cmd_probe(a) -> int:
    """Exit code 0: the window file of the probe is written, or the report is printed. 2: a fault."""
    try:
        flow, facts = workflow.load(a.workflow), inspect(resolve(a.model, a.hub))
        if a.report:
            r = probe.report(facts, flow, a.run, a.rank)
            print(json.dumps(r, indent=2) if a.json else probe.text(r))
            return 0
        if not a.machine:
            raise probe.ProbeError("the window file of a probe needs --machine")
        plan = probe.plan(facts, flow, probe.load_profile(facts.model_type), window.load_machine(a.machine),
                          a.port, a.tokens, a.turns, a.rank, a.gates, a.flags, dict(e.split("=", 1) for e in a.env))
        path = probe.write(facts, flow, plan)
    except (ModelNotFound, WeightsError, workflow.WorkflowError, window.WindowError, probe.ProbeError, OSError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    names = ", ".join(f"{p['name']} ({p['need_gib']:g} GiB)" for p in plan["points"])
    run = f"prune-kit window {path} --machine {a.machine} --results {path.with_name('window.json')}"
    print(f"window    {path}\npoints    {names}\n\nThe probe stops no worker. A window does: name each worker "
          f"with --stop, and set the three deadlines.\nFirst the dry run, then the same command with no --dry-run:"
          f"\n  {run} --stop WORKER --start-after TIME --boot-by TIME --restore-at TIME --dry-run")
    return 0


def cmd_judge(a) -> int:
    """Exit code 0: the judge finds no loss. 1: the judge finds a loss. 2: a fault."""
    try:
        flow = workflow.load(a.workflow)
        folder = Path(a.data) if a.data else flow.folder
        served = a.served or flow.gates["judge"]["model"]
        served = served or capture.post(a.base_url, "/v1/models", None)["data"][0]["id"]
        result = judge.run(a.base_url, served, sessions.load(folder / "sessions.jsonl"), sessions.load(a.ref),
                           sessions.load(a.new), flow.gates["judge"]["turns"], a.out, a.page, json.loads(a.body))
    except (workflow.WorkflowError, OSError, ValueError, KeyError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    print(f"judge     {result['summary']}\nverdict   " + ("no loss that the test shows" if result["ok"] else "a loss"))
    return 0 if result["ok"] else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prune-kit", description="Prune a local model for one workflow.")
    p.add_argument("--version", action="version", version=f"prune-kit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stages", help="list the stages of a prune and the command of each stage")
    s.set_defaults(run=cmd_stages)
    f = sub.add_parser("preflight", help="what each cut frees, and whether the pipeline can run (files only)")
    f.add_argument("model", help="a folder, a Hugging Face name in a local cache, or a part of such a name")
    f.add_argument("--k", type=int, nargs="+", metavar="K",
                   help="the cuts of the table (default: steps down to 5/8 of the experts)")
    f.add_argument("--workflow", metavar="FILE", help="a workflow file: adds the cut that reaches its memory goal")
    f.add_argument("--hub", action="append", default=[], metavar="DIR", help="one more cache folder to search")
    f.add_argument("--json", action="store_true", help="print the report as JSON")
    f.set_defaults(run=cmd_preflight)
    d = sub.add_parser("data", help="build the sessions of a workflow, and compare them with its minimums")
    d.add_argument("workflow", metavar="FILE", help="a workflow file")
    d.add_argument("--stats", action="store_true", help="read the sessions that exist; build nothing")
    d.set_defaults(run=cmd_data)
    c = sub.add_parser("capture", help="count the routed experts of the calibration sessions on a capture server")
    c.add_argument("model", help="the model of the server: a folder, or a name in a local cache")
    c.add_argument("--workflow", required=True, metavar="FILE", help="the workflow file: its data folder is the input")
    c.add_argument("--base-url", required=True, metavar="URL", help="the server, for example http://127.0.0.1:8890")
    c.add_argument("--served", metavar="NAME", help="the model name of the server (default: its first model)")
    c.add_argument("--out", metavar="FILE", help="the counts (default: runs/<model>/<workflow>/capture.jsonl)")
    c.add_argument("--limit", type=int, metavar="N", help="capture the first N segments only")
    c.add_argument("--tokens", type=int, metavar="N", help="a sample: stop at N prompt tokens, in a fixed mixed order")
    c.add_argument("--generate", type=int, default=0, metavar="N", help="also count the routing of N new tokens")
    c.add_argument("--body", default="{}", metavar="JSON", help="more keys for each request, as a JSON mapping")
    c.add_argument("--hub", action="append", default=[], metavar="DIR", help="one more cache folder to search")
    c.set_defaults(run=cmd_capture)
    r = sub.add_parser("rank", help="order the experts of each layer by their use in a capture")
    r.add_argument("capture", metavar="CAPTURE", help="the counts of prune-kit capture")
    r.add_argument("--out", metavar="FILE", help="the rank file (default: rank.json beside the capture)")
    r.add_argument("--k", type=int, nargs="+", metavar="K", help="the cuts of the table")
    r.set_defaults(run=cmd_rank)
    b = sub.add_parser("build", help="write a pruned copy of a model, and compare it with its source")
    b.add_argument("model", help="the full model: a folder, or a name in a local cache")
    b.add_argument("--out", required=True, metavar="DIR", help="the pruned folder: it must not exist for a build")
    b.add_argument("--rank", metavar="FILE", help="the rank file: the experts of each layer, the most used first")
    b.add_argument("--k", type=int, metavar="K", help="the experts to keep in each layer")
    b.add_argument("--draft", choices=("rank", "first", "keep"), default="rank",
                   help="the experts of the draft model: by the rank, the first K, or all of them")
    b.add_argument("--check", action="store_true", help="only compare a pruned folder that exists with its source")
    b.add_argument("--hub", action="append", default=[], metavar="DIR", help="one more cache folder to search")
    b.set_defaults(run=cmd_build)
    g = sub.add_parser("gate", help="the replay gate: record turns of the full model, score them, compare two scores")
    what = g.add_mutually_exclusive_group(required=True)
    what.add_argument("--record", action="store_true", help="the server writes one turn for each held-out target")
    what.add_argument("--score", action="store_true", help="the server scores the recorded turns")
    what.add_argument("--speed", action="store_true", help="measure the decode speed of the server")
    what.add_argument("--compare", nargs=2, metavar=("REF", "NEW"), help="compare two score files with the limits")
    g.add_argument("--workflow", metavar="FILE", help="the workflow file: its data folder and its limits")
    g.add_argument("--data", metavar="DIR", help="a data folder with sessions.jsonl, in place of the workflow's")
    g.add_argument("--base-url", metavar="URL", help="the server, for example http://127.0.0.1:8889")
    g.add_argument("--served", metavar="NAME", help="the model name of the server (default: its first model)")
    g.add_argument("--recorded", metavar="FILE", help="--score: the recorded turns")
    g.add_argument("--out", metavar="FILE", help="the output of --record or --score; a second start continues it")
    g.add_argument("--n", type=int, metavar="N", help="--record: the number of turns (default: gates.nll.turns)")
    g.add_argument("--from-data", action="store_true", help="--record: the own turns of the data, with no server")
    g.add_argument("--profile", metavar="TYPE", help="the model type: its profile gives the replay settings")
    g.set_defaults(run=cmd_gate)
    w = sub.add_parser("window", help="run the points of a window file on test servers, then restore the workers")
    w.add_argument("window", metavar="FILE", help="the window file: the points with their servers and clients")
    w.add_argument("--machine", required=True, metavar="FILE", help="the machine file: the workers and the limits")
    w.add_argument("--stop", action="append", default=[], metavar="WORKER",
                   help="a live worker that the window stops and restores; no worker stops without this option")
    for name, text in (("start-after", "the window waits until this time"), ("boot-by", "no point boots after it"),
                       ("restore-at", "each client stops, and the restore starts")):
        w.add_argument(f"--{name}", required=True, metavar="TIME", help=f'{text}, as "2026-11-28 22:30"')
    w.add_argument("--dry-run", action="store_true", help="print each command; run no command and stop no worker")
    w.add_argument("--results", metavar="FILE", help="write the result and the memory of each point as JSON")
    w.set_defaults(run=cmd_window)
    o = sub.add_parser("probe", help="write the window file of a probe: a sample capture and two or three cuts")
    o.add_argument("model", help="the full model: a folder, or a name in a local cache")
    o.add_argument("--workflow", required=True, metavar="FILE", help="the workflow file: its data, and its points")
    o.add_argument("--machine", metavar="FILE", help="the machine file: the path of the server tools")
    o.add_argument("--rank", metavar="FILE", help="the rank of a full capture: no sample capture in the window")
    o.add_argument("--gates", nargs="+", default=[], choices=("tasks", "speed", "judge"),
                   help="each point also runs these: the tool-call suite, the decode speed, the turns for the judge")
    o.add_argument("--flags", default="", metavar="TEXT", help="more flags for each test server, as one text")
    o.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="one more variable for each server")
    o.add_argument("--report", action="store_true", help="print the result of the probe that ran; write no file")
    o.add_argument("--json", action="store_true", help="--report: print the result as JSON")
    o.add_argument("--run", metavar="DIR", help="--report: the folder of the probe, if it is not at its first place")
    o.add_argument("--port", type=int, default=8890, help="the test port (default 8890); never a live port")
    o.add_argument("--tokens", type=int, default=400_000, help="the prompt tokens of the sample capture")
    o.add_argument("--turns", type=int, default=20, help="the held-out turns that each point scores")
    o.add_argument("--hub", action="append", default=[], metavar="DIR", help="one more cache folder to search")
    o.set_defaults(run=cmd_probe)
    j = sub.add_parser("judge", help="a blind judge compares the recorded turns of the full model and of a point")
    j.add_argument("ref", metavar="REF", help="the recorded turns of the full model")
    j.add_argument("new", metavar="NEW", help="the recorded turns of the pruned point, for the same targets")
    j.add_argument("--workflow", required=True, metavar="FILE", help="the workflow file: its data, and gates.judge")
    j.add_argument("--data", metavar="DIR", help="a data folder with sessions.jsonl, in place of the workflow's")
    j.add_argument("--base-url", required=True, metavar="URL", help="the server of the judge model")
    j.add_argument("--served", metavar="NAME", help="the judge model (default: gates.judge.model, or the first model)")
    j.add_argument("--out", required=True, metavar="FILE", help="the result, for example judge-k208.json in the run")
    j.add_argument("--page", metavar="FILE", help="also write the blind pairs as an HTML page for your own review")
    j.add_argument("--body", default="{}", metavar="JSON", help="more keys for each judge request")
    j.set_defaults(run=cmd_judge)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
