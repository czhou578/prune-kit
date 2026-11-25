"""Command line: prune-kit stages | preflight | data | capture | rank | build."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__, capture, preflight, rank, sessions, sources, st_moe, stages, workflow
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


def cmd_capture(a) -> int:
    """Exit code 0: each segment has its counts. 1: a segment failed. 2: a fault before the first request."""
    try:
        flow, folder = workflow.load(a.workflow), resolve(a.model, a.hub)
        experts = inspect(folder).max_experts
        data = sessions.load(flow.folder / "sessions.jsonl")
        tools = json.loads(t.read_text()) if (t := flow.folder / "tools.json").exists() else None
        tools = tools.get("tools") if isinstance(tools, dict) else tools
        served = a.served or capture.post(a.base_url, "/v1/models", None)["data"][0]["id"]
    except (ModelNotFound, WeightsError, workflow.WorkflowError, OSError, ValueError) as e:
        print(f"prune-kit: {e}", file=sys.stderr)
        return 2
    out = Path(a.out or Path("runs") / slug(folder) / flow.name / "capture.jsonl")
    failed = capture.run(a.base_url, served, data, tools, out, experts, a.limit, json.loads(a.body), a.generate)
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
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
