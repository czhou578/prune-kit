"""Command line: prune-kit stages | preflight | data."""

import argparse
import json
import sys

from . import __version__, preflight, sessions, sources, stages, workflow
from .facts import inspect
from .resolve import ModelNotFound, resolve
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
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
