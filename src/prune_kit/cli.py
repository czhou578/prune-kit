"""Command line: prune-kit stages | preflight."""

import argparse
import json
import sys

from . import __version__, preflight, stages, workflow
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
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
