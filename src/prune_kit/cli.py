"""Command line: prune-kit stages."""

import argparse

from . import __version__, stages


def cmd_stages(_args) -> int:
    print(stages.table())
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prune-kit", description="Prune a local model for one workflow.")
    p.add_argument("--version", action="version", version=f"prune-kit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stages", help="list the stages of a prune and the command of each stage")
    s.set_defaults(run=cmd_stages)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
