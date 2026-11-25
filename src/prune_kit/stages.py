"""The stages of a prune (BLUEPRINT.md), and the command of each stage that prune-kit has today."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    question: str
    command: str | None   # None: prune-kit has no command for this stage yet


STAGES = (
    Stage(0, "preflight", "How much memory does each cut free, and can the pipeline run on this model?",
          "prune-kit preflight"),
    Stage(1, "data", "Which recorded sessions are the workflow, and which sessions are held out?",
          "prune-kit data"),
    Stage(2, "capture", "Which experts does the model route the workflow to?", "prune-kit capture"),
    Stage(3, "rank", "Which experts of each layer does the workflow use least?", "prune-kit rank"),
    Stage(4, "screen", "Which cuts pass a short replay?", None),
    Stage(5, "prune", "Does the pruned folder hold the kept experts, byte for byte?", "prune-kit build"),
    Stage(6, "gates", "Does the point pass the replay, the tools, the tasks, the judge, the speed and the memory?",
          "prune-kit gate"),
    Stage(7, "decision", "Which point do you choose?", None),
)


def table() -> str:
    """One line per stage: the number, the name, the command or 'by hand', and the question."""
    rows = [f"{s.number}  {s.name:<10} {s.command or 'by hand':<22} {s.question}" for s in STAGES]
    return "\n".join(rows)
