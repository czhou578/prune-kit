"""Find the folder of a local model from the name that the user gives."""

import os
from pathlib import Path


class ModelNotFound(Exception):
    """No local folder matches the name, or more than one folder matches it."""


def hubs(extra=()) -> list[Path]:
    """The Hugging Face cache folders to search, in this order: ``extra``, the environment, the default."""
    env = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and os.path.join(os.environ["HF_HOME"], "hub")]
    found = [Path(p).expanduser() for p in (*extra, *filter(None, env), "~/.cache/huggingface/hub")]
    return [p for i, p in enumerate(found) if p.is_dir() and p not in found[:i]]


def snapshot(repo: Path) -> Path:
    """The newest snapshot of a repo folder of a cache (``models--org--name``)."""
    snaps = [p for p in (repo / "snapshots").iterdir() if p.is_dir()] if (repo / "snapshots").is_dir() else []
    if not snaps:
        raise ModelNotFound(f"{repo}: the cache folder has no snapshot. Download the model first")
    return max(snaps, key=lambda p: p.stat().st_mtime)


def resolve(model: str, extra_hubs=()) -> Path:
    """The folder that holds ``config.json`` and the weight files of ``model``.

    ``model`` is a folder, a repo folder of a cache, a Hugging Face name (``org/name``), or a part of such a name."""
    path = Path(model).expanduser()
    if path.is_dir():
        return snapshot(path) if (path / "snapshots").is_dir() else path
    repos = [r for hub in hubs(extra_hubs) for r in sorted(hub.glob("models--*"))]
    exact = [r for r in repos if r.name == "models--" + model.replace("/", "--")]
    part = [r for r in repos if model.lower().replace("/", "--") in r.name.lower()]
    match = exact or part
    if len(match) == 1:
        return snapshot(match[0])
    if not match:
        raise ModelNotFound(f"{model}: no folder with this path, and no model with this name in a local cache")
    names = ", ".join(r.name.removeprefix("models--").replace("--", "/", 1) for r in match)
    raise ModelNotFound(f"{model}: more than one local model matches: {names}")
