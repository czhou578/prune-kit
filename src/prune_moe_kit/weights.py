"""The tensor list of a checkpoint folder, from the headers of its safetensors files. No tensor bytes are read."""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER = 512 << 20   # a larger header length means that the file is not a safetensors file


class WeightsError(Exception):
    """The folder has no weight file that prune-moe-kit can read."""


class FormatError(WeightsError):
    """The weights have a format that prune-moe-kit cannot copy by bytes."""


@dataclass(frozen=True)
class Tensor:
    name: str
    file: str               # the file name in the folder
    dtype: str
    shape: tuple[int, ...] | None   # None: the header of the file is not readable
    size: int                       # bytes; an equal share of the file size where the header is not readable

    @property
    def draft(self) -> bool:
        """A tensor of the draft model: the ``mtp.`` tensors, and each tensor of a file with the name ``mtp*``."""
        return self.name.startswith("mtp.") or self.file.startswith("mtp")


def read_header(path: Path) -> dict:
    """The header of a safetensors file, without ``__metadata__``."""
    with open(path, "rb") as f:
        raw = f.read(8)
        length = struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
        if not 0 < length <= MAX_HEADER:
            raise WeightsError(f"{path}: not a safetensors file")
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def _indexed(folder: Path) -> dict[str, list[str]]:
    """``{file name: tensor names}`` from ``model.safetensors.index.json``, or ``{}`` with no index."""
    path, out = folder / "model.safetensors.index.json", {}
    if path.is_file():
        for name, file in json.loads(path.read_text()).get("weight_map", {}).items():
            out.setdefault(file, []).append(name)
    return out


def tensors(folder: Path) -> list[Tensor]:
    """Each tensor of each ``*.safetensors`` file of the folder, in file order.

    A file with no read permission gives the names of the index of the folder. Each of its tensors gets an equal
    share of the file size. That is exact for a group of experts with the same tensors each (shape: None)."""
    folder = Path(folder)
    files = sorted(folder.glob("*.safetensors"))
    if not files and any(folder.glob("*.gguf")):
        raise FormatError(f"{folder}: GGUF files. prune-moe-kit copies the bytes of safetensors files only")
    if not files:
        raise WeightsError(f"{folder}: no safetensors file. The weights are not on disk, or they have another format")
    out, index = [], None
    for path in files:
        try:
            for name, t in read_header(path).items():
                start, end = t["data_offsets"]
                out.append(Tensor(name, path.name, t["dtype"], tuple(t["shape"]), end - start))
        except PermissionError:
            index = _indexed(folder) if index is None else index
            names = index.get(path.name)
            if not names:
                raise WeightsError(f"{path}: no permission to read the file, and the index does not list it")
            out += [Tensor(name, path.name, "?", None, path.stat().st_size // len(names)) for name in names]
    return out
