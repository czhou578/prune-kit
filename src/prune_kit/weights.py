"""The tensor list of a checkpoint folder, from the headers of its safetensors files. No tensor bytes are read."""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER = 512 << 20   # a larger header length means that the file is not a safetensors file


class WeightsError(Exception):
    """The folder has no weight file that prune-kit can read."""


@dataclass(frozen=True)
class Tensor:
    name: str
    file: str               # the file name in the folder
    dtype: str
    shape: tuple[int, ...]
    size: int               # bytes

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


def tensors(folder: Path) -> list[Tensor]:
    """Each tensor of each ``*.safetensors`` file of the folder, in file order."""
    files = sorted(Path(folder).glob("*.safetensors"))
    if not files:
        raise WeightsError(f"{folder}: no safetensors file. The weights are not on disk, or they have another format")
    out = []
    for path in files:
        for name, t in read_header(path).items():
            start, end = t["data_offsets"]
            out.append(Tensor(name, path.name, t["dtype"], tuple(t["shape"]), end - start))
    return out
