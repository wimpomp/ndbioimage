from __future__ import annotations

from abc import ABC
from pathlib import Path

import numpy as np

from .. import rs
from .. import AbstractReader

for file in (Path(__file__).parent / "deps").glob("*_"):
    file.rename(str(file)[:-1])

if not list((Path(__file__).parent / "jassets").glob("bioformats*.jar")):
    rs.download_bioformats(True)


class Reader(AbstractReader, ABC):
    priority = 99  # panic and open with BioFormats
    do_not_pickle = 'reader'

    @staticmethod
    def _can_open(path: Path) -> bool:
        try:
            _ = rs.Reader(path)
            return True
        except Exception:
            return False

    def open(self) -> None:
        self.reader = rs.Reader(str(self.path), int(self.series))

    def __frame__(self, c: int, z: int, t: int) -> np.ndarray:
        return self.reader.get_frame(int(c), int(z), int(t))

    def close(self) -> None:
        self.reader.close()
