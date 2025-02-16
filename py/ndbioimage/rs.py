from .ndbioimage_rs import Reader, download_bioformats  # noqa
from pathlib import Path


if not list((Path(__file__).parent / "jassets").glob("bioformats*.jar")):
    download_bioformats(True)

for file in (Path(__file__).parent / "deps").glob("*_"):
    file.rename(str(file)[:-1])
