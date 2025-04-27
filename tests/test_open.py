import pickle
from pathlib import Path

import pytest

from ndbioimage import Imread


@pytest.mark.parametrize(
    "file",
    [
        file
        for file in (Path(__file__).parent / "files").iterdir()
        if not file.suffix == ".pzl"
    ],
)
def test_open(file):
    with Imread(file, axes="cztyx") as im:
        mean = im[0, 0, 0].mean()
        b = pickle.dumps(im)
        jm = pickle.loads(b)
        assert jm.get_frame(0, 0, 0).mean() == mean
        b = pickle.dumps(im)
        jm = pickle.loads(b)
        assert jm[0, 0, 0].mean() == mean
