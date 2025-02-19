import pickle
from pathlib import Path

import pytest

from ndbioimage import Imread


@pytest.mark.parametrize('file',
                         [file for file in (Path(__file__).parent / 'files').iterdir() if not file.suffix == '.pzl'])
def test_open(file):
    with Imread(file) as im:
        mean = im[dict(c=0, z=0, t=0)].mean()
        b = pickle.dumps(im)
        jm = pickle.loads(b)
        assert jm[dict(c=0, z=0, t=0)].mean() == mean
        v = im.view()
        assert v[dict(c=0, z=0, t=0)].mean() == mean
        b = pickle.dumps(v)
        w = pickle.loads(b)
        assert w[dict(c=0, z=0, t=0)].mean() == mean
