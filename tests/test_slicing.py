import tempfile
from itertools import combinations_with_replacement
from numbers import Number
from pathlib import Path

import numpy as np
import pytest
from tiffwrite import tiffwrite

from ndbioimage import Imread


@pytest.fixture
def array():
    return np.random.randint(0, 255, (64, 64, 2, 3, 4), "uint8")


@pytest.fixture()
def image(array):
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder) / "test.tif"
        tiffwrite(file, array, "yxczt")
        with Imread(file, axes="yxczt") as im:
            yield im


@pytest.mark.parametrize(
    "s",
    combinations_with_replacement(
        (0, -1, 1, slice(None), slice(0, 1), slice(-1, 0), slice(1, 1)), 5
    ),
)
def test_slicing(s, image, array):
    s_im, s_a = image[s], array[s]
    if isinstance(s_a, Number):
        assert isinstance(s_im, Number)
        assert s_im == s_a
    else:
        assert isinstance(s_im, Imread)
        assert tuple(s_im.shape) == s_a.shape
        assert np.all(s_im == s_a)
