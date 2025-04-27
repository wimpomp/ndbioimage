import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from tiffwrite import tiffwrite

from ndbioimage import Imread


@pytest.fixture
def array():
    return np.random.randint(0, 255, (64, 64, 2, 3, 4), "uint16")


@pytest.fixture()
def image(array):
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder) / "test.tif"
        tiffwrite(file, array, "yxczt")
        with Imread(file, axes="yxczt") as im:
            yield im


@pytest.mark.parametrize(
    "fun_and_axis",
    product(
        (
            np.sum,
            np.nansum,
            np.min,
            np.nanmin,
            np.max,
            np.nanmax,
            np.argmin,
            np.argmax,
            np.mean,
            np.nanmean,
            np.var,
            np.nanvar,
            np.std,
            np.nanstd,
        ),
        (None, 0, 1, 2, 3, 4),
    ),
)
def test_ufuncs(fun_and_axis, image, array):
    fun, axis = fun_and_axis
    assert np.all(np.isclose(np.asarray(fun(image, axis)), fun(array, axis))), (
        f"function {fun.__name__} over axis {axis} does not give the correct result"
    )
