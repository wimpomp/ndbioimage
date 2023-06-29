import numpy as np
from ome_types import model
from functools import cached_property
from abc import ABC
from ndbioimage import Imread
from itertools import product


class Reader(Imread, ABC):
    priority = 20
    do_not_pickle = 'ome'
    do_not_copy = 'ome'

    @staticmethod
    def _can_open(path):
        return isinstance(path, np.ndarray) and 1 <= path.ndim <= 5

    @cached_property
    def ome(self):
        def shape(size_x=1, size_y=1, size_c=1, size_z=1, size_t=1):
            return size_x, size_y, size_c, size_z, size_t
        size_x, size_y, size_c, size_z, size_t = shape(*self.array.shape)
        try:
            pixel_type = model.simple_types.PixelType(self.array.dtype.name)
        except ValueError:
            if self.array.dtype.name.startswith('int'):
                pixel_type = model.simple_types.PixelType('int32')
            else:
                pixel_type = model.simple_types.PixelType('float')

        ome = model.OME()
        ome.instruments.append(model.Instrument())
        ome.images.append(
            model.Image(
                pixels=model.Pixels(
                    size_c=size_c, size_z=size_z, size_t=size_t, size_x=size_x, size_y=size_y,
                    dimension_order="XYCZT", type=pixel_type,
                    objective_settings=model.ObjectiveSettings(id="Objective:0"))))
        for c, z, t in product(range(size_c), range(size_z), range(size_t)):
            ome.images[0].pixels.planes.append(model.Plane(the_c=c, the_z=z, the_t=t, delta_t=0))
        return ome

    def open(self):
        self.array = np.array(self.path)
        self.path = 'numpy array'

    def __frame__(self, c, z, t):
        # xyczt = (slice(None), slice(None), c, z, t)
        # in_idx = tuple(xyczt['xyczt'.find(i)] for i in self.axes)
        # print(f'{in_idx = }')
        frame = self.array[:, :, c, z, t]
        if self.axes.find('y') < self.axes.find('x'):
            return frame.T
        else:
            return frame
