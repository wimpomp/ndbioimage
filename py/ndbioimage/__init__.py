from __future__ import annotations

import os
import warnings
from abc import ABC
from argparse import ArgumentParser
from collections import OrderedDict
from functools import cached_property, wraps
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from tiffwrite import FrameInfo, IJTiffParallel
from tqdm.auto import tqdm

from ome_metadata import Ome
from ome_metadata.ome_metadata_rs import Length  # noqa
from . import ndbioimage_rs as rs  # noqa
from .transforms import Transform, Transforms  # noqa: F401

try:
    __version__ = version(Path(__file__).parent.name)
except Exception:  # noqa
    __version__ = "unknown"

try:
    with open(Path(__file__).parent.parent / ".git" / "HEAD") as g:
        head = g.read().split(":")[1].strip()
    with open(Path(__file__).parent.parent / ".git" / head) as h:
        __git_commit_hash__ = h.read().rstrip("\n")
except Exception:  # noqa
    __git_commit_hash__ = "unknown"

warnings.filterwarnings("ignore", "Reference to unknown ID")
Number = int | float | np.integer | np.floating


for dep_file in (Path(__file__).parent / "deps").glob("*_"):
    dep_file.rename(str(dep_file)[:-1])

if not list((Path(__file__).parent / "jassets").glob("bioformats*.jar")):
    rs.download_bioformats(True)


class TransformTiff(IJTiffParallel):
    """transform frames in a parallel process to speed up saving"""

    def __init__(self, image: Imread, *args: Any, **kwargs: Any) -> None:
        self.image = image
        super().__init__(*args, **kwargs)

    def parallel(self, frame: tuple[int, int, int]) -> tuple[FrameInfo]:
        return ((np.asarray(self.image(*frame)), 0, 0, 0),)


class DequeDict(OrderedDict):
    def __init__(self, maxlen: int = None, *args: Any, **kwargs: Any) -> None:
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        super().__setitem__(*args, **kwargs)
        self.truncate()

    def truncate(self) -> None:
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(False)

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)  # type: ignore
        self.truncate()


def find(obj: Sequence[Any], **kwargs: Any) -> Any:
    for item in obj:
        try:
            if all([getattr(item, key) == value for key, value in kwargs.items()]):
                return item
        except AttributeError:
            pass
    return None


R = TypeVar("R")


def try_default(fun: Callable[..., R], default: Any, *args: Any, **kwargs: Any) -> R:
    try:
        return fun(*args, **kwargs)
    except Exception:  # noqa
        return default


class Shape(tuple):
    def __new__(cls, shape: Sequence[int] | Shape, axes: str = "yxczt") -> Shape:
        if isinstance(shape, Shape):
            axes = shape.axes  # type: ignore
        new = super().__new__(cls, shape)
        new.axes = axes.lower()
        return new  # type: ignore

    def __getitem__(self, n: int | str) -> int | tuple[int]:
        if isinstance(n, str):
            if len(n) == 1:
                return self[self.axes.find(n.lower())] if n.lower() in self.axes else 1
            else:
                return tuple(self[i] for i in n)  # type: ignore
        return super().__getitem__(n)

    @cached_property
    def yxczt(self) -> tuple[int, int, int, int, int]:
        return tuple(self[i] for i in "yxczt")  # type: ignore


class OmeCache(DequeDict):
    """prevent (potentially expensive) rereading of ome data by caching"""

    instance = None

    def __new__(cls) -> OmeCache:
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        super().__init__(64)

    def __reduce__(self) -> tuple[type, tuple]:
        return self.__class__, ()

    def __getitem__(self, path: Path | str | tuple) -> Ome:
        if isinstance(path, tuple):
            return super().__getitem__(path)
        else:
            return super().__getitem__(self.path_and_lstat(path))

    def __setitem__(self, path: Path | str | tuple, value: Ome) -> None:
        if isinstance(path, tuple):
            super().__setitem__(path, value)
        else:
            super().__setitem__(self.path_and_lstat(path), value)

    def __contains__(self, path: Path | str | tuple) -> bool:
        if isinstance(path, tuple):
            return super().__contains__(path)
        else:
            return super().__contains__(self.path_and_lstat(path))

    @staticmethod
    def path_and_lstat(
        path: str | Path,
    ) -> tuple[Path, Optional[os.stat_result], Optional[os.stat_result]]:
        path = Path(path)
        return (
            path,
            (path.lstat() if path.exists() else None),
            (
                path.with_suffix(".ome.xml").lstat()
                if path.with_suffix(".ome.xml").exists()
                else None
            ),
        )


def get_positions(path: str | Path) -> Optional[list[int]]:  # noqa
    # TODO
    return None


class Imread(rs.View, np.lib.mixins.NDArrayOperatorsMixin, ABC):
    """class to read image files, while taking good care of important metadata,
    currently optimized for .czi files, but can open anything that bioformats can handle
        path: path to the image file
        optional:
        axes: order of axes, default: cztyx, but omitting any axes with lenght 1
        dtype: datatype to be used when returning frames

        Examples:
            >> im = Imread('/path/to/file.image', axes='czt)
            >> im
             << shows summary
            >> im.shape
             << (15, 26, 1000, 1000)
            >> im.axes
             << 'ztyx'
            >> plt.imshow(im[1, 0])
             << plots frame at position z=1, t=0 (python type indexing)
            >> plt.imshow(im[:, 0].max('z'))
             << plots max-z projection at t=0
            >> im.pxsize
             << 0.09708737864077668 image-plane pixel size in um
            >> im.laserwavelengths
             << [642, 488]
            >> im.laserpowers
             << [0.02, 0.0005] in %

            See __init__ and other functions for more ideas.

            # TODO: argmax, argmin, nanmax, nanmin, nanmean, nansum, nanstd, nanvar, std, var, squeeze
    """

    def __getitem__(self, item):
        new = super().__getitem__(item)
        return Imread(new) if isinstance(new, rs.View) else new

    def __copy__(self):
        Imread(super().__copy__())

    def copy(self):
        Imread(super().copy())

    def astype(self):
        Imread(super().astype())

    def squeeze(self):
        new = super().squeeze()
        return Imread(new) if isinstance(new, rs.View) else new

    def min(self, *args, **kwargs) -> Imread | float:
        new = super().min(*args, **kwargs)
        return Imread(new) if isinstance(new, rs.View) else new

    def max(self, *args, **kwargs) -> Imread | float:
        new = super().max(*args, **kwargs)
        return Imread(new) if isinstance(new, rs.View) else new

    def mean(self, *args, **kwargs) -> Imread | float:
        new = super().mean(*args, **kwargs)
        return Imread(new) if isinstance(new, rs.View) else new

    def sum(self, *args, **kwargs) -> Imread | float:
        new = super().sum(*args, **kwargs)
        return Imread(new) if isinstance(new, rs.View) else new

    def transpose(self, *args, **kwargs) -> Imread | float:
        new = super().transpose(*args, **kwargs)
        return Imread(new) if isinstance(new, rs.View) else new

    def swap_axes(self, *args, **kwargs) -> Imread | float:
        new = super().swap_axes(*args, **kwargs)
        return Imread(new) if isinstance(new, rs.View) else new

    @property
    def T(self) -> Imread | float:
        return Imread(super().T)

    @staticmethod
    def get_positions(path: str | Path) -> Optional[list[int]]:  # noqa
        # TODO
        return None

    @staticmethod
    def as_axis(axis):
        if axis is None:
            return None
        elif isinstance(axis, int):
            return axis
        else:
            return str(axis)

    @wraps(np.moveaxis)
    def moveaxis(self, source, destination):
        raise NotImplementedError("moveaxis is not implemented")

    @wraps(np.ndarray.flatten)
    def flatten(self, *args, **kwargs) -> np.ndarray:
        return np.asarray(self).flatten(*args, **kwargs)

    @wraps(np.ndarray.reshape)
    def reshape(self, *args, **kwargs) -> np.ndarray:
        return np.asarray(self).reshape(*args, **kwargs)  # noqa

    def as_array(self) -> np.ndarray:
        return self.__array__()

    @wraps(np.ndarray.astype)
    def astype(self, dtype: DTypeLike, *_, **__) -> Imread:
        return Imread(super().astype(str(np.dtype(dtype))))

    @staticmethod
    def fix_ome(ome: Ome) -> Ome:
        # fix ome if necessary
        for image in ome.image:
            try:
                if (
                    image.pixels.physical_size_z is None
                    and len(set([plane.the_z for plane in image.pixels.planes])) > 1
                ):
                    z = np.array(
                        [
                            (
                                plane.position_z_unit.convert("um", plane.position_z),
                                plane.the_z,
                            )
                            for plane in image.pixels.planes
                            if plane.the_c == 0 and plane.the_t == 0
                        ]
                    )
                    i = np.argsort(z[:, 1])
                    image.pixels.physical_size_z = (
                        np.nanmean(np.true_divide(*np.diff(z[i], axis=0).T)) * 1e6
                    )
                    image.pixels.physical_size_z_unit = Length("um")  # type: ignore
            except Exception:  # noqa
                pass
        return ome

    @staticmethod
    def read_ome(path: str | Path) -> Optional[Ome]:
        path = Path(path)  # type: ignore
        if path.with_suffix(".ome.xml").exists():
            return Ome.from_xml(path.with_suffix(".ome.xml").read_text())
        return None

    def get_ome(self) -> Ome:
        """OME metadata structure"""
        return Ome.from_xml(self.get_ome_xml())

    @cached_property
    def ome(self) -> Ome:
        cache = OmeCache()
        if self.path not in cache:
            ome = self.read_ome(self.path)
            if ome is None:
                ome = self.get_ome()
            cache[self.path] = self.fix_ome(ome)
        return cache[self.path]

    def is_noise(self, volume: ArrayLike = None) -> bool:
        """True if volume only has noise"""
        if volume is None:
            volume = self
        fft = np.fft.fftn(volume)
        corr = np.fft.fftshift(np.fft.ifftn(fft * fft.conj()).real / np.sum(volume**2))
        return 1 - corr[tuple([0] * corr.ndim)] < 0.0067

    @staticmethod
    def kill_vm() -> None:
        pass

    def save_as_movie(
        self,
        fname: Path | str = None,
        c: int | Sequence[int] = None,  # noqa
        z: int | Sequence[int] = None,  # noqa
        t: str | int | Sequence[int] = None,  # noqa
        colors: tuple[str] = None,
        brightnesses: tuple[float] = None,
        scale: int = None,
        bar: bool = True,
    ) -> None:
        """saves the image as a mp4 or mkv file"""
        from matplotlib.colors import to_rgb
        from skvideo.io import FFmpegWriter

        if t is None:
            t = np.arange(self.shape["t"])
        elif isinstance(t, str):
            t = eval(f"np.arange(self.shape['t'])[{t}]")
        elif np.isscalar(t):
            t = (t,)

        def get_ab(
            tyx: Imread, p: tuple[float, float] = (1, 99)
        ) -> tuple[float, float]:
            s = tyx.flatten()
            s = s[s > 0]
            a, b = np.percentile(s, p)
            if a == b:
                a, b = np.min(s), np.max(s)
            if a == b:
                a, b = 0, 1
            return a, b

        def cframe(
            frame: ArrayLike,
            color: str,
            a: float,
            b: float,
            scale: float = 1,  # noqa
        ) -> np.ndarray:
            color = to_rgb(color)
            frame = (frame - a) / (b - a)
            frame = np.dstack([255 * frame * i for i in color])
            return np.clip(np.round(frame), 0, 255).astype("uint8")

        ab = list(zip(*[get_ab(i) for i in self.transpose("cztyx")]))  # type: ignore
        colors = colors or ("r", "g", "b")[: self.shape["c"]] + max(
            0, self.shape["c"] - 3
        ) * ("w",)
        brightnesses = brightnesses or (1,) * self.shape["c"]
        scale = scale or 1
        shape_x = 2 * ((self.shape["x"] * scale + 1) // 2)
        shape_y = 2 * ((self.shape["y"] * scale + 1) // 2)

        with FFmpegWriter(
            str(fname).format(name=self.path.stem, path=str(self.path.parent)),
            outputdict={
                "-vcodec": "libx264",
                "-preset": "veryslow",
                "-pix_fmt": "yuv420p",
                "-r": "7",
                "-vf": f"setpts={25 / 7}*PTS,scale={shape_x}:{shape_y}:flags=neighbor",
            },
        ) as movie:
            im = self.transpose("tzcyx")  # type: ignore
            for ti in tqdm(t, desc="Saving movie", disable=not bar):
                movie.writeFrame(
                    np.max(
                        [
                            cframe(yx, c, a, b / s, scale)
                            for yx, a, b, c, s in zip(
                                im[ti].max("z"), *ab, colors, brightnesses
                            )
                        ],
                        0,
                    )
                )

    def save_as_tiff(
        self,
        fname: Path | str = None,
        c: int | Sequence[int] = None,
        z: int | Sequence[int] = None,
        t: int | Sequence[int] = None,
        split: bool = False,
        bar: bool = True,
        pixel_type: str = "uint16",
        **kwargs: Any,
    ) -> None:
        """saves the image as a tif file
        split: split channels into different files"""
        fname = Path(str(fname).format(name=self.path.stem, path=str(self.path.parent)))
        if fname is None:
            fname = self.path.with_suffix(".tif")
            if fname == self.path:
                raise FileExistsError(f"File {fname} exists already.")
        if not isinstance(fname, Path):
            fname = Path(fname)
        if split:
            for i in range(self.shape["c"]):
                if self.timeseries:
                    self.save_as_tiff(
                        fname.with_name(f"{fname.stem}_C{i:01d}").with_suffix(".tif"),
                        i,
                        0,
                        None,
                        False,
                        bar,
                        pixel_type,
                    )
                else:
                    self.save_as_tiff(
                        fname.with_name(f"{fname.stem}_C{i:01d}").with_suffix(".tif"),
                        i,
                        None,
                        0,
                        False,
                        bar,
                        pixel_type,
                    )
        else:
            n = [c, z, t]
            for i, ax in enumerate("czt"):
                if n[i] is None:
                    n[i] = range(self.shape[ax])
                elif not isinstance(n[i], (tuple, list)):
                    n[i] = (n[i],)

            shape = [len(i) for i in n]
            with TransformTiff(
                self,
                fname.with_suffix(".tif"),
                dtype=pixel_type,
                pxsize=self.pxsize,
                deltaz=self.deltaz,
                **kwargs,
            ) as tif:
                for i, m in tqdm(  # noqa
                    zip(product(*[range(s) for s in shape]), product(*n)),  # noqa
                    total=np.prod(shape),
                    desc="Saving tiff",
                    disable=not bar,
                ):
                    tif.save(m, *i)  # type: ignore

    def with_transform(
        self,
        channels: bool = True,
        drift: bool = False,
        file: Path | str = None,
        bead_files: Sequence[Path | str] = (),
    ) -> Imread:
        """returns a view where channels and/or frames are registered with an affine transformation
        channels: True/False register channels using bead_files
        drift: True/False register frames to correct drift
        file: load registration from file with name file, default: transform.yml in self.path.parent
        bead_files: files used to register channels, default: files in self.path.parent,
            with names starting with 'beads'
        """
        raise NotImplementedError("transforms are not yet implemented")
        # view = self.copy()
        # if file is None:
        #     file = Path(view.path.parent) / 'transform.yml'
        # else:
        #     file = Path(file)
        # if not bead_files:
        #     try:
        #         bead_files = Transforms.get_bead_files(view.path.parent)
        #     except Exception:  # noqa
        #         if not file.exists():
        #             raise Exception('No transform file and no bead file found.')
        #         bead_files = ()
        #
        # if channels:
        #     try:
        #         view.transform = Transforms.from_file(file, T=drift)
        #     except Exception:  # noqa
        #         view.transform = Transforms().with_beads(view.cyllens, bead_files)
        #         if drift:
        #             view.transform = view.transform.with_drift(view)
        #         view.transform.save(file.with_suffix('.yml'))
        #         view.transform.save_channel_transform_tiff(bead_files, file.with_suffix('.tif'))
        # elif drift:
        #     try:
        #         view.transform = Transforms.from_file(file, C=False)
        #     except Exception:  # noqa
        #         view.transform = Transforms().with_drift(self)
        # view.transform.adapt(view.frameoffset, view.shape.yxczt, view.channel_names)
        # return view


def main() -> None:
    parser = ArgumentParser(description="Display info and save as tif")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("file", help="image_file", type=str, nargs="*")
    parser.add_argument(
        "-w",
        "--write",
        help="path to tif/movie out, {folder}, {name} and {ext} take this from file in",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-o", "--extract_ome", help="extract ome to xml file", action="store_true"
    )
    parser.add_argument(
        "-r", "--register", help="register channels", action="store_true"
    )
    parser.add_argument("-c", "--channel", help="channel", type=int, default=None)
    parser.add_argument("-z", "--zslice", help="z-slice", type=int, default=None)
    parser.add_argument(
        "-t",
        "--time",
        help="time (frames) in python slicing notation",
        type=str,
        default=None,
    )
    parser.add_argument("-s", "--split", help="split channels", action="store_true")
    parser.add_argument("-f", "--force", help="force overwrite", action="store_true")
    parser.add_argument(
        "-C", "--movie-colors", help="colors for channels in movie", type=str, nargs="*"
    )
    parser.add_argument(
        "-B",
        "--movie-brightnesses",
        help="scale brightness of each channel",
        type=float,
        nargs="*",
    )
    parser.add_argument(
        "-S", "--movie-scale", help="upscale movie xy size, int", type=float
    )
    args = parser.parse_args()

    for file in tqdm(args.file, desc="operating on files", disable=len(args.file) == 1):
        file = Path(file)
        with Imread(file) as im:  # noqa
            if args.register:
                im = im.with_transform()  # noqa
            if args.write:
                write = Path(
                    args.write.format(
                        folder=str(file.parent), name=file.stem, ext=file.suffix
                    )
                ).absolute()  # noqa
                write.parent.mkdir(parents=True, exist_ok=True)
                if write.exists() and not args.force:
                    print(
                        f"File {args.write} exists already, add the -f flag if you want to overwrite it."
                    )
                elif write.suffix in (".mkv", ".mp4"):
                    im.save_as_movie(
                        write,
                        args.channel,
                        args.zslice,
                        args.time,
                        args.movie_colors,
                        args.movie_brightnesses,
                        args.movie_scale,
                        bar=len(args.file) == 1,
                    )
                else:
                    im.save_as_tiff(
                        write,
                        args.channel,
                        args.zslice,
                        args.time,
                        args.split,
                        bar=len(args.file) == 1,
                    )
            if args.extract_ome:
                with open(im.path.with_suffix(".ome.xml"), "w") as f:
                    f.write(im.ome.to_xml())
            if len(args.file) == 1:
                print(im.summary)
