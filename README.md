# ndbioimage

Exposes (bio) images as a numpy ndarray like object, but without loading the whole
image into memory, reading from the file only when needed. Some metadata is read
and exposed as attributes to the Imread object (TODO: structure data in OME format).
Additionally, it can automatically calculate an affine transform that corrects for
chromatic abberrations etc. and apply it on the fly to the image.

Currently supports imagej tif files, czi files, micromanager tif sequences and anything
bioformats can handle. 

## Installation

    pip install ndbioimage@git+https://github.com/wimpomp/ndbioimage.git

### With bioformats (if java is properly installed)

    pip install ndbioimage[bioformats]@git+https://github.com/wimpomp/ndbioimage.git

### With affine transforms (only for python 3.8, 3.9 and 3.10)

    pip install ndbioimage[transforms]@git+https://github.com/wimpomp/ndbioimage.git

## Usage

- Reading an image file and plotting the frame at channel=2, time=1


    import matplotlib.pyplot as plt
    from ndbioimage import imread
    with imread('image_file.tif', axes='ctxy', dtype=int) as im:
        plt.imshow(im[2, 1])

- Showing some image metadata


    from ndbioimage import imread
    from pprint import pprint
    with imread('image_file.tif') as im:
        pprint(im)

- Slicing the image without loading the image into memory


    from ndbioimage import imread
    with imread('image_file.tif', axes='cztxy') as im:
        sliced_im = im[1, :, :, 100:200, 100:200]

sliced_im is an instance of imread which will load any image data from file only when needed


- Converting (part) of the image to a numpy ndarray


    from ndbioimage import imread
    import numpy as np
    with imread('image_file.tif', axes='cztxy') as im:
        array = np.asarray(im[0, 0])

## Adding more formats
Readers for image formats subclass Imread. When an image reader is imported, Imread will
automatically recognize it and use it to open the appropriate file format. Image readers
subclass Imread and are required to implement the following methods:

- staticmethod _can_open(path): return True if path can be opened by this reader
- \_\_metadata__(self): reads metadata from file and adds them to self as attributes,
  - the shape of the data in the file needs to be set as self.shape = (X, Y, C, Z, T)
  - other attributes like pxsize, acquisitiontime and title can be set here as well
- \_\_frame__(self, c, z, t): return the frame at channel=c, z-slice=z, time=t from the file

Optional methods:
- open(self): maybe open some file
- close(self): close any file handles

Optional fields:
- priority (int): Imread will try readers with a lower number first, default: 99
- do_not_pickle (strings): any attributes that should not be included when the object is pickled,
for example: any file handles

# TODO
- structure the metadata in OME format tree
- re-implement transforms 
