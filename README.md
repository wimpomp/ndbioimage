# ndbioimage
[![Pytest](https://github.com/wimpomp/ndbioimage/actions/workflows/pytest.yml/badge.svg)](https://github.com/wimpomp/ndbioimage/actions/workflows/pytest.yml)

## Work in progress
Rust rewrite of python version. Read bio image formats using the bio-formats java package.
[https://www.openmicroscopy.org/bio-formats/](https://www.openmicroscopy.org/bio-formats/)

Exposes (bio) images as a numpy ndarray-like object, but without loading the whole
image into memory, reading from the file only when needed. Some metadata is read
and stored in an [ome](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2005-6-5-r47) structure.
Additionally, it can automatically calculate an affine transform that corrects for chromatic aberrations etc. and apply
it on the fly to the image.

Currently, it supports imagej tif files, czi files, micromanager tif sequences and anything
[bioformats](https://www.openmicroscopy.org/bio-formats/) can handle.

## Installation

```
pip install ndbioimage
```

### Installation with option to write mp4 or mkv:
Work in progress! Make sure ffmpeg is installed.

```
pip install ndbioimage[write]
```

## Usage
### Python

- Reading an image file and plotting the frame at channel=2, time=1

```
import matplotlib.pyplot as plt
from ndbioimage import Imread
with Imread('image_file.tif', axes='ctyx', dtype=int) as im:
    plt.imshow(im[2, 1])
```        

- Showing some image metadata

```
from ndbioimage import Imread
from pprint import pprint
with Imread('image_file.tif') as im:
    pprint(im)
```

- Slicing the image without loading the image into memory

```
from ndbioimage import Imread
with Imread('image_file.tif', axes='cztyx') as im:
    sliced_im = im[1, :, :, 100:200, 100:200]
```

sliced_im is an instance of Imread which will load any image data from file only when needed


- Converting (part) of the image to a numpy ndarray

```
from ndbioimage import Imread
import numpy as np
with Imread('image_file.tif', axes='cztyx') as im:
    array = np.asarray(im[0, 0])
```

### Rust
```
use ndarray::Array2;
use ndbioimage::Reader;

let path = "/path/to/file";
let reader = Reader::new(&path, 0)?;
println!("size: {}, {}", reader.size_y, reader.size_y);
let frame = reader.get_frame(0, 0, 0).unwrap();
if let Ok(arr) = <Frame as TryInto<Array2<i8>>>::try_into(frame) {
    println!("{:?}", arr);
} else {
    println!("could not convert Frame to Array<i8>");
}
let xml = reader.get_ome_xml().unwrap();
println!("{}", xml);
```

``` 
use ndarray::Array2;
use ndbioimage::Reader;

let path = "/path/to/file";
let reader = Reader::new(&path, 0)?;
let view = reader.view();
let view = view.max_proj(3)?;
let array = view.as_array::<u16>()?
```

### Command line
```ndbioimage --help```: show help  
```ndbioimage image```: show metadata about image  
```ndbioimage image -w {name}.tif -r```: copy image into image.tif (replacing {name} with image), while registering channels  
```ndbioimage image -w image.mp4 -C cyan lime red``` copy image into image.mp4 (z will be max projected), make channel colors cyan lime and red
