mod bioformats;

#[cfg(feature = "python")]
mod py;

use anyhow::{anyhow, Result};
use bioformats::{DebugTools, ImageReader, MetadataTools};
use ndarray::Array2;
use num::{FromPrimitive, Zero};
use std::any::type_name;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

/// Pixel types (u)int(8/16/32) or float(32/64)
#[derive(Clone, Debug)]
pub enum PixelType {
    INT8 = 0,
    UINT8 = 1,
    INT16 = 2,
    UINT16 = 3,
    INT32 = 4,
    UINT32 = 5,
    FLOAT = 6,
    DOUBLE = 7,
}

impl TryFrom<i32> for PixelType {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PixelType::INT8),
            1 => Ok(PixelType::UINT8),
            2 => Ok(PixelType::INT16),
            3 => Ok(PixelType::UINT16),
            4 => Ok(PixelType::INT32),
            5 => Ok(PixelType::UINT32),
            6 => Ok(PixelType::FLOAT),
            7 => Ok(PixelType::DOUBLE),
            _ => Err(anyhow::anyhow!("Unknown pixel type {}", value)),
        }
    }
}

/// Struct containing frame data in one of eight pixel types. Cast to `Array2<T>` using try_into.
#[derive(Clone, Debug)]
pub enum Frame {
    INT8(Array2<i8>),
    UINT8(Array2<u8>),
    INT16(Array2<i16>),
    UINT16(Array2<u16>),
    INT32(Array2<i32>),
    UINT32(Array2<u32>),
    FLOAT(Array2<f32>),
    DOUBLE(Array2<f64>),
}

macro_rules! impl_frame_cast {
    ($t:tt, $s:ident) => {
        impl From<Array2<$t>> for Frame {
            fn from(value: Array2<$t>) -> Self {
                Frame::$s(value)
            }
        }
    };
}

impl_frame_cast!(i8, INT8);
impl_frame_cast!(u8, UINT8);
impl_frame_cast!(i16, INT16);
impl_frame_cast!(u16, UINT16);
impl_frame_cast!(i32, INT32);
impl_frame_cast!(u32, UINT32);
impl_frame_cast!(f32, FLOAT);
impl_frame_cast!(f64, DOUBLE);

impl<T> TryInto<Array2<T>> for Frame
where
    T: FromPrimitive + Zero + 'static,
{
    type Error = anyhow::Error;

    fn try_into(self) -> std::result::Result<Array2<T>, Self::Error> {
        let mut err = Ok(());
        let arr = match self {
            Frame::INT8(v) => v.mapv_into_any(|x| {
                T::from_i8(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::UINT8(v) => v.mapv_into_any(|x| {
                T::from_u8(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::INT16(v) => v.mapv_into_any(|x| {
                T::from_i16(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::UINT16(v) => v.mapv_into_any(|x| {
                T::from_u16(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::INT32(v) => v.mapv_into_any(|x| {
                T::from_i32(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::UINT32(v) => v.mapv_into_any(|x| {
                T::from_u32(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::FLOAT(v) => v.mapv_into_any(|x| {
                T::from_f32(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
            Frame::DOUBLE(v) => v.mapv_into_any(|x| {
                T::from_f64(x).unwrap_or_else(|| {
                    err = Err(anyhow!("cannot convert {} into {}", x, type_name::<T>()));
                    T::zero()
                })
            }),
        };
        match err {
            Err(err) => Err(err),
            Ok(()) => Ok(arr),
        }
    }
}

/// Reader interface to file. Use get_frame to get data.
pub struct Reader {
    image_reader: ImageReader,
    /// path to file
    pub path: PathBuf,
    /// which (if more than 1) of the series in the file to open
    pub series: i32,
    /// size x (horizontal)
    pub size_x: usize,
    /// size y (vertical)
    pub size_y: usize,
    /// size c (# channels)
    pub size_c: usize,
    /// size z (# slices)
    pub size_z: usize,
    /// size t (# time/frames)
    pub size_t: usize,
    /// pixel type ((u)int(8/16/32) or float(32/64))
    pub pixel_type: PixelType,
    little_endian: bool,
}

impl Clone for Reader {
    fn clone(&self) -> Self {
        Reader::new(&self.path, self.series).unwrap()
    }
}

impl Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("path", &self.path)
            .field("series", &self.series)
            .field("size_x", &self.size_x)
            .field("size_y", &self.size_y)
            .field("size_c", &self.size_c)
            .field("size_z", &self.size_z)
            .field("size_t", &self.size_t)
            .field("pixel_type", &self.pixel_type)
            .field("little_endian", &self.little_endian)
            .finish()
    }
}

impl Reader {
    /// Create new reader for image file at path.
    pub fn new(path: &Path, series: i32) -> Result<Self> {
        DebugTools::set_root_level("ERROR")?;
        let reader = ImageReader::new()?;
        let meta_data_tools = MetadataTools::new()?;
        let ome_meta = meta_data_tools.create_ome_xml_metadata()?;
        reader.set_metadata_store(ome_meta)?;
        reader.set_id(path.to_str().unwrap())?;
        reader.set_series(series)?;
        let size_x = reader.get_size_x()?;
        let size_y = reader.get_size_y()?;
        let size_c = reader.get_size_c()?;
        let size_z = reader.get_size_z()?;
        let size_t = reader.get_size_t()?;
        let pixel_type = PixelType::try_from(reader.get_pixel_type()?)?;
        let little_endian = reader.is_little_endian()?;
        Ok(Reader {
            image_reader: reader,
            path: PathBuf::from(path),
            series,
            size_x: size_x as usize,
            size_y: size_y as usize,
            size_c: size_c as usize,
            size_z: size_z as usize,
            size_t: size_t as usize,
            pixel_type,
            little_endian,
        })
    }

    /// Get ome metadata as xml string
    pub fn ome_xml(&self) -> Result<String> {
        self.image_reader.ome_xml()
    }

    fn deinterleave(&self, bytes: Vec<u8>, channel: usize) -> Result<Vec<u8>> {
        let chunk_size = match self.pixel_type {
            PixelType::INT8 => 1,
            PixelType::UINT8 => 1,
            PixelType::INT16 => 2,
            PixelType::UINT16 => 2,
            PixelType::INT32 => 4,
            PixelType::UINT32 => 4,
            PixelType::FLOAT => 4,
            PixelType::DOUBLE => 8,
        };
        Ok(bytes
            .chunks(chunk_size)
            .skip(channel)
            .step_by(self.size_c)
            .flat_map(|a| a.to_vec())
            .collect())
    }

    /// Retrieve fame at channel c, slize z and time t.
    pub fn get_frame(&self, c: usize, z: usize, t: usize) -> Result<Frame> {
        let bytes = if self.image_reader.is_rgb()? & self.image_reader.is_interleaved()? {
            let index = self.image_reader.get_index(z as i32, 0, t as i32)?;
            self.deinterleave(self.image_reader.open_bytes(index)?, c)?
        } else if self.image_reader.get_rgb_channel_count()? > 1 {
            let channel_separator = bioformats::ChannelSeparator::new(&self.image_reader)?;
            let index = channel_separator.get_index(z as i32, c as i32, t as i32)?;
            channel_separator.open_bytes(index)?
        } else if self.image_reader.is_indexed()? {
            let index = self.image_reader.get_index(z as i32, 0, t as i32)?;
            self.image_reader.open_bytes(index)?
            // TODO: apply LUT
            // let _bytes_lut = match self.pixel_type {
            //     PixelType::INT8 | PixelType::UINT8 => {
            //         let _lut = self.image_reader.get_8bit_lookup_table()?;
            //     }
            //     PixelType::INT16 | PixelType::UINT16 => {
            //         let _lut = self.image_reader.get_16bit_lookup_table()?;
            //     }
            //     _ => {}
            // };
        } else {
            let index = self.image_reader.get_index(z as i32, c as i32, t as i32)?;
            self.image_reader.open_bytes(index)?
        };
        self.bytes_to_frame(bytes)
    }

    fn bytes_to_frame(&self, bytes: Vec<u8>) -> Result<Frame> {
        macro_rules! get_frame {
            ($t:tt, <$n:expr) => {
                Ok(Frame::from(Array2::from_shape_vec(
                    (self.size_y, self.size_x),
                    bytes
                        .chunks($n)
                        .map(|x| $t::from_le_bytes(x.try_into().unwrap()))
                        .collect(),
                )?))
            };
            ($t:tt, >$n:expr) => {
                Ok(Frame::from(Array2::from_shape_vec(
                    (self.size_y, self.size_x),
                    bytes
                        .chunks($n)
                        .map(|x| $t::from_be_bytes(x.try_into().unwrap()))
                        .collect(),
                )?))
            };
        }

        match (&self.pixel_type, self.little_endian) {
            (PixelType::INT8, true) => get_frame!(i8, <1),
            (PixelType::UINT8, true) => get_frame!(u8, <1),
            (PixelType::INT16, true) => get_frame!(i16, <2),
            (PixelType::UINT16, true) => get_frame!(u16, <2),
            (PixelType::INT32, true) => get_frame!(i32, <4),
            (PixelType::UINT32, true) => get_frame!(u32, <4),
            (PixelType::FLOAT, true) => get_frame!(f32, <4),
            (PixelType::DOUBLE, true) => get_frame!(f64, <8),
            (PixelType::INT8, false) => get_frame!(i8, >1),
            (PixelType::UINT8, false) => get_frame!(u8, >1),
            (PixelType::INT16, false) => get_frame!(i16, >2),
            (PixelType::UINT16, false) => get_frame!(u16, >2),
            (PixelType::INT32, false) => get_frame!(i32, >4),
            (PixelType::UINT32, false) => get_frame!(u32, >4),
            (PixelType::FLOAT, false) => get_frame!(f32, >4),
            (PixelType::DOUBLE, false) => get_frame!(f64, >8),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    fn open(file: &str) -> Result<Reader> {
        let path = std::env::current_dir()?
            .join("tests")
            .join("files")
            .join(file);
        Reader::new(&path, 0)
    }

    fn get_pixel_type(file: &str) -> Result<String> {
        let reader = open(file)?;
        Ok(format!(
            "file: {}, pixel type: {:?}",
            file, reader.pixel_type
        ))
    }

    fn get_frame(file: &str) -> Result<Frame> {
        let reader = open(file)?;
        reader.get_frame(0, 0, 0)
    }

    #[test]
    fn read_ser() -> Result<()> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        println!("size: {}, {}", reader.size_y, reader.size_y);
        let frame = reader.get_frame(0, 0, 0)?;
        if let Ok(arr) = <Frame as TryInto<Array2<i8>>>::try_into(frame) {
            println!("{:?}", arr);
        } else {
            println!("could not convert Frame to Array<i8>");
        }
        Ok(())
    }

    #[test]
    fn read_par() -> Result<()> {
        let files = vec!["Experiment-2029.czi", "test.tif"];
        let pixel_type = files
            .into_par_iter()
            .map(|file| get_pixel_type(file).unwrap())
            .collect::<Vec<_>>();
        println!("{:?}", pixel_type);
        Ok(())
    }

    #[test]
    fn read_frame_par() -> Result<()> {
        let files = vec!["Experiment-2029.czi", "test.tif"];
        let frames = files
            .into_par_iter()
            .map(|file| get_frame(file).unwrap())
            .collect::<Vec<_>>();
        println!("{:?}", frames);
        Ok(())
    }

    #[test]
    fn read_sequence() -> Result<()> {
        let file = "YTL1841B2-2-1_1hr_DMSO_galinduction_1/Pos0/img_000000000_mScarlet_GFP-mSc-filter_004.tif";
        let reader = open(file)?;
        println!("reader: {:?}", reader);
        let frame = reader.get_frame(0, 4, 0)?;
        println!("frame: {:?}", frame);
        let frame = reader.get_frame(0, 2, 0)?;
        println!("frame: {:?}", frame);
        Ok(())
    }

    #[test]
    fn read_sequence1() -> Result<()> {
        let file = "4-Pos_001_002/img_000000000_Cy3-Cy3_filter_000.tif";
        let reader = open(file)?;
        println!("reader: {:?}", reader);
        Ok(())
    }

    #[test]
    fn ome_xml() -> Result<()> {
        let file = "Experiment-2029.czi";
        let reader = open(file)?;
        let xml = reader.ome_xml()?;
        println!("{}", xml);
        Ok(())
    }
}
