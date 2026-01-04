use crate::axes::Axis;
use crate::bioformats;
use crate::bioformats::{DebugTools, ImageReader, MetadataTools};
use crate::error::Error;
use crate::view::View;
use ndarray::{Array2, Ix5, s};
use num::{FromPrimitive, Zero};
use ome_metadata::Ome;
use serde::{Deserialize, Serialize};
use std::any::type_name;
use std::fmt::Debug;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use thread_local::ThreadLocal;

pub fn split_path_and_series<P>(path: P) -> Result<(PathBuf, Option<usize>), Error>
where
    P: Into<PathBuf>,
{
    let path = path.into();
    let file_name = path
        .file_name()
        .ok_or(Error::InvalidFileName)?
        .to_str()
        .ok_or(Error::InvalidFileName)?;
    if file_name.to_lowercase().starts_with("pos") {
        if let Some(series) = file_name.get(3..) {
            if let Ok(series) = series.parse::<usize>() {
                return Ok((path, Some(series)));
            }
        }
    }
    Ok((path, None))
}

/// Pixel types (u)int(8/16/32) or float(32/64), (u/i)(64/128) are not included in bioformats
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PixelType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
    I128,
    U128,
    F128,
}

impl TryFrom<i32> for PixelType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PixelType::I8),
            1 => Ok(PixelType::U8),
            2 => Ok(PixelType::I16),
            3 => Ok(PixelType::U16),
            4 => Ok(PixelType::I32),
            5 => Ok(PixelType::U32),
            6 => Ok(PixelType::F32),
            7 => Ok(PixelType::F64),
            8 => Ok(PixelType::I64),
            9 => Ok(PixelType::U64),
            10 => Ok(PixelType::I128),
            11 => Ok(PixelType::U128),
            12 => Ok(PixelType::F128),
            _ => Err(Error::UnknownPixelType(value.to_string())),
        }
    }
}

impl FromStr for PixelType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "int8" | "i8" => Ok(PixelType::I8),
            "uint8" | "u8" => Ok(PixelType::U8),
            "int16" | "i16" => Ok(PixelType::I16),
            "uint16" | "u16" => Ok(PixelType::U16),
            "int32" | "i32" => Ok(PixelType::I32),
            "uint32" | "u32" => Ok(PixelType::U32),
            "float" | "f32" | "float32" => Ok(PixelType::F32),
            "double" | "f64" | "float64" => Ok(PixelType::F64),
            "int64" | "i64" => Ok(PixelType::I64),
            "uint64" | "u64" => Ok(PixelType::U64),
            "int128" | "i128" => Ok(PixelType::I128),
            "uint128" | "u128" => Ok(PixelType::U128),
            "extended" | "f128" => Ok(PixelType::F128),
            _ => Err(Error::UnknownPixelType(s.to_string())),
        }
    }
}

/// Struct containing frame data in one of eight pixel types. Cast to `Array2<T>` using try_into.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub enum Frame {
    I8(Array2<i8>),
    U8(Array2<u8>),
    I16(Array2<i16>),
    U16(Array2<u16>),
    I32(Array2<i32>),
    U32(Array2<u32>),
    F32(Array2<f32>),
    F64(Array2<f64>),
    I64(Array2<i64>),
    U64(Array2<u64>),
    I128(Array2<i128>),
    U128(Array2<u128>),
    F128(Array2<f64>), // f128 is nightly
}

macro_rules! impl_frame_cast {
    ($($t:tt: $s:ident $(,)?)*) => {
        $(
            impl From<Array2<$t>> for Frame {
                fn from(value: Array2<$t>) -> Self {
                    Frame::$s(value)
                }
            }
        )*
    };
}

impl_frame_cast! {
    u8: U8
    i8: I8
    i16: I16
    u16: U16
    i32: I32
    u32: U32
    f32: F32
    f64: F64
    i64: I64
    u64: U64
    i128: I128
    u128: U128
}

#[cfg(target_pointer_width = "32")]
impl_frame_cast! {
    usize: UINT32
    isize: INT32
}

impl<T> TryInto<Array2<T>> for Frame
where
    T: FromPrimitive + Zero + 'static,
{
    type Error = Error;

    fn try_into(self) -> Result<Array2<T>, Self::Error> {
        let mut err = Ok(());
        let arr = match self {
            Frame::I8(v) => v.mapv_into_any(|x| {
                T::from_i8(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::U8(v) => v.mapv_into_any(|x| {
                T::from_u8(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::I16(v) => v.mapv_into_any(|x| {
                T::from_i16(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::U16(v) => v.mapv_into_any(|x| {
                T::from_u16(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::I32(v) => v.mapv_into_any(|x| {
                T::from_i32(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::U32(v) => v.mapv_into_any(|x| {
                T::from_u32(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::F32(v) => v.mapv_into_any(|x| {
                T::from_f32(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::F64(v) | Frame::F128(v) => v.mapv_into_any(|x| {
                T::from_f64(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::I64(v) => v.mapv_into_any(|x| {
                T::from_i64(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::U64(v) => v.mapv_into_any(|x| {
                T::from_u64(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::I128(v) => v.mapv_into_any(|x| {
                T::from_i128(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
                    T::zero()
                })
            }),
            Frame::U128(v) => v.mapv_into_any(|x| {
                T::from_u128(x).unwrap_or_else(|| {
                    err = Err(Error::Cast(x.to_string(), type_name::<T>().to_string()));
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
#[derive(Serialize, Deserialize)]
pub struct Reader {
    #[serde(skip)]
    image_reader: ThreadLocal<ImageReader>,
    /// path to file
    pub path: PathBuf,
    /// which (if more than 1) of the series in the file to open
    pub series: usize,
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

impl Deref for Reader {
    type Target = ImageReader;

    fn deref(&self) -> &Self::Target {
        self.get_reader().unwrap()
    }
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
    /// Create a new reader for the image file at a path, and open series #.
    pub fn new<P>(path: P, series: usize) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        DebugTools::set_root_level("ERROR")?;
        let mut reader = Reader {
            image_reader: ThreadLocal::default(),
            path: path.as_ref().to_path_buf(),
            series,
            size_x: 0,
            size_y: 0,
            size_c: 0,
            size_z: 0,
            size_t: 0,
            pixel_type: PixelType::I8,
            little_endian: false,
        };
        reader.set_reader()?;
        reader.size_x = reader.get_size_x()? as usize;
        reader.size_y = reader.get_size_y()? as usize;
        reader.size_c = reader.get_size_c()? as usize;
        reader.size_z = reader.get_size_z()? as usize;
        reader.size_t = reader.get_size_t()? as usize;
        reader.pixel_type = PixelType::try_from(reader.get_pixel_type()?)?;
        reader.little_endian = reader.is_little_endian()?;
        Ok(reader)
    }

    fn get_reader(&self) -> Result<&ImageReader, Error> {
        self.image_reader.get_or_try(|| {
            let reader = ImageReader::new()?;
            let meta_data_tools = MetadataTools::new()?;
            let ome_meta = meta_data_tools.create_ome_xml_metadata()?;
            reader.set_metadata_store(ome_meta)?;
            reader.set_id(self.path.to_str().ok_or(Error::InvalidFileName)?)?;
            reader.set_series(self.series as i32)?;
            Ok(reader)
        })
    }

    pub fn set_reader(&self) -> Result<(), Error> {
        self.get_reader().map(|_| ())
    }

    /// Get ome metadata as ome structure
    pub fn get_ome(&self) -> Result<Ome, Error> {
        let mut ome = self.ome_xml()?.parse::<Ome>()?;
        if let Some(image) = ome.image.as_ref() {
            if image.len() > 1 {
                ome.image = Some(vec![image[self.series].clone()]);
            }
        }
        Ok(ome)
    }

    /// Get ome metadata as xml string
    pub fn get_ome_xml(&self) -> Result<String, Error> {
        self.ome_xml()
    }

    fn deinterleave(&self, bytes: Vec<u8>, channel: usize) -> Result<Vec<u8>, Error> {
        let chunk_size = match self.pixel_type {
            PixelType::I8 => 1,
            PixelType::U8 => 1,
            PixelType::I16 => 2,
            PixelType::U16 => 2,
            PixelType::I32 => 4,
            PixelType::U32 => 4,
            PixelType::F32 => 4,
            PixelType::F64 => 8,
            PixelType::I64 => 8,
            PixelType::U64 => 8,
            PixelType::I128 => 16,
            PixelType::U128 => 16,
            PixelType::F128 => 8,
        };
        Ok(bytes
            .chunks(chunk_size)
            .skip(channel)
            .step_by(self.size_c)
            .flat_map(|a| a.to_vec())
            .collect())
    }

    /// Retrieve fame at channel c, slize z and time t.
    #[allow(clippy::if_same_then_else)]
    pub fn get_frame(&self, c: usize, z: usize, t: usize) -> Result<Frame, Error> {
        let bytes = if self.is_rgb()? && self.is_interleaved()? {
            let index = self.get_index(z as i32, 0, t as i32)?;
            self.deinterleave(self.open_bytes(index)?, c)?
        } else if self.get_rgb_channel_count()? > 1 {
            let channel_separator = bioformats::ChannelSeparator::new(self)?;
            let index = channel_separator.get_index(z as i32, c as i32, t as i32)?;
            channel_separator.open_bytes(index)?
        } else if self.is_indexed()? {
            let index = self.get_index(z as i32, c as i32, t as i32)?;
            self.open_bytes(index)?
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
            let index = self.get_index(z as i32, c as i32, t as i32)?;
            self.open_bytes(index)?
        };
        self.bytes_to_frame(bytes)
    }

    fn bytes_to_frame(&self, bytes: Vec<u8>) -> Result<Frame, Error> {
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
            (PixelType::I8, true) => get_frame!(i8, <1),
            (PixelType::U8, true) => get_frame!(u8, <1),
            (PixelType::I16, true) => get_frame!(i16, <2),
            (PixelType::U16, true) => get_frame!(u16, <2),
            (PixelType::I32, true) => get_frame!(i32, <4),
            (PixelType::U32, true) => get_frame!(u32, <4),
            (PixelType::F32, true) => get_frame!(f32, <4),
            (PixelType::F64, true) => get_frame!(f64, <8),
            (PixelType::I64, true) => get_frame!(i64, <8),
            (PixelType::U64, true) => get_frame!(u64, <8),
            (PixelType::I128, true) => get_frame!(i128, <16),
            (PixelType::U128, true) => get_frame!(u128, <16),
            (PixelType::F128, true) => get_frame!(f64, <8),
            (PixelType::I8, false) => get_frame!(i8, >1),
            (PixelType::U8, false) => get_frame!(u8, >1),
            (PixelType::I16, false) => get_frame!(i16, >2),
            (PixelType::U16, false) => get_frame!(u16, >2),
            (PixelType::I32, false) => get_frame!(i32, >4),
            (PixelType::U32, false) => get_frame!(u32, >4),
            (PixelType::F32, false) => get_frame!(f32, >4),
            (PixelType::F64, false) => get_frame!(f64, >8),
            (PixelType::I64, false) => get_frame!(i64, >8),
            (PixelType::U64, false) => get_frame!(u64, >8),
            (PixelType::I128, false) => get_frame!(i128, >16),
            (PixelType::U128, false) => get_frame!(u128, >16),
            (PixelType::F128, false) => get_frame!(f64, >8),
        }
    }

    /// get a sliceable view on the image file
    pub fn view(&self) -> View<Ix5> {
        let slice = s![
            0..self.size_c,
            0..self.size_z,
            0..self.size_t,
            0..self.size_y,
            0..self.size_x
        ];
        View::new(
            Arc::new(self.clone()),
            slice.as_ref().to_vec(),
            vec![Axis::C, Axis::Z, Axis::T, Axis::Y, Axis::X],
        )
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        if let Ok(reader) = self.get_reader() {
            reader.close().unwrap();
        }
    }
}
