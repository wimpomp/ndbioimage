use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
    #[error(transparent)]
    J4rs(#[from] j4rs::errors::J4RsError),
    #[error(transparent)]
    Infallible(#[from] std::convert::Infallible),
    #[error(transparent)]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error(transparent)]
    Ome(#[from] ome_metadata::error::Error),
    #[cfg(feature = "tiff")]
    #[error(transparent)]
    TemplateError(#[from] indicatif::style::TemplateError),
    #[cfg(feature = "tiff")]
    #[error(transparent)]
    TiffWrite(#[from] tiffwrite::error::Error),
    #[error("invalid axis: {0}")]
    InvalidAxis(String),
    #[error("axis {0} not found in axes {1}")]
    AxisNotFound(String, String),
    #[error("conversion error: {0}")]
    TryInto(String),
    #[error("file already exists {0}")]
    FileAlreadyExists(String),
    #[error("could not download ffmpeg: {0}")]
    Ffmpeg(String),
    #[error("index {0} out of bounds {1}")]
    OutOfBounds(isize, isize),
    #[error("axis {0} has length {1}, but was not included")]
    OutOfBoundsAxis(String, usize),
    #[error("dimensionality mismatch: {0} != {0}")]
    DimensionalityMismatch(usize, usize),
    #[error("axis {0}: {1} is already operated on!")]
    AxisAlreadyOperated(usize, String),
    #[error("not enough free dimensions")]
    NotEnoughFreeDimensions,
    #[error("cannot cast {0} to {1}")]
    Cast(String, String),
    #[error("empty view")]
    EmptyView,
    #[error("invalid color: {0}")]
    InvalidColor(String),
    #[error("no image or pixels found")]
    NoImageOrPixels,
    #[error("invalid attenuation value: {0}")]
    InvalidAttenuation(String),
    #[error("not a valid file name")]
    InvalidFileName,
    #[error("unknown pixel type {0}")]
    UnknownPixelType(String),
    #[error("no mean")]
    NoMean,
    #[error("tiff is locked")]
    TiffLock,
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("cannot parse: {0}")]
    Parse(String),
}
