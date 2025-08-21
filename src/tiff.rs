use crate::colors::Color;
use crate::metadata::Metadata;
use crate::reader::PixelType;
use crate::stats::MinMax;
use crate::view::{Number, View};
use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::iproduct;
use ndarray::{Array0, Array1, Array2, ArrayD, Dimension};
use rayon::prelude::*;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tiffwrite::{Bytes, Colors, Compression, IJTiffFile};

#[derive(Clone)]
pub struct TiffOptions {
    bar: Option<ProgressStyle>,
    compression: Compression,
    colors: Option<Vec<Vec<u8>>>,
    overwrite: bool,
}

impl Default for TiffOptions {
    fn default() -> Self {
        Self {
            bar: None,
            compression: Compression::Zstd(10),
            colors: None,
            overwrite: false,
        }
    }
}

impl TiffOptions {
    pub fn new(
        bar: bool,
        compression: Option<Compression>,
        colors: Vec<String>,
        overwrite: bool,
    ) -> Result<Self> {
        let mut options = Self {
            bar: None,
            compression: compression.unwrap_or(Compression::Zstd(10)),
            colors: None,
            overwrite,
        };
        if bar {
            options.enable_bar()?;
        }
        if !colors.is_empty() {
            options.set_colors(&colors)?;
        }
        Ok(options)
    }

    /// show a progress bar while saving tiff
    pub fn enable_bar(&mut self) -> Result<()> {
        self.bar = Some(ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}, {percent}%] [{wide_bar:.green/lime}] {pos:>7}/{len:7} ({eta_precise}, {per_sec:<5})",
        )?.progress_chars("▰▱▱"));
        Ok(())
    }

    /// do not show a progress bar while saving tiff
    pub fn disable_bar(&mut self) {
        self.bar = None;
    }

    /// save tiff with zstd compression (default)
    pub fn set_zstd_compression(&mut self) {
        self.compression = Compression::Zstd(10)
    }

    /// save tiff with zstd compression, choose a level between 7..=22
    pub fn set_zstd_compression_level(&mut self, level: i32) {
        self.compression = Compression::Zstd(level)
    }

    /// save tiff with deflate compression
    pub fn set_deflate_compression(&mut self) {
        self.compression = Compression::Deflate
    }

    pub fn set_colors(&mut self, colors: &[String]) -> Result<()> {
        let colors = colors
            .iter()
            .map(|c| c.parse::<Color>())
            .collect::<Result<Vec<_>>>()?;
        self.colors = Some(colors.into_iter().map(|c| c.to_rgb()).collect());
        Ok(())
    }

    pub fn set_overwrite(&mut self, overwrite: bool) {
        self.overwrite = overwrite;
    }
}

impl<D> View<D>
where
    D: Dimension,
{
    /// save as tiff with a certain type
    pub fn save_as_tiff_with_type<T, P>(&self, path: P, options: &TiffOptions) -> Result<()>
    where
        P: AsRef<Path>,
        T: Bytes + Number + Send + Sync,
        ArrayD<T>: MinMax<Output = ArrayD<T>>,
        Array1<T>: MinMax<Output = Array0<T>>,
        Array2<T>: MinMax<Output = Array1<T>>,
    {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            if options.overwrite {
                std::fs::remove_file(&path)?;
            } else {
                return Err(anyhow!("File {} already exists", path.display()));
            }
        }
        let size_c = self.size_c();
        let size_z = self.size_z();
        let size_t = self.size_t();
        let mut tiff = IJTiffFile::new(path)?;
        tiff.set_compression(options.compression.clone());
        let ome = self.get_ome()?;
        tiff.px_size = ome.pixel_size()?.map(|i| i / 1e3);
        tiff.time_interval = ome.time_interval()?.map(|i| i / 1e3);
        tiff.delta_z = ome.delta_z()?.map(|i| i / 1e3);
        tiff.comment = Some(self.ome_xml()?);
        if let Some(mut colors) = options.colors.clone() {
            while colors.len() < self.size_c {
                colors.push(vec![255, 255, 255]);
            }
            tiff.colors = Colors::Colors(colors);
        }
        let tiff = Arc::new(Mutex::new(tiff));
        if let Some(style) = &options.bar {
            let bar = ProgressBar::new((size_c as u64) * (size_z as u64) * (size_t as u64))
                .with_style(style.clone());
            iproduct!(0..size_c, 0..size_z, 0..size_t)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(c, z, t)| {
                    if let Ok(mut tiff) = tiff.lock() {
                        tiff.save(&self.get_frame::<T, _>(c, z, t)?, c, z, t)?;
                        bar.inc(1);
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("tiff is locked"))
                    }
                })
                .collect::<Result<()>>()?;
            bar.finish();
        } else {
            iproduct!(0..size_c, 0..size_z, 0..size_t)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(c, z, t)| {
                    if let Ok(mut tiff) = tiff.lock() {
                        tiff.save(&self.get_frame::<T, _>(c, z, t)?, c, z, t)?;
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("tiff is locked"))
                    }
                })
                .collect::<Result<()>>()?;
        };
        Ok(())
    }

    /// save as tiff with whatever pixel type the view has
    pub fn save_as_tiff<P>(&self, path: P, options: &TiffOptions) -> Result<()>
    where
        P: AsRef<Path>,
    {
        match self.pixel_type {
            PixelType::I8 => self.save_as_tiff_with_type::<i8, P>(path, options)?,
            PixelType::U8 => self.save_as_tiff_with_type::<u8, P>(path, options)?,
            PixelType::I16 => self.save_as_tiff_with_type::<i16, P>(path, options)?,
            PixelType::U16 => self.save_as_tiff_with_type::<u16, P>(path, options)?,
            PixelType::I32 => self.save_as_tiff_with_type::<i32, P>(path, options)?,
            PixelType::U32 => self.save_as_tiff_with_type::<u32, P>(path, options)?,
            PixelType::F32 => self.save_as_tiff_with_type::<f32, P>(path, options)?,
            PixelType::F64 => self.save_as_tiff_with_type::<f64, P>(path, options)?,
            PixelType::I64 => self.save_as_tiff_with_type::<i64, P>(path, options)?,
            PixelType::U64 => self.save_as_tiff_with_type::<u64, P>(path, options)?,
            PixelType::I128 => self.save_as_tiff_with_type::<i64, P>(path, options)?,
            PixelType::U128 => self.save_as_tiff_with_type::<u64, P>(path, options)?,
            PixelType::F128 => self.save_as_tiff_with_type::<f64, P>(path, options)?,
        }

        Ok(())
    }
}
