use anyhow::Result;
use clap::{Parser, Subcommand};
#[cfg(feature = "movie")]
use ndbioimage::movie::MovieOptions;
use ndbioimage::reader::split_path_and_series;
#[cfg(feature = "tiff")]
use ndbioimage::tiff::TiffOptions;
use ndbioimage::view::View;
use std::path::PathBuf;

#[derive(Parser)]
#[command(arg_required_else_help = true, version, about, long_about = None, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print some metadata
    Info {
        #[arg(value_name = "FILE", num_args(1..))]
        file: Vec<PathBuf>,
    },
    /// Save the image as tiff file
    #[cfg(feature = "tiff")]
    Tiff {
        #[arg(value_name = "FILE", num_args(1..))]
        file: Vec<PathBuf>,
        #[arg(short, long, value_name = "COLOR", num_args(1..))]
        colors: Vec<String>,
        #[arg(short, long, value_name = "OVERWRITE")]
        overwrite: bool,
    },
    /// Save the image as mp4 file
    #[cfg(feature = "movie")]
    Movie {
        #[arg(value_name = "FILE", num_args(1..))]
        file: Vec<PathBuf>,
        #[arg(short, long, value_name = "Velocity", default_value = "3.6")]
        velocity: f64,
        #[arg(short, long, value_name = "BRIGHTNESS")]
        brightness: Vec<f64>,
        #[arg(short, long, value_name = "SCALE", default_value = "1.0")]
        scale: f64,
        #[arg(short, long, value_name = "COLOR", num_args(1..))]
        colors: Vec<String>,
        #[arg(short, long, value_name = "OVERWRITE")]
        overwrite: bool,
    },
    /// Download the BioFormats jar into the correct folder
    DownloadBioFormats {
        #[arg(short, long, value_name = "GPL_FORMATS")]
        gpl_formats: bool,
    },
}

pub(crate) fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Info { file } => {
            for f in file {
                let (path, series) = split_path_and_series(f)?;
                let view = View::from_path(path, series.unwrap_or(0))?.squeeze()?;
                println!("{}", view.summary()?);
            }
        }
        #[cfg(feature = "tiff")]
        Commands::Tiff {
            file,
            colors,
            overwrite,
        } => {
            let mut options = TiffOptions::new(true, None, colors.clone(), *overwrite)?;
            options.enable_bar()?;
            for f in file {
                let (path, series) = split_path_and_series(f)?;
                let view = View::from_path(path, series.unwrap_or(0))?;
                view.save_as_tiff(f.with_extension("tiff"), &options)?;
            }
        }
        #[cfg(feature = "movie")]
        Commands::Movie {
            file,
            velocity: speed,
            brightness,
            scale,
            colors,
            overwrite,
        } => {
            let options = MovieOptions::new(
                *speed,
                brightness.to_vec(),
                *scale,
                colors.to_vec(),
                *overwrite,
            )?;
            for f in file {
                let (path, series) = split_path_and_series(f)?;
                let view = View::from_path(path, series.unwrap_or(0))?;
                view.save_as_movie(f.with_extension("mp4"), &options)?;
            }
        }
        Commands::DownloadBioFormats { gpl_formats } => {
            ndbioimage::download_bioformats(*gpl_formats)?
        }
    }

    Ok(())
}
