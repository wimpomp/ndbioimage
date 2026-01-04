use clap::{Parser, Subcommand};
#[cfg(feature = "movie")]
use ndarray::SliceInfoElem;
use ndbioimage::error::Error;
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
        #[arg(short, long, value_name = "VELOCITY", default_value = "3.6")]
        velocity: f64,
        #[arg(short, long, value_name = "BRIGHTNESS", num_args(1..))]
        brightness: Vec<f64>,
        #[arg(short, long, value_name = "SCALE", default_value = "1.0")]
        scale: f64,
        #[arg(short = 'C', long, value_name = "COLOR", num_args(1..))]
        colors: Vec<String>,
        #[arg(short, long, value_name = "OVERWRITE")]
        overwrite: bool,
        #[arg(short, long, value_name = "REGISTER")]
        register: bool,
        #[arg(short, long, value_name = "CHANNEL")]
        channel: Option<isize>,
        #[arg(short, long, value_name = "ZSLICE")]
        zslice: Option<String>,
        #[arg(short, long, value_name = "TIME")]
        time: Option<String>,
        #[arg(short, long, value_name = "NO-SCALE-BRIGHTNESS")]
        no_scaling: bool,
    },
    /// Download the BioFormats jar into the correct folder
    DownloadBioFormats {
        #[arg(short, long, value_name = "GPL_FORMATS")]
        gpl_formats: bool,
    },
}

#[cfg(feature = "movie")]
fn parse_slice(s: &str) -> Result<SliceInfoElem, Error> {
    let mut t = s
        .trim()
        .replace("..", ":")
        .split(":")
        .map(|i| i.parse().ok())
        .collect::<Vec<Option<isize>>>();
    if t.len() > 3 {
        return Err(Error::Parse(s.to_string()));
    }
    while t.len() < 3 {
        t.push(None);
    }
    match t[..] {
        [Some(start), None, None] => Ok(SliceInfoElem::Index(start)),
        [Some(start), end, None] => Ok(SliceInfoElem::Slice {
            start,
            end,
            step: 1,
        }),
        [Some(start), end, Some(step)] => Ok(SliceInfoElem::Slice { start, end, step }),
        [None, end, None] => Ok(SliceInfoElem::Slice {
            start: 0,
            end,
            step: 1,
        }),
        [None, end, Some(step)] => Ok(SliceInfoElem::Slice {
            start: 0,
            end,
            step,
        }),
        _ => Err(Error::Parse(s.to_string())),
    }
}

pub(crate) fn main() -> Result<(), Error> {
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
            register,
            channel,
            zslice,
            time,
            no_scaling,
        } => {
            let options = MovieOptions::new(
                *speed,
                brightness.to_vec(),
                *scale,
                colors.to_vec(),
                *overwrite,
                *register,
                *no_scaling,
            )?;
            for f in file {
                let (path, series) = split_path_and_series(f)?;
                let view = View::from_path(path, series.unwrap_or(0))?;
                let mut s = [SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }; 5];
                if let Some(channel) = channel {
                    s[0] = SliceInfoElem::Index(*channel);
                };
                if let Some(zslice) = zslice {
                    s[1] = parse_slice(zslice)?;
                }
                if let Some(time) = time {
                    s[2] = parse_slice(time)?;
                }
                view.into_dyn()
                    .slice(s.as_slice())?
                    .save_as_movie(f.with_extension("mp4"), &options)?;
            }
        }
        Commands::DownloadBioFormats { gpl_formats } => {
            ndbioimage::download_bioformats(*gpl_formats)?
        }
    }

    Ok(())
}
