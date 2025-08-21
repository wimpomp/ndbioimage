use crate::axes::Axis;
use crate::colors::Color;
use crate::view::View;
use anyhow::{Result, anyhow};
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::download::auto_download;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use itertools::Itertools;
use ndarray::{Array2, Array3, Dimension, IxDyn, s, stack};
use ordered_float::OrderedFloat;
use std::io::Write;
use std::path::Path;
use std::thread;

pub struct MovieOptions {
    velocity: f64,
    brightness: Vec<f64>,
    scale: f64,
    colors: Option<Vec<Vec<u8>>>,
    overwrite: bool,
}

impl Default for MovieOptions {
    fn default() -> Self {
        Self {
            velocity: 3.6,
            brightness: Vec::new(),
            scale: 1.0,
            colors: None,
            overwrite: false,
        }
    }
}

impl MovieOptions {
    pub fn new(
        velocity: f64,
        brightness: Vec<f64>,
        scale: f64,
        colors: Vec<String>,
        overwrite: bool,
    ) -> Result<Self> {
        let colors = if colors.is_empty() {
            None
        } else {
            let colors = colors
                .iter()
                .map(|c| c.parse::<Color>())
                .collect::<Result<Vec<_>>>()?;
            Some(colors.into_iter().map(|c| c.to_rgb()).collect())
        };
        Ok(Self {
            velocity,
            brightness,
            scale,
            colors,
            overwrite,
        })
    }

    pub fn set_velocity(&mut self, velocity: f64) {
        self.velocity = velocity;
    }

    pub fn set_brightness(&mut self, brightness: Vec<f64>) {
        self.brightness = brightness;
    }

    pub fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
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

fn get_ab(tyx: View<IxDyn>) -> Result<(f64, f64)> {
    let s = tyx
        .as_array::<f64>()?
        .iter()
        .filter_map(|&i| {
            if i == 0.0 || !i.is_finite() {
                None
            } else {
                Some(OrderedFloat::from(i))
            }
        })
        .sorted_unstable()
        .map(f64::from)
        .collect::<Vec<_>>();

    let n = s.len();
    let mut a = s[n / 100];
    let mut b = s[n - n / 100 - 1];
    if a == b {
        a = s[0];
        b = s[n - 1];
    }
    if a == b {
        a = 1.0;
        b = 1.0;
    }
    Ok((a, b))
}

fn cframe(frame: Array2<f64>, color: &[u8], a: f64, b: f64) -> Array3<f64> {
    let frame = (frame - a) / (b - a);
    let color = color
        .iter()
        .map(|&c| (c as f64) / 255.0)
        .collect::<Vec<_>>();
    let frame = color
        .iter()
        .map(|&c| (c * &frame).to_owned())
        .collect::<Vec<Array2<f64>>>();
    let view = frame.iter().map(|c| c.view()).collect::<Vec<_>>();
    stack(ndarray::Axis(0), &view).unwrap()
}

impl<D> View<D>
where
    D: Dimension,
{
    pub fn save_as_movie<P>(&self, path: P, options: &MovieOptions) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            if options.overwrite {
                std::fs::remove_file(&path)?;
            } else {
                return Err(anyhow!("File {} already exists", path.display()));
            }
        }
        let view = self.max_proj(Axis::Z)?.reset_axes()?;
        let velocity = options.velocity;
        let mut brightness = options.brightness.clone();
        let scale = options.scale;
        let shape = view.shape();
        let size_c = shape[0];
        let size_t = shape[2];
        let size_x = shape[3];
        let size_y = shape[4];
        let shape_x = 2 * (((size_x as f64 * scale) / 2.).round() as usize);
        let shape_y = 2 * (((size_y as f64 * scale) / 2.).round() as usize);

        while brightness.len() < size_c {
            brightness.push(1.0);
        }
        let mut colors = if let Some(colors) = options.colors.as_ref() {
            colors.to_vec()
        } else {
            Vec::new()
        };
        while colors.len() < size_c {
            colors.push(vec![255, 255, 255]);
        }

        auto_download()?;
        let mut movie = FfmpegCommand::new()
            .args([
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                &format!("{}x{}", size_x, size_y),
            ])
            .input("-")
            .args([
                "-vcodec",
                "libx264",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "7",
                "-vf",
                &format!("setpts={velocity}*PTS,scale={shape_x}:{shape_y}:flags=neighbor"),
            ])
            .output(path.to_str().expect("path cannot be converted to string"))
            .spawn()?;
        let mut stdin = movie.take_stdin().unwrap();

        let ab = (0..size_c)
            .map(|c| match view.slice(s![c, .., .., .., ..]) {
                Ok(slice) => get_ab(slice.into_dyn()),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>>>()?;

        thread::spawn(move || {
            for t in 0..size_t {
                let mut frame = Array3::<f64>::zeros((3, size_y, size_y));
                for c in 0..size_c {
                    frame = frame
                        + cframe(
                            view.get_frame(c, 0, t).unwrap(),
                            &colors[c],
                            ab[c].0,
                            ab[c].1 / brightness[c],
                        );
                }
                let frame = frame.mapv(|i| {
                    if i < 0.0 {
                        0
                    } else if i > 1.0 {
                        1
                    } else {
                        (255.0 * i).round() as u8
                    }
                });
                let bytes: Vec<_> = frame.flatten().into_iter().collect();
                stdin.write_all(&bytes).unwrap();
            }
        });

        movie.iter()?.for_each(|e| match e {
            FfmpegEvent::Log(LogLevel::Error, e) => println!("Error: {}", e),
            FfmpegEvent::Progress(p) => println!("Progress: {} / 00:00:15", p.time),
            _ => {}
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Reader;

    #[test]
    fn movie() -> Result<()> {
        let file = "1xp53-01-AP1.czi";
        let path = std::env::current_dir()?
            .join("tests")
            .join("files")
            .join(file);
        let reader = Reader::new(&path, 0)?;
        let view = reader.view();
        let mut options = MovieOptions::default();
        options.set_overwrite(true);
        view.save_as_movie("/home/wim/tmp/movie.mp4", &options)?;
        Ok(())
    }
}
