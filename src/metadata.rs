use crate::error::Error;
use itertools::Itertools;
use ome_metadata::Ome;
use ome_metadata::ome::{
    BinningType, Convert, Image, Instrument, Objective, Pixels, UnitsLength, UnitsTime,
};

impl Metadata for Ome {
    fn get_instrument(&self) -> Option<&Instrument> {
        let instrument_id = self.get_image()?.instrument_ref.as_ref()?.id.clone();
        self.instrument
            .as_ref()?
            .iter()
            .find(|i| i.id == instrument_id)
    }

    fn get_image(&self) -> Option<&Image> {
        if let Some(image) = &self.image {
            if !image.is_empty() {
                Some(&image[0])
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub trait Metadata {
    fn get_instrument(&self) -> Option<&Instrument>;
    fn get_image(&self) -> Option<&Image>;

    fn get_pixels(&self) -> Option<&Pixels> {
        if let Some(image) = self.get_image() {
            Some(&image.pixels)
        } else {
            None
        }
    }

    fn get_objective(&self) -> Option<&Objective> {
        let objective_id = self.get_image()?.objective_settings.as_ref()?.id.clone();
        self.get_instrument()?
            .objective
            .iter()
            .find(|o| o.id == objective_id)
    }

    fn get_tube_lens(&self) -> Option<Objective> {
        Some(Objective {
            manufacturer: None,
            model: Some("Unknown".to_string()),
            serial_number: None,
            lot_number: None,
            id: "TubeLens:1".to_string(),
            correction: None,
            immersion: None,
            lens_na: None,
            nominal_magnification: Some(1.0),
            calibrated_magnification: None,
            working_distance: None,
            working_distance_unit: UnitsLength::um,
            iris: None,
            annotation_ref: vec![],
        })
    }

    /// shape of the data along cztyx axes
    fn shape(&self) -> Result<(usize, usize, usize, usize, usize), Error> {
        if let Some(pixels) = self.get_pixels() {
            Ok((
                pixels.size_c as usize,
                pixels.size_z as usize,
                pixels.size_t as usize,
                pixels.size_y as usize,
                pixels.size_x as usize,
            ))
        } else {
            Err(Error::NoImageOrPixels)
        }
    }

    /// pixel size in nm
    fn pixel_size(&self) -> Result<Option<f64>, Error> {
        if let Some(pixels) = self.get_pixels() {
            match (pixels.physical_size_x, pixels.physical_size_y) {
                (Some(x), Some(y)) => Ok(Some(
                    (pixels
                        .physical_size_x_unit
                        .convert(&UnitsLength::nm, x as f64)?
                        + pixels
                            .physical_size_y_unit
                            .convert(&UnitsLength::nm, y as f64)?)
                        / 2f64,
                )),
                (Some(x), None) => Ok(Some(
                    pixels
                        .physical_size_x_unit
                        .convert(&UnitsLength::nm, x as f64)?
                        .powi(2),
                )),
                (None, Some(y)) => Ok(Some(
                    pixels
                        .physical_size_y_unit
                        .convert(&UnitsLength::nm, y as f64)?
                        .powi(2),
                )),
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    /// distance between planes in z-stack in nm
    fn delta_z(&self) -> Result<Option<f64>, Error> {
        if let Some(pixels) = self.get_pixels() {
            if let Some(z) = pixels.physical_size_z {
                return Ok(Some(
                    pixels
                        .physical_size_z_unit
                        .convert(&UnitsLength::nm, z as f64)?,
                ));
            }
        }
        Ok(None)
    }

    /// time interval in seconds for time-lapse images
    fn time_interval(&self) -> Result<Option<f64>, Error> {
        if let Some(pixels) = self.get_pixels() {
            if let Some(plane) = &pixels.plane {
                if let Some(t) = plane.iter().map(|p| p.the_t).max() {
                    if t > 0 {
                        let plane_a = plane
                            .iter()
                            .find(|p| (p.the_c == 0) && (p.the_z == 0) && (p.the_t == 0));
                        let plane_b = plane
                            .iter()
                            .find(|p| (p.the_c == 0) && (p.the_z == 0) && (p.the_t == t));
                        if let (Some(a), Some(b)) = (plane_a, plane_b) {
                            if let (Some(a_t), Some(b_t)) = (a.delta_t, b.delta_t) {
                                return Ok(Some(
                                    (b.delta_t_unit.convert(&UnitsTime::s, b_t as f64)?
                                        - a.delta_t_unit.convert(&UnitsTime::s, a_t as f64)?)
                                    .abs()
                                        / (t as f64),
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    /// exposure time for channel, z=0 and t=0
    fn exposure_time(&self, channel: usize) -> Result<Option<f64>, Error> {
        let c = channel as i32;
        if let Some(pixels) = self.get_pixels() {
            if let Some(plane) = &pixels.plane {
                if let Some(p) = plane
                    .iter()
                    .find(|p| (p.the_c == c) && (p.the_z == 0) && (p.the_t == 0))
                {
                    if let Some(t) = p.exposure_time {
                        return Ok(Some(p.exposure_time_unit.convert(&UnitsTime::s, t as f64)?));
                    }
                }
            }
        }
        Ok(None)
    }

    fn binning(&self, channel: usize) -> Option<usize> {
        match self
            .get_pixels()?
            .channel
            .get(channel)?
            .detector_settings
            .as_ref()?
            .binning
            .as_ref()?
        {
            BinningType::_1X1 => Some(1),
            BinningType::_2X2 => Some(2),
            BinningType::_4X4 => Some(4),
            BinningType::_8X8 => Some(8),
            BinningType::Other => None,
        }
    }

    fn laser_wavelengths(&self, channel: usize) -> Result<Option<f64>, Error> {
        if let Some(pixels) = self.get_pixels() {
            if let Some(channel) = pixels.channel.get(channel) {
                if let Some(w) = channel.excitation_wavelength {
                    return Ok(Some(
                        channel
                            .excitation_wavelength_unit
                            .convert(&UnitsLength::nm, w as f64)?,
                    ));
                }
            }
        }
        Ok(None)
    }

    fn laser_powers(&self, channel: usize) -> Result<Option<f64>, Error> {
        if let Some(pixels) = self.get_pixels() {
            if let Some(channel) = pixels.channel.get(channel) {
                if let Some(ls) = &channel.light_source_settings {
                    if let Some(a) = ls.attenuation {
                        return if (0. ..=1.).contains(&a) {
                            Ok(Some(1f64 - (a as f64)))
                        } else {
                            Err(Error::InvalidAttenuation(a.to_string()))
                        };
                    }
                }
            }
        }
        Ok(None)
    }

    fn objective_name(&self) -> Option<String> {
        Some(self.get_objective()?.model.as_ref()?.clone())
    }

    fn magnification(&self) -> Option<f64> {
        Some(
            (self.get_objective()?.nominal_magnification? as f64)
                * (self.get_tube_lens()?.nominal_magnification? as f64),
        )
    }

    fn tube_lens_name(&self) -> Option<String> {
        self.get_tube_lens()?.model.clone()
    }

    fn filter_set_name(&self, channel: usize) -> Option<String> {
        let filter_set_id = self
            .get_pixels()?
            .channel
            .get(channel)?
            .filter_set_ref
            .as_ref()?
            .id
            .clone();
        self.get_instrument()
            .as_ref()?
            .filter_set
            .iter()
            .find(|f| f.id == filter_set_id)?
            .model
            .clone()
    }

    fn gain(&self, channel: usize) -> Option<f64> {
        if let Some(pixels) = self.get_pixels() {
            Some(
                *pixels
                    .channel
                    .get(channel)?
                    .detector_settings
                    .as_ref()?
                    .gain
                    .as_ref()? as f64,
            )
        } else {
            None
        }
    }

    fn summary(&self) -> Result<String, Error> {
        let size_c = if let Some(pixels) = self.get_pixels() {
            pixels.channel.len()
        } else {
            0
        };
        let mut s = "".to_string();
        if let Ok(Some(pixel_size)) = self.pixel_size() {
            s.push_str(&format!("pixel size:    {pixel_size:.2} nm\n"));
        }
        if let Ok(Some(delta_z)) = self.delta_z() {
            s.push_str(&format!("z-interval:    {delta_z:.2} nm\n"))
        }
        if let Ok(Some(time_interval)) = self.time_interval() {
            s.push_str(&format!("time interval: {time_interval:.2} s\n"))
        }
        let exposure_time = (0..size_c)
            .map(|c| self.exposure_time(c))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !exposure_time.is_empty() {
            s.push_str(&format!(
                "exposure time: {}\n",
                exposure_time.into_iter().join(" | ")
            ));
        }
        if let Some(magnification) = self.magnification() {
            s.push_str(&format!("magnification: {magnification}x\n"))
        }
        if let Some(objective_name) = self.objective_name() {
            s.push_str(&format!("objective:     {objective_name}\n"))
        }
        if let Some(tube_lens_name) = self.tube_lens_name() {
            s.push_str(&format!("tube lens:     {tube_lens_name}\n"))
        }
        let filter_set_name = (0..size_c)
            .map(|c| self.filter_set_name(c))
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !filter_set_name.is_empty() {
            s.push_str(&format!(
                "filter set:    {}\n",
                filter_set_name.into_iter().join(" | ")
            ));
        }
        let gain = (0..size_c)
            .map(|c| self.gain(c))
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !gain.is_empty() {
            s.push_str(&format!("gain:          {}\n", gain.into_iter().join(" ")));
        }
        let laser_wavelengths = (0..size_c)
            .map(|c| self.laser_wavelengths(c))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !laser_wavelengths.is_empty() {
            s.push_str(&format!(
                "laser colors:  {} nm\n",
                laser_wavelengths.into_iter().join(" | ")
            ));
        }
        let laser_powers = (0..size_c)
            .map(|c| self.laser_powers(c))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !laser_powers.is_empty() {
            s.push_str(&format!(
                "laser powers:   {}\n",
                laser_powers.into_iter().join(" | ")
            ));
        }
        let binning = (0..size_c)
            .map(|c| self.binning(c))
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if !binning.is_empty() {
            s.push_str(&format!(
                "binning:       {}\n",
                binning.into_iter().join(" | ")
            ));
        }
        Ok(s.to_string())
    }
}
