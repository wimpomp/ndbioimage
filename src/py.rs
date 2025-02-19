use crate::bioformats::download_bioformats;
use crate::{Frame, Reader};
use numpy::ToPyArray;
use pyo3::prelude::*;
use std::path::PathBuf;

#[pyclass(subclass)]
#[pyo3(name = "Reader")]
#[derive(Debug)]
struct PyReader {
    reader: Reader,
}

#[pymethods]
impl PyReader {
    #[new]
    fn new(path: &str, series: usize) -> PyResult<Self> {
        let mut path = PathBuf::from(path);
        if path.is_dir() {
            for file in path.read_dir()? {
                if let Ok(f) = file {
                    let p = f.path();
                    if f.path().is_file() & (p.extension() == Some("tif".as_ref())) {
                        path = p;
                        break;
                    }
                }
            }
        }
        Ok(PyReader {
            reader: Reader::new(&path, series as i32)?,
        })
    }

    fn get_frame<'py>(
        &self,
        py: Python<'py>,
        c: usize,
        z: usize,
        t: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.reader.get_frame(c, z, t)? {
            Frame::INT8(arr) => arr.to_pyarray(py).into_any(),
            Frame::UINT8(arr) => arr.to_pyarray(py).into_any(),
            Frame::INT16(arr) => arr.to_pyarray(py).into_any(),
            Frame::UINT16(arr) => arr.to_pyarray(py).into_any(),
            Frame::INT32(arr) => arr.to_pyarray(py).into_any(),
            Frame::UINT32(arr) => arr.to_pyarray(py).into_any(),
            Frame::FLOAT(arr) => arr.to_pyarray(py).into_any(),
            Frame::DOUBLE(arr) => arr.to_pyarray(py).into_any(),
        })
    }

    fn get_ome_xml(&self) -> PyResult<String> {
        Ok(self.reader.get_ome_xml()?)
    }

    fn close(&mut self) -> PyResult<()> {
        self.reader.close()?;
        Ok(())
    }
}

pub(crate) fn ndbioimage_file() -> anyhow::Result<PathBuf> {
    let file = Python::with_gil(|py| {
        py.import("ndbioimage")
            .unwrap()
            .filename()
            .unwrap()
            .to_string()
    });
    Ok(PathBuf::from(file))
}

#[pymodule]
#[pyo3(name = "ndbioimage_rs")]
fn ndbioimage_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReader>()?;

    #[pyfn(m)]
    #[pyo3(name = "download_bioformats")]
    fn py_download_bioformats(gpl_formats: bool) -> PyResult<()> {
        download_bioformats(gpl_formats)?;
        Ok(())
    }

    Ok(())
}
