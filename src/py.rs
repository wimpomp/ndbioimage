use crate::{Frame, Reader};
use numpy::{IntoPyArray, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::BoundObject;
use std::path::PathBuf;

#[pyclass(subclass)]
#[pyo3(name = "Reader")]
#[derive(Debug)]
struct PyReader {
    path: PathBuf,
    series: i32,
}

#[pymethods]
impl PyReader {
    #[new]
    fn new(path: &str, series: usize) -> PyResult<Self> {
        Ok(PyReader {
            path: PathBuf::from(path),
            series: series as i32,
        })
    }

    fn get_frame<'py>(
        &self,
        py: Python<'py>,
        c: usize,
        z: usize,
        t: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reader = Reader::new(&self.path, self.series)?; // TODO: prevent making a new Reader each time
        Ok(match reader.get_frame(c, z, t)? {
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
}

#[pymodule]
#[pyo3(name = "ndbioimage_rs")]
fn ndbioimage_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReader>()?;
    Ok(())
}
