use crate::axes::Axis;
use crate::bioformats::download_bioformats;
use crate::reader::{PixelType, Reader};
use crate::view::{Item, View};
use anyhow::{anyhow, Result};
use ndarray::{Ix0, IxDyn, SliceInfoElem};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyInt, PyList, PySlice, PySliceMethods, PyString, PyTuple};
use pyo3::IntoPyObjectExt;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use std::path::PathBuf;

#[pyclass(module = "ndbioimage.ndbioimage_rs")]
struct ViewConstructor;

#[pymethods]
impl ViewConstructor {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __getstate__(&self) -> (u8,) {
        (0,)
    }

    fn __setstate__(&self, _state: (u8,)) {}

    #[staticmethod]
    fn __call__(state: String) -> PyResult<PyView> {
        if let Ok(new) = from_str(&state) {
            Ok(new)
        } else {
            Err(anyhow!("cannot parse state").into())
        }
    }
}

#[pyclass(subclass, module = "ndbioimage.ndbioimage_rs")]
#[pyo3(name = "View")]
#[derive(Debug, Serialize, Deserialize)]
struct PyView {
    view: View<IxDyn>,
    dtype: PixelType,
}

#[pymethods]
impl PyView {
    #[new]
    #[pyo3(signature = (path, series = 0, dtype = "uint16"))]
    /// new view on a file at path, open series #, open as dtype: (u)int(8/16/32) or float(32/64)
    fn new(path: &str, series: usize, dtype: &str) -> PyResult<Self> {
        let mut path = PathBuf::from(path);
        if path.is_dir() {
            for file in path.read_dir()?.flatten() {
                let p = file.path();
                if file.path().is_file() & (p.extension() == Some("tif".as_ref())) {
                    path = p;
                    break;
                }
            }
        }
        Ok(Self {
            view: Reader::new(&path, series as i32)?.view().into_dyn(),
            dtype: dtype.parse()?,
        })
    }

    /// close the file: does nothing as this is handled automatically
    fn close(&self) -> PyResult<()> {
        Ok(())
    }

    fn copy(&self) -> PyView {
        PyView {
            view: self.view.clone(),
            dtype: self.dtype.clone(),
        }
    }

    /// slice the view and return a new view or a single number
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        n: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slice: Vec<_> = if n.is_instance_of::<PyTuple>() {
            n.downcast_into::<PyTuple>()?.into_iter().collect()
        } else if n.is_instance_of::<PyList>() {
            n.downcast_into::<PyList>()?.into_iter().collect()
        } else {
            vec![n]
        };
        let mut new_slice = Vec::new();
        let mut ellipsis = None;
        let shape = self.view.shape();
        for (i, (s, t)) in slice.iter().zip(shape.iter()).enumerate() {
            if s.is_instance_of::<PyInt>() {
                new_slice.push(SliceInfoElem::Index(
                    s.downcast::<PyInt>()?.extract::<isize>()?,
                ));
            } else if s.is_instance_of::<PySlice>() {
                let u = s.downcast::<PySlice>()?.indices(*t as isize)?;
                new_slice.push(SliceInfoElem::Slice {
                    start: u.start,
                    end: Some(u.stop),
                    step: u.step,
                });
            } else if s.is_instance_of::<PyEllipsis>() {
                if ellipsis.is_some() {
                    return Err(anyhow!("cannot have more than one ellipsis").into());
                }
                let _ = ellipsis.insert(i);
            } else {
                return Err(anyhow!("cannot convert {:?} to slice", s).into());
            }
        }
        if new_slice.len() > shape.len() {
            return Err(anyhow!(
                "got more indices ({}) than dimensions ({})",
                new_slice.len(),
                shape.len()
            )
            .into());
        }
        while new_slice.len() < shape.len() {
            if let Some(i) = ellipsis {
                new_slice.insert(
                    i,
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    },
                )
            } else {
                new_slice.push(SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                })
            }
        }
        let view = self.view.slice(new_slice.as_slice())?;
        if view.ndim() == 0 {
            Ok(match self.dtype {
                PixelType::I8 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i8>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U8 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u8>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I16 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i16>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U16 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u16>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F32 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f32>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U64 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u64>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::I128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<i128>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::U128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<u128>()?
                    .into_pyobject(py)?
                    .into_any(),
                PixelType::F128 => view
                    .into_dimensionality::<Ix0>()?
                    .item::<f64>()?
                    .into_pyobject(py)?
                    .into_any(),
            })
        } else {
            PyView {
                view,
                dtype: self.dtype.clone(),
            }
            .into_bound_py_any(py)
        }
    }

    fn __reduce__(&self) -> PyResult<(ViewConstructor, (String,))> {
        if let Ok(s) = to_string(self) {
            Ok((ViewConstructor, (s,)))
        } else {
            Err(anyhow!("cannot get state").into())
        }
    }

    /// retrieve a single frame at czt, sliced accordingly
    fn get_frame<'py>(
        &self,
        py: Python<'py>,
        c: isize,
        z: isize,
        t: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.dtype {
            PixelType::I8 => self
                .view
                .get_frame::<i8>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U8 => self
                .view
                .get_frame::<u8>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I16 => self
                .view
                .get_frame::<i16>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U16 => self
                .view
                .get_frame::<u16>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I32 => self
                .view
                .get_frame::<i32>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U32 => self
                .view
                .get_frame::<u32>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::F32 => self
                .view
                .get_frame::<f32>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::F64 => self
                .view
                .get_frame::<f64>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I64 => self
                .view
                .get_frame::<i64>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U64 => self
                .view
                .get_frame::<u64>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::I128 => self
                .view
                .get_frame::<i64>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::U128 => self
                .view
                .get_frame::<u64>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
            PixelType::F128 => self
                .view
                .get_frame::<f64>(c, z, t)?
                .into_pyarray(py)
                .into_any(),
        })
    }

    /// retrieve the ome metadata as an xml string
    fn get_ome_xml(&self) -> PyResult<String> {
        Ok(self.view.get_ome_xml()?)
    }

    /// the file path
    #[getter]
    fn path(&self) -> PyResult<String> {
        Ok(self.view.path.display().to_string())
    }

    /// the series in the file
    #[getter]
    fn series(&self) -> PyResult<i32> {
        Ok(self.view.series)
    }

    /// the axes in the view
    #[getter]
    fn axes(&self) -> Vec<String> {
        self.view
            .axes()
            .iter()
            .map(|a| format!("{:?}", a))
            .collect()
    }

    /// the shape of the view
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.view.shape()
    }

    #[getter]
    fn slice(&self) -> PyResult<Vec<String>> {
        Ok(self
            .view
            .get_slice()
            .iter()
            .map(|s| format!("{:#?}", s))
            .collect())
    }

    /// the number of pixels in the view
    #[getter]
    fn size(&self) -> usize {
        self.view.size()
    }

    /// the number of dimensions in the view
    #[getter]
    fn ndim(&self) -> usize {
        self.view.ndim()
    }

    /// find the position of an axis
    #[pyo3(text_signature = "axis: str | int")]
    fn get_ax(&self, axis: Bound<'_, PyAny>) -> PyResult<usize> {
        if axis.is_instance_of::<PyString>() {
            let axis = axis
                .downcast_into::<PyString>()?
                .extract::<String>()?
                .parse::<Axis>()?;
            Ok(self
                .view
                .axes()
                .iter()
                .position(|a| *a == axis)
                .ok_or_else(|| anyhow!("cannot find axis {:?}", axis))?)
        } else if axis.is_instance_of::<PyInt>() {
            Ok(axis.downcast_into::<PyInt>()?.extract::<usize>()?)
        } else {
            Err(anyhow!("cannot convert to axis").into())
        }
    }

    /// swap two axes
    #[pyo3(text_signature = "ax0: str | int, ax1: str | int")]
    fn swap_axes(&self, ax0: Bound<'_, PyAny>, ax1: Bound<'_, PyAny>) -> PyResult<Self> {
        let ax0 = self.get_ax(ax0)?;
        let ax1 = self.get_ax(ax1)?;
        let view = self.view.swap_axes(ax0, ax1)?;
        Ok(PyView {
            view,
            dtype: self.dtype.clone(),
        })
    }

    /// permute the order of the axes
    #[pyo3(signature = (axes = None), text_signature = "axes: list[str | int] = None")]
    fn transpose(&self, axes: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<Self> {
        let view = if let Some(axes) = axes {
            let ax = axes
                .into_iter()
                .map(|a| self.get_ax(a))
                .collect::<Result<Vec<_>, _>>()?;
            self.view.permute_axes(&ax)?
        } else {
            self.view.transpose()?
        };
        Ok(PyView {
            view,
            dtype: self.dtype.clone(),
        })
    }

    /// collect data into a numpy array
    fn as_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.dtype {
            PixelType::I8 => self.view.as_array_dyn::<i8>()?.into_pyarray(py).into_any(),
            PixelType::U8 => self.view.as_array_dyn::<u8>()?.into_pyarray(py).into_any(),
            PixelType::I16 => self.view.as_array_dyn::<i16>()?.into_pyarray(py).into_any(),
            PixelType::U16 => self.view.as_array_dyn::<u16>()?.into_pyarray(py).into_any(),
            PixelType::I32 => self.view.as_array_dyn::<i32>()?.into_pyarray(py).into_any(),
            PixelType::U32 => self.view.as_array_dyn::<u32>()?.into_pyarray(py).into_any(),
            PixelType::F32 => self.view.as_array_dyn::<f32>()?.into_pyarray(py).into_any(),
            PixelType::F64 => self.view.as_array_dyn::<f64>()?.into_pyarray(py).into_any(),
            PixelType::I64 => self.view.as_array_dyn::<i64>()?.into_pyarray(py).into_any(),
            PixelType::U64 => self.view.as_array_dyn::<u64>()?.into_pyarray(py).into_any(),
            PixelType::I128 => self.view.as_array_dyn::<i64>()?.into_pyarray(py).into_any(),
            PixelType::U128 => self.view.as_array_dyn::<u64>()?.into_pyarray(py).into_any(),
            PixelType::F128 => self.view.as_array_dyn::<f64>()?.into_pyarray(py).into_any(),
        })
    }

    /// change the data type of the view: (u)int(8/16/32) or float(32/64)
    fn as_type(&self, dtype: String) -> PyResult<Self> {
        Ok(PyView {
            view: self.view.clone(),
            dtype: dtype.parse()?,
        })
    }

    #[getter]
    fn get_dtype(&self) -> PyResult<&str> {
        Ok(match self.dtype {
            PixelType::I8 => "int8",
            PixelType::U8 => "uint8",
            PixelType::I16 => "int16",
            PixelType::U16 => "uint16",
            PixelType::I32 => "int32",
            PixelType::U32 => "uint32",
            PixelType::F32 => "float32",
            PixelType::F64 => "float64",
            PixelType::I64 => "int64",
            PixelType::U64 => "uint64",
            PixelType::I128 => "int128",
            PixelType::U128 => "uint128",
            PixelType::F128 => "float128",
        })
    }

    #[setter]
    fn set_dtype(&mut self, dtype: String) -> PyResult<()> {
        self.dtype = dtype.parse()?;
        Ok(())
    }

    /// get the maximum overall or along a given axis
    #[pyo3(signature = (axis = None), text_signature = "axis: str | int")]
    fn max<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(axis) = axis {
            PyView {
                dtype: self.dtype.clone(),
                view: self.view.max_proj(self.get_ax(axis)?)?,
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::I8 => self.view.max::<i8>()?.into_pyobject(py)?.into_any(),
                PixelType::U8 => self.view.max::<u8>()?.into_pyobject(py)?.into_any(),
                PixelType::I16 => self.view.max::<i16>()?.into_pyobject(py)?.into_any(),
                PixelType::U16 => self.view.max::<u16>()?.into_pyobject(py)?.into_any(),
                PixelType::I32 => self.view.max::<i32>()?.into_pyobject(py)?.into_any(),
                PixelType::U32 => self.view.max::<u32>()?.into_pyobject(py)?.into_any(),
                PixelType::F32 => self.view.max::<f32>()?.into_pyobject(py)?.into_any(),
                PixelType::F64 => self.view.max::<f64>()?.into_pyobject(py)?.into_any(),
                PixelType::I64 => self.view.max::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U64 => self.view.max::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::I128 => self.view.max::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U128 => self.view.max::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::F128 => self.view.max::<f64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    /// get the minimum overall or along a given axis
    #[pyo3(signature = (axis = None), text_signature = "axis: str | int")]
    fn min<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(axis) = axis {
            PyView {
                dtype: self.dtype.clone(),
                view: self.view.min_proj(self.get_ax(axis)?)?,
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::I8 => self.view.min::<i8>()?.into_pyobject(py)?.into_any(),
                PixelType::U8 => self.view.min::<u8>()?.into_pyobject(py)?.into_any(),
                PixelType::I16 => self.view.min::<i16>()?.into_pyobject(py)?.into_any(),
                PixelType::U16 => self.view.min::<u16>()?.into_pyobject(py)?.into_any(),
                PixelType::I32 => self.view.min::<i32>()?.into_pyobject(py)?.into_any(),
                PixelType::U32 => self.view.min::<u32>()?.into_pyobject(py)?.into_any(),
                PixelType::F32 => self.view.min::<f32>()?.into_pyobject(py)?.into_any(),
                PixelType::F64 => self.view.min::<f64>()?.into_pyobject(py)?.into_any(),
                PixelType::I64 => self.view.min::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U64 => self.view.min::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::I128 => self.view.min::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U128 => self.view.min::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::F128 => self.view.min::<f64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    /// get the mean overall or along a given axis
    #[pyo3(signature = (axis = None), text_signature = "axis: str | int")]
    fn mean<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(axis) = axis {
            let dtype = if let PixelType::F32 = self.dtype {
                PixelType::F32
            } else {
                PixelType::F64
            };
            PyView {
                dtype,
                view: self.view.mean_proj(self.get_ax(axis)?)?,
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::F32 => self.view.mean::<f32>()?.into_pyobject(py)?.into_any(),
                _ => self.view.mean::<f64>()?.into_pyobject(py)?.into_any(),
            })
        }
    }

    /// get the sum overall or along a given axis
    #[pyo3(signature = (axis = None), text_signature = "axis: str | int")]
    fn sum<'py>(
        &self,
        py: Python<'py>,
        axis: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dtype = match self.dtype {
            PixelType::I8 => PixelType::I16,
            PixelType::U8 => PixelType::U16,
            PixelType::I16 => PixelType::I32,
            PixelType::U16 => PixelType::U32,
            PixelType::I32 => PixelType::I64,
            PixelType::U32 => PixelType::U64,
            PixelType::F32 => PixelType::F32,
            PixelType::F64 => PixelType::F64,
            PixelType::I64 => PixelType::I128,
            PixelType::U64 => PixelType::U128,
            PixelType::I128 => PixelType::I128,
            PixelType::U128 => PixelType::U128,
            PixelType::F128 => PixelType::F128,
        };
        if let Some(axis) = axis {
            PyView {
                dtype,
                view: self.view.sum_proj(self.get_ax(axis)?)?,
            }
            .into_bound_py_any(py)
        } else {
            Ok(match self.dtype {
                PixelType::F32 => self.view.sum::<f32>()?.into_pyobject(py)?.into_any(),
                PixelType::F64 => self.view.sum::<f64>()?.into_pyobject(py)?.into_any(),
                PixelType::I64 => self.view.sum::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U64 => self.view.sum::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::I128 => self.view.sum::<i64>()?.into_pyobject(py)?.into_any(),
                PixelType::U128 => self.view.sum::<u64>()?.into_pyobject(py)?.into_any(),
                PixelType::F128 => self.view.sum::<f64>()?.into_pyobject(py)?.into_any(),
                _ => self.view.sum::<i64>()?.into_pyobject(py)?.into_any(),
            })
        }
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
    m.add_class::<PyView>()?;
    m.add_class::<ViewConstructor>()?;

    #[pyfn(m)]
    #[pyo3(name = "download_bioformats")]
    fn py_download_bioformats(gpl_formats: bool) -> PyResult<()> {
        download_bioformats(gpl_formats)?;
        Ok(())
    }

    Ok(())
}
